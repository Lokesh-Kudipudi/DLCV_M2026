"""PyTorch (MPS / CUDA / CPU) path for the shared-basis variants (global or reused t-SVD bases with a
fixed group transform).

With shared bases the t-SVD projection (orthonormal channel FFT + per-slice U^H P V) is one fixed
real orthogonal map on vectorised patches, so it becomes a single dense matrix M (c*ps^2 square).
Complex coefficients of non-self-conjugate slices are stored as (sqrt2*Re, sqrt2*Im) pairs, which
keeps M orthogonal and lets each pair be hard-thresholded on its complex magnitude, exactly as
in the NumPy path. MPS has no complex support, so this real form is what makes the GPU usable.
"""
import numpy as np
import torch

from .transforms import channel_fft, project


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def real_patch_transform(U, V, c, ps):
    """Return M (D x D, D = c*ps^2), and index arrays (real, re, im) of its output coordinates."""
    D = c * ps * ps
    E = np.eye(D).reshape(D, c, ps, ps)
    S = project(channel_fft(E), U.astype(np.complex128), V.astype(np.complex128))  # (D, s, ps, ps)
    cols, real_idx, re_idx, im_idx, n = [], [], [], [], 0
    m = ps * ps
    for j in range(S.shape[1]):
        Sj = S[:, j].reshape(D, m)
        self_conj = j == 0 or (c % 2 == 0 and j == c // 2)
        if self_conj:
            cols.append(Sj.real)
            real_idx += range(n, n + m)
            n += m
        else:
            cols += [np.sqrt(2) * Sj.real, np.sqrt(2) * Sj.imag]
            re_idx += range(n, n + m)
            im_idx += range(n + m, n + 2 * m)
            n += 2 * m
    M = np.concatenate(cols, axis=1).T  # coefficients = M @ patch
    err = np.abs(M @ M.T - np.eye(D)).max()
    if err > 1e-6:
        raise ValueError(f"patch transform is not orthogonal (err {err:.2e}); "
                         "self-conjugate slices need real bases")
    return M, np.array(real_idx), np.array(re_idx, dtype=int), np.array(im_idx, dtype=int)


def filter_aggregate(img, gy, gx, U, V, Hm, tau, ps, weights, batch, device):
    """Filter all groups and aggregate on `device`. Returns (H, W, c) float32 image and stage timings."""
    H, W, c = img.shape
    dev = pick_device(device)
    M, ri, rei, imi = real_patch_transform(U, V, c, ps)
    f32 = dict(dtype=torch.float32, device=dev)
    Mt = torch.as_tensor(M, **f32)
    T = torch.as_tensor(Hm, **f32) if Hm is not None else None
    rei_t = torch.as_tensor(rei, device=dev)
    imi_t = torch.as_tensor(imi, device=dev)

    a, b, ch = np.meshgrid(np.arange(ps), np.arange(ps), np.arange(c), indexing="ij")
    tmpl = torch.as_tensor(((a * W + b) * c + ch).transpose(2, 0, 1).ravel(), device=dev)
    flat = torch.as_tensor(img.ravel(), **f32)
    num = torch.zeros(H * W * c, **f32)
    den = torch.zeros(H * W * c, **f32)
    base = torch.as_tensor(((gy.astype(np.int64) * W + gx) * c), device=dev)  # (n, K)

    for s in range(0, base.shape[0], batch):
        idx = base[s:s + batch, :, None] + tmpl  # (B, K, D)
        G = flat[idx]
        C = G @ Mt.T
        if T is not None:
            C = T @ C
        mag = C.abs()
        if len(rei):
            pair = torch.sqrt((C[..., rei_t] ** 2 + C[..., imi_t] ** 2) * 0.5)
            mag[..., rei_t] = pair
            mag[..., imi_t] = pair
        keep = mag >= tau
        C = C * keep
        if T is not None:
            C = T.T @ C
        G = C @ Mt
        if weights == "sparse":
            w = 1.0 / keep.sum(dim=(1, 2)).clamp(min=1).to(torch.float32)
            wb = w[:, None, None].expand_as(G)
        else:
            wb = torch.ones_like(G)
        num.index_add_(0, idx.reshape(-1), (G * wb).reshape(-1))
        den.index_add_(0, idx.reshape(-1), wb.reshape(-1))

    out = torch.where(den > 0, num / den.clamp(min=1e-12), flat)
    return out.reshape(H, W, c).cpu().numpy()
