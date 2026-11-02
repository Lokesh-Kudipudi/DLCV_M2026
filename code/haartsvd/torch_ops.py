"""Device-generic PyTorch kernels used on CUDA (and optionally MPS / CPU).

* `block_match`: GCP block matching on the GPU. Every search offset is one shifted image; patch
  SSDs for all positions come from a box filter, then the K best offsets per reference are kept.
* `filter_aggregate_complex`: the full complex-domain filter (channel FFT, t-SVD projection, group
  transform, hard threshold, inverse, aggregation) for every variant, including locally learned
  bases and per-group PCA (batched `eigh`). Needs complex tensors, so it runs on CUDA/CPU, not MPS.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .gpu import pick_device


def block_match(img, ref_y, ref_x, ps, sr, K, gamma=1.2, matching="gcp", device="auto", chunk=96):
    dev = pick_device(device)
    x = torch.as_tensor(img, dtype=torch.float32, device=dev)
    H, W, _ = x.shape
    chans = {"green": x[..., 1], "avg": x.mean(dim=2)}
    ry = torch.as_tensor(ref_y, dtype=torch.long, device=dev)
    rx = torch.as_tensor(ref_x, dtype=torch.long, device=dev)

    if matching == "gcp":
        P = x.unfold(0, ps, 1).unfold(1, ps, 1)[ry, rx]  # (n, c, ps, ps)
        nrm = P.pow(2).sum(dim=(2, 3)).sqrt()
        use_green = nrm[:, 1] >= torch.maximum(nrm[:, 0], nrm[:, 2]) / gamma
    else:
        use_green = torch.full_like(ry, matching == "green", dtype=torch.bool)

    offs = torch.stack(torch.meshgrid(torch.arange(-sr, sr + 1), torch.arange(-sr, sr + 1),
                                      indexing="ij"), -1).reshape(-1, 2).to(dev)
    pads = {k: F.pad(v[None, None], (sr, sr, sr, sr))[0, 0] for k, v in chans.items()}

    def ssd_at_refs(name, o):
        """Patch SSD between each reference and its candidate at every offset in `o` -> (n, O)."""
        pad = pads[name]
        shifted = torch.stack([pad[sr + dy: sr + dy + H, sr + dx: sr + dx + W] for dy, dx in o.tolist()])
        sq = (chans[name][None] - shifted).pow(2)[:, None]  # (O, 1, H, W)
        ssd = F.avg_pool2d(sq, ps, stride=1)[:, 0] * (ps * ps)  # (O, H-ps+1, W-ps+1)
        return ssd[:, ry, rx].T

    need_green, need_avg = bool(use_green.any()), bool((~use_green).any())
    dists = []
    for s in range(0, offs.shape[0], chunk):
        o = offs[s:s + chunk]
        if need_green and need_avg:
            d = torch.where(use_green[:, None], ssd_at_refs("green", o), ssd_at_refs("avg", o))
        else:
            d = ssd_at_refs("green" if need_green else "avg", o)
        dists.append(d)
    D = torch.cat(dists, dim=1)  # (n, n_offsets)
    cy = ry[:, None] + offs[None, :, 0]
    cx = rx[:, None] + offs[None, :, 1]
    valid = (cy >= 0) & (cy <= H - ps) & (cx >= 0) & (cx <= W - ps)
    D = torch.where(valid, D, torch.full_like(D, float("inf")))
    best = torch.topk(D, K, dim=1, largest=False).indices
    gy = torch.gather(cy, 1, best).to(torch.int32).cpu().numpy()
    gx = torch.gather(cx, 1, best).to(torch.int32).cpu().numpy()
    return gy, gx


def _flip(e):
    return torch.flip(e, dims=[-1])


def filter_aggregate_complex(img, gy, gx, U, V, Hm, tau, p, device="auto"):
    dev = pick_device(device)
    H, W, c = img.shape
    ps, K = p.ps, p.K
    f32 = dict(dtype=torch.float32, device=dev)
    flat = torch.as_tensor(img.ravel(), **f32)
    num = torch.zeros(H * W * c, **f32)
    den = torch.zeros(H * W * c, **f32)
    a, b, ch = np.meshgrid(np.arange(ps), np.arange(ps), np.arange(c), indexing="ij")
    tmpl = torch.as_tensor(((a * W + b) * c + ch).transpose(2, 0, 1).ravel(), device=dev)
    base = torch.as_tensor((gy.astype(np.int64) * W + gx) * c, device=dev)
    if U is not None:
        Ug = torch.as_tensor(U, dtype=torch.complex64, device=dev)
        Vg = torch.as_tensor(V, dtype=torch.complex64, device=dev)
    T_haar = torch.as_tensor(Hm, dtype=torch.complex64, device=dev) if Hm is not None else None

    for s in range(0, base.shape[0], p.batch):
        idx = base[s:s + p.batch, :, None] + tmpl  # (B, K, c*ps*ps)
        B = idx.shape[0]
        G = flat[idx].reshape(B, K, c, ps, ps)
        Fq = torch.fft.rfft(G, dim=2, norm="ortho")  # (B, K, sl, ps, ps)
        if p.bases == "local":
            C_row = torch.einsum("bksxy,bkszy->bsxz", Fq, Fq.conj())
            C_col = torch.einsum("bksyx,bksyz->bsxz", Fq.conj(), Fq)
            Uu = _flip(torch.linalg.eigh(C_row).eigenvectors)[:, None]
            Vv = _flip(torch.linalg.eigh(C_col).eigenvectors)[:, None]
        else:
            Uu, Vv = Ug, Vg
        S = Uu.mH @ Fq @ Vv
        fl = S.reshape(B, K, -1)
        if p.group == "haar":
            T = T_haar[None]
        elif p.group == "pca":
            X = G.reshape(B, K, -1)
            E = _flip(torch.linalg.eigh(X @ X.transpose(1, 2)).eigenvectors)
            T = E.transpose(1, 2).to(torch.complex64)
        else:
            T = None
        coef = T @ fl if T is not None else fl
        keep = coef.abs() >= tau
        coef = coef * keep
        fl = T.transpose(1, 2) @ coef if T is not None else coef
        out = torch.fft.irfft(Uu @ fl.reshape(S.shape) @ Vv.mH, n=c, dim=2, norm="ortho")
        if p.weights == "sparse":
            nz = keep.reshape(S.shape).sum(dim=(1, 3, 4)).to(torch.float32)  # (B, sl)
            nnz = nz[:, 0] + 2 * nz[:, 1:].sum(dim=1) if c > 1 else nz[:, 0]
            wb = (1.0 / nnz.clamp(min=1))[:, None, None].expand(idx.shape)
        else:
            wb = torch.ones(idx.shape, **f32)
        num.index_add_(0, idx.reshape(-1), (out.reshape(idx.shape) * wb).reshape(-1))
        den.index_add_(0, idx.reshape(-1), wb.reshape(-1))

    res = torch.where(den > 0, num / den.clamp(min=1e-12), flat)
    return res.reshape(H, W, c).cpu().numpy()
