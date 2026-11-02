"""One-step Haar-tSVD denoiser (paper Algorithm 1, Eqs. (19)-(21)) and its ablation variants."""
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .matching import block_match, reference_grid
from .transforms import (channel_fft, channel_ifft, haar_matrix, learn_local_bases,
                         learn_tsvd_bases, project, unproject)


@dataclass
class Params:
    ps: int = 8              # patch size
    K: int = 32              # patches per group (power of 2 for Haar)
    sr: int = 20             # search radius: window is (2*sr+1)^2, as in the authors' code (SR=20)
    step: int = 4            # stride between reference patches (authors' N_step)
    gamma: float = 1.2       # GCP weight in Eq. (9)
    tau_scale: float = 1.0   # multiplies tau = sigma * sqrt(2 log(c K ps^2))
    group: str = "haar"      # group-level transform: "haar" | "pca" | "none"
    bases: str = "global"    # patch-level bases: "global" | "local" | "reused" (pass UV=...)
    matching: str = "gcp"    # "gcp" | "avg" | "green"
    weights: str = "uniform"  # aggregation: "uniform" (paper) | "sparse" (1 / #nonzero coeffs)
    batch: int = 2048
    device: str = "auto"     # "auto" (cuda > mps > cpu) | "cuda" | "mps" | "cpu" (torch) | "numpy"
    matcher: str = "auto"    # block matching: "auto" (torch on cuda, numba otherwise) | "numba" | "torch"


@dataclass
class Result:
    image: np.ndarray
    timings: dict = field(default_factory=dict)
    UV: tuple = None


def threshold(sigma, p, c):
    return p.tau_scale * sigma * np.sqrt(2.0 * np.log(c * p.K * p.ps ** 2))


def learn_global_bases(img, p):
    """Globally learned t-SVD bases from all reference patches of `img` (H, W, c)."""
    H, W, _ = img.shape
    ry, rx = reference_grid(H, W, p.ps, p.step)
    P = sliding_window_view(img.astype(np.float32), (p.ps, p.ps), axis=(0, 1))[ry, rx]
    return learn_tsvd_bases(P)


def _filter_batch(G, sigma_tau, p, U, V, Hm):
    """G: (B, K, c, ps, ps) noisy groups -> denoised groups and per-group nonzero counts."""
    c = G.shape[2]
    F = channel_fft(G)  # (B, K, s, ps, ps)
    if p.bases == "local":
        U, V = learn_local_bases(F)          # (B, s, ps, ps)
        U, V = U[:, None], V[:, None]
    S = project(F, U, V)
    B, K = S.shape[:2]
    flat = S.reshape(B, K, -1)

    if p.group == "haar":
        T = Hm[None]
    elif p.group == "pca":
        # Per-group PCA along the grouping dimension. The Gram matrix is invariant to the
        # orthonormal patch transform, so compute it from the spatial patches.
        X = G.reshape(B, K, -1).astype(np.float64)
        _, E = np.linalg.eigh(X @ np.swapaxes(X, 1, 2))
        T = np.swapaxes(E[..., ::-1], 1, 2)  # rows = principal directions
    else:
        T = np.eye(K)[None]
    coef = T @ flat
    keep = np.abs(coef) >= sigma_tau
    coef = coef * keep
    flat = np.swapaxes(T, 1, 2) @ coef  # all group transforms are orthonormal

    # rfft slice s>0 stands for two conjugate slices when c is odd; count both for weights
    nz = keep.reshape(B, K, *S.shape[2:]).sum(axis=(1, 3, 4)).astype(np.float64)
    nnz = nz[:, 0] + 2 * nz[:, 1:].sum(axis=1) if c > 1 else nz[:, 0]
    out = channel_ifft(unproject(flat.reshape(S.shape), U, V), c)
    return out.astype(np.float32), nnz


def _resolve(device):
    from .gpu import pick_device
    return pick_device(device).type


def haar_tsvd(noisy, sigma, params=None, UV=None):
    """Denoise an (H, W, c) image in [0, 255] given noise level sigma (same scale).

    UV: optional pre-learned (U, V) bases ("globally-reused" variant).
    """
    p = params or Params()
    if p.group == "haar" and p.K & (p.K - 1):
        raise ValueError("Haar group transform needs K to be a power of two")
    img = noisy.astype(np.float32)
    H, W, c = img.shape
    t = {}

    dev = None if p.device == "numpy" else _resolve(p.device)
    t0 = time.perf_counter()
    ry, rx = reference_grid(H, W, p.ps, p.step)
    if p.matcher == "torch" or (p.matcher == "auto" and dev == "cuda"):
        from .torch_ops import block_match as torch_block_match
        gy, gx = torch_block_match(img, ry, rx, p.ps, p.sr, p.K, p.gamma, p.matching,
                                   dev or "cpu")
    else:
        gy, gx = block_match(img, ry, rx, p.ps, p.sr, p.K, p.gamma, p.matching)
    t["search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if p.bases == "local":
        U = V = None
    elif UV is not None:
        U, V = UV
    else:
        U, V = learn_global_bases(img, p)
    if U is not None:
        U, V = U.astype(np.complex64), V.astype(np.complex64)
    t["bases"] = time.perf_counter() - t0

    Hm = haar_matrix(p.K) if p.group == "haar" else None
    tau = threshold(sigma, p, c)

    shared = p.bases != "local" and p.group in ("haar", "none")
    if dev is not None and (shared or dev != "mps"):  # MPS has no complex tensors
        t0 = time.perf_counter()
        if shared:
            from .gpu import filter_aggregate
            out = filter_aggregate(img, gy, gx, U, V, Hm, tau, p.ps, p.weights, 4 * p.batch, dev)
        else:
            from .torch_ops import filter_aggregate_complex
            out = filter_aggregate_complex(img, gy, gx, U, V, Hm, tau, p, dev)
        t["filter+aggregate"] = time.perf_counter() - t0
        t["total"] = sum(t.values())
        return Result(out, t, (U, V))

    win = sliding_window_view(img, (p.ps, p.ps), axis=(0, 1))  # (H', W', c, ps, ps)
    a, b, ch = np.meshgrid(np.arange(p.ps), np.arange(p.ps), np.arange(c), indexing="ij")
    tmpl = ((a * W + b) * c + ch).transpose(2, 0, 1).ravel()  # offsets in (c, ps, ps) order
    num = np.zeros(H * W * c)
    den = np.zeros(H * W * c)

    t["filter"] = t["aggregate"] = 0.0
    for s in range(0, len(ry), p.batch):
        y, x = gy[s:s + p.batch], gx[s:s + p.batch]
        t0 = time.perf_counter()
        est, nnz = _filter_batch(win[y, x], tau, p, U, V, Hm)
        t1 = time.perf_counter()
        w = 1.0 / np.maximum(nnz, 1.0) if p.weights == "sparse" else np.ones(len(y))
        idx = ((y * W + x) * c)[..., None] + tmpl  # (B, K, c*ps*ps)
        wb = np.broadcast_to(w[:, None, None], idx.shape)
        num += np.bincount(idx.ravel(), (est.reshape(idx.shape) * wb).ravel(), H * W * c)
        den += np.bincount(idx.ravel(), wb.ravel(), H * W * c)
        t["filter"] += t1 - t0
        t["aggregate"] += time.perf_counter() - t1

    covered = den > 0
    out = img.ravel().astype(np.float64).copy()
    out[covered] = num[covered] / den[covered]
    t["total"] = sum(t.values())
    return Result(out.reshape(H, W, c).astype(np.float32), t, (U, V))
