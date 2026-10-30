"""Green-channel-prior (GCP) block matching, paper Eq. (9)."""
import numpy as np
from numba import njit, prange


def reference_grid(H, W, ps, step):
    """Top-left corners of reference patches; the last row/column is always covered."""
    ys = list(range(0, H - ps + 1, step))
    xs = list(range(0, W - ps + 1, step))
    if ys[-1] != H - ps:
        ys.append(H - ps)
    if xs[-1] != W - ps:
        xs.append(W - ps)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.ravel().astype(np.int32), xx.ravel().astype(np.int32)


def green_mode(img, ref_y, ref_x, ps, gamma):
    """True where the reference patch's green norm dominates: ||G|| >= max(||R||, ||B||) / gamma."""
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(img, (ps, ps), axis=(0, 1))[ref_y, ref_x]  # (n, 3, ps, ps)
    norms = np.sqrt((win.astype(np.float64) ** 2).sum(axis=(2, 3)))  # (n, 3)
    return norms[:, 1] >= np.maximum(norms[:, 0], norms[:, 2]) / gamma


@njit(parallel=True, cache=True)
def _search(green, avg, ref_y, ref_x, use_green, ps, sr, K):
    H, W = green.shape
    n = ref_y.shape[0]
    out_y = np.empty((n, K), np.int32)
    out_x = np.empty((n, K), np.int32)
    side = 2 * sr + 1
    for r in prange(n):
        y0 = ref_y[r]
        x0 = ref_x[r]
        ch = green if use_green[r] else avg
        top = max(y0 - sr, 0)
        bot = min(y0 + sr, H - ps)
        lft = max(x0 - sr, 0)
        rgt = min(x0 + sr, W - ps)
        m = (bot - top + 1) * (rgt - lft + 1)
        dist = np.empty(m, np.float32)
        cy = np.empty(m, np.int32)
        cx = np.empty(m, np.int32)
        t = 0
        for y in range(top, bot + 1):
            for x in range(lft, rgt + 1):
                d = np.float32(0.0)
                for a in range(ps):
                    for b in range(ps):
                        diff = ch[y0 + a, x0 + b] - ch[y + a, x + b]
                        d += diff * diff
                dist[t] = d
                cy[t] = y
                cx[t] = x
                t += 1
        order = np.argsort(dist, kind="mergesort")  # stable: reference patch (d=0) stays first
        for k in range(K):
            j = order[min(k, m - 1)]
            out_y[r, k] = cy[j]
            out_x[r, k] = cx[j]
    return out_y, out_x


def block_match(img, ref_y, ref_x, ps, sr, K, gamma=1.2, matching="gcp"):
    """Return (n_ref, K) arrays of top-left corners of the K most similar patches.

    matching: "gcp" (paper Eq. 9), "avg" (channel average only) or "green" (green only).
    """
    img = img.astype(np.float32)
    green = np.ascontiguousarray(img[..., 1])
    avg = np.ascontiguousarray(img.mean(axis=2))
    if matching == "gcp":
        use_green = green_mode(img, ref_y, ref_x, ps, gamma)
    else:
        use_green = np.full(ref_y.shape, matching == "green")
    return _search(green, avg, ref_y, ref_x, use_green, ps, sr, K)
