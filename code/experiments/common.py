import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haartsvd import Params, haar_tsvd  # noqa: E402
from haartsvd.utils import psnr, ssim  # noqa: E402

RESULTS = Path(os.environ.get("HAARTSVD_RESULTS", ROOT / "results"))  # override for smoke tests
TAU_SCALE = 0.45  # tuned on kodim01 / kodim13 (Kodak case 1 and 2), see README
# torch device for Haar-tSVD: auto (cuda > mps > cpu) | cuda | mps | cpu | numpy
DEVICE = os.environ.get("HAARTSVD_DEVICE", "auto")


def haar(**kw):
    kw.setdefault("device", DEVICE)
    return Params(tau_scale=TAU_SCALE, **kw)


METHODS = {
    "Haar-tSVD": lambda y, s: haar_tsvd(y, s, haar()).image,
    # t-SVD baseline with locally learned bases and per-group PCA (MSt-SVD-style)
    "Local-tSVD+PCA": lambda y, s: haar_tsvd(y, s, haar(bases="local", group="pca")).image,
}


def cbm3d(y, s):
    import bm3d
    return (bm3d.bm3d_rgb(y / 255.0, s / 255.0) * 255.0).astype(np.float32)


METHODS["CBM3D"] = cbm3d


def run(method, noisy, sigma, clean):
    t0 = time.perf_counter()
    out = METHODS[method](noisy, sigma)
    dt = time.perf_counter() - t0
    return out, psnr(out, clean), ssim(out, clean), dt


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows, keys, metrics=("psnr", "ssim", "time")):
    """Average metrics over rows grouped by `keys`."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for k, rs in groups.items():
        d = dict(zip(keys, k))
        for m in metrics:
            d[m] = float(np.mean([float(r[m]) for r in rs]))
        d["n"] = len(rs)
        out.append(d)
    return out


def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(
            fmt.get(c, "{}").format(r[c]) if not isinstance(r[c], str) else r[c] for c in cols) + " |")
    return "\n".join(lines)
