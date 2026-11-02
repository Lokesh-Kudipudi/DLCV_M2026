"""Check the selected torch device and that every backend matches the NumPy reference on one image.

    HAARTSVD_DEVICE=cuda python scripts/check_device.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from haartsvd import Params, haar_tsvd  # noqa: E402
from haartsvd.gpu import pick_device  # noqa: E402
from haartsvd.utils import add_channel_noise, kodak, psnr  # noqa: E402

dev = pick_device(os.environ.get("HAARTSVD_DEVICE", "auto")).type
print("torch device:", dev)
_, clean = kodak()[0]
clean = clean[:256, :256]
noisy = add_channel_noise(clean, [15, 10, 20], seed=0)

ok = True
for kw in [{}, dict(bases="local"), dict(group="pca"), dict(bases="local", group="pca")]:
    ref = haar_tsvd(noisy, 20, Params(tau_scale=0.45, device="numpy", **kw)).image
    for matcher in ["numba", "torch"]:
        out = haar_tsvd(noisy, 20, Params(tau_scale=0.45, device=dev, matcher=matcher, **kw)).image
        d = abs(psnr(out, clean) - psnr(ref, clean))
        ok &= d < 0.05
        print(f"{str(kw or 'global+haar'):40s} matcher={matcher:5s} "
              f"PSNR {psnr(out, clean):.3f} vs numpy {psnr(ref, clean):.3f}  |dPSNR|={d:.4f}")
print("OK" if ok else "MISMATCH: check the backend before running experiments")
sys.exit(0 if ok else 1)
