"""PolyU real-noise experiment (100 cropped 512x512 pairs).

The paper's CNN noise estimator is replaced by the classical MAD estimator (report Sec. 5.3).
Real sRGB noise is spatially correlated, so MAD on the finest band underestimates the effective
sigma; a single scale factor kappa (sigma = kappa * MAD) is tuned on the first 10 images
(validation) and results are reported on the remaining 90 (test).
An oracle-sigma run (best sigma per image from a grid) gives an upper bound.
"""
import argparse

import numpy as np

from common import METHODS, RESULTS, haar, md_table, run, summarize, write_csv
from haartsvd import haar_tsvd
from haartsvd.utils import estimate_sigma, polyu, psnr, save, ssim

KAPPAS = [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0]  # widened twice from {1.0..4.0}; see report 6.2
ORACLE = [2, 3, 4, 5, 6, 8, 10, 12, 15]
N_VAL = 10

ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=list(METHODS))
ap.add_argument("--no-oracle", action="store_true")
args = ap.parse_args()

data = list(polyu())
val, test = data[:N_VAL], data[N_VAL:]

# 1) tune kappa on validation images
tune = []
for k in KAPPAS:
    ps = []
    for name, noisy, clean in val:
        out = haar_tsvd(noisy, k * estimate_sigma(noisy), haar()).image
        ps.append(psnr(out, clean))
    tune.append(dict(kappa=k, psnr=float(np.mean(ps))))
    print(f"kappa {k}: val PSNR {np.mean(ps):.2f}", flush=True)
write_csv(tune, RESULTS / "polyu_kappa_tuning.csv")
kappa = max(tune, key=lambda r: r["psnr"])["kappa"]
print("chosen kappa", kappa)

# 2) test set: all methods with sigma = kappa * MAD, plus oracle sigma for Haar-tSVD
rows = []
for i, (name, noisy, clean) in enumerate(test):
    mad = estimate_sigma(noisy)
    sigma = kappa * mad
    base = dict(image=name, mad=mad, sigma=sigma, noisy_psnr=psnr(noisy, clean))
    for m in args.methods:
        out, p, s, dt = run(m, noisy, sigma, clean)
        rows.append(dict(base, method=m, psnr=p, ssim=s, time=dt))
        print(f"{name} {m:15s} sigma={sigma:.2f} {p:.2f} {s:.4f} {dt:.1f}s", flush=True)
        if i < 3:
            save(out, RESULTS / "images" / f"polyu_{name}_{m}.png")
    if i < 3:
        save(noisy, RESULTS / "images" / f"polyu_{name}_noisy.png")
        save(clean, RESULTS / "images" / f"polyu_{name}_clean.png")
    if args.no_oracle:
        continue
    best = max(((psnr(o := haar_tsvd(noisy, sg, haar()).image, clean), ssim(o, clean), sg)
                for sg in ORACLE), key=lambda t: t[0])
    rows.append(dict(base, method="Haar-tSVD (oracle sigma)", psnr=best[0], ssim=best[1], time=0.0))

write_csv(rows, RESULTS / "polyu_per_image.csv")
summ = summarize(rows, ["method"])
summ.insert(0, dict(method="Noisy input", psnr=float(np.mean([r["noisy_psnr"] for r in rows])),
                    ssim=float("nan"), time=0.0, n=len(test)))
write_csv(summ, RESULTS / "polyu_summary.csv")
print(md_table(summ, ["method", "psnr", "ssim", "time", "n"],
               {"psnr": "{:.2f}", "ssim": "{:.4f}", "time": "{:.2f}"}))
