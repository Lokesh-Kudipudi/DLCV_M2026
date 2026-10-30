"""Kodak synthetic experiment (paper Table IV, cases 1 and 2).

Independent Gaussian noise per RGB channel: case 1 std [15, 10, 20], case 2 [40, 30, 50].
Denoisers are given sigma = 20 and 50 respectively, as in the paper.
"""
import argparse

from common import METHODS, RESULTS, md_table, run, summarize, write_csv
from haartsvd.utils import add_channel_noise, kodak, psnr, save

CASES = {1: ([15, 10, 20], 20), 2: ([40, 30, 50], 50)}

ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=list(METHODS))
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

rows = []
for case, (stds, sigma) in CASES.items():
    for i, (name, clean) in enumerate(kodak()[: args.limit]):
        noisy = add_channel_noise(clean, stds, seed=1000 * case + i)
        if i == 22:  # keep one example for figures
            save(noisy, RESULTS / "images" / f"kodak{case}_{name}_noisy.png")
            save(clean, RESULTS / "images" / f"kodak{case}_{name}_clean.png")
        for m in args.methods:
            out, p, s, dt = run(m, noisy, sigma, clean)
            rows.append(dict(case=case, image=name, method=m, psnr=p, ssim=s, time=dt,
                             noisy_psnr=psnr(noisy, clean)))
            print(f"case{case} {name} {m:15s} {p:.2f} {s:.4f} {dt:.1f}s", flush=True)
            if i == 22:
                save(out, RESULTS / "images" / f"kodak{case}_{name}_{m}.png")

write_csv(rows, RESULTS / "kodak_per_image.csv")
summ = summarize(rows, ["case", "method"])
write_csv(summ, RESULTS / "kodak_summary.csv")
print(md_table(summ, ["case", "method", "psnr", "ssim", "time", "n"],
               {"psnr": "{:.2f}", "ssim": "{:.4f}", "time": "{:.2f}"}))
