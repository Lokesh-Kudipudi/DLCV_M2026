"""Ablations on a Kodak subset (report Sec. 5.4, RQ1-RQ3).

1. Group transform (Haar / per-group PCA / none) x patch bases (global / local / reused).
2. Sensitivity to ps, K, search radius, matching rule (case 1).
3. Sensitivity to errors in sigma, Haar vs PCA (cases 1 and 2).
4. Stage-wise runtime on a 512x512x3 image.
"""
import time

import numpy as np

from common import DEVICE, RESULTS, cbm3d, haar, md_table, summarize, write_csv
from haartsvd.gpu import pick_device
from haartsvd import haar_tsvd, learn_global_bases
from haartsvd.utils import add_channel_noise, kodak, psnr, ssim

CASES = {1: ([15, 10, 20], 20), 2: ([40, 30, 50], 50)}
SUBSET = ["kodim03", "kodim05", "kodim08", "kodim15", "kodim19", "kodim23"]
FMT = {"psnr": "{:.2f}", "ssim": "{:.4f}", "time": "{:.2f}"}

imgs = dict(kodak())
# "globally-reused" bases: learned once from noisy kodim01 (not in the subset), reused everywhere
reused = {c: learn_global_bases(add_channel_noise(imgs["kodim01"], s, seed=7), haar())
          for c, (s, _) in CASES.items()}


def evaluate(tag, cases, make_params, sigma_factor=1.0, use_reused=False):
    rows = []
    for case in cases:
        stds, sigma = CASES[case]
        for i, name in enumerate(SUBSET):
            clean = imgs[name]
            noisy = add_channel_noise(clean, stds, seed=1000 * case + i)
            r = haar_tsvd(noisy, sigma * sigma_factor, make_params(),
                          UV=reused[case] if use_reused else None)
            rows.append(dict(tag, case=case, image=name, psnr=psnr(r.image, clean),
                             ssim=ssim(r.image, clean), time=r.timings["total"]))
    print(tag, f"{np.mean([r['psnr'] for r in rows]):.2f}", flush=True)
    return rows


report = []

# 1) group transform x bases
rows = []
for group in ["haar", "pca", "none"]:
    for bases in ["global", "local", "reused"]:
        if group == "none" and bases != "global":
            continue
        rows += evaluate(dict(group=group, bases=bases), [1, 2],
                         lambda g=group, b=bases: haar(group=g, bases="global" if b == "reused" else b),
                         use_reused=bases == "reused")
write_csv(rows, RESULTS / "ablation_transforms_per_image.csv")
s = summarize(rows, ["case", "group", "bases"])
write_csv(s, RESULTS / "ablation_transforms.csv")
report.append("## Group transform x t-SVD bases\n\n" + md_table(
    s, ["case", "group", "bases", "psnr", "ssim", "time"], FMT))

# 2) parameter sensitivity (case 1)
rows = []
for ps in [6, 8, 10]:
    rows += evaluate(dict(param="ps", value=str(ps)), [1], lambda ps=ps: haar(ps=ps))
for K in [8, 16, 32, 64]:
    rows += evaluate(dict(param="K", value=str(K)), [1], lambda K=K: haar(K=K))
for sr in [5, 10, 15, 20]:
    rows += evaluate(dict(param="search radius", value=str(sr)), [1], lambda sr=sr: haar(sr=sr))
for m in ["gcp", "avg", "green"]:
    rows += evaluate(dict(param="matching", value=m), [1], lambda m=m: haar(matching=m))
for w in ["uniform", "sparse"]:
    rows += evaluate(dict(param="aggregation", value=w), [1], lambda w=w: haar(weights=w))
write_csv(rows, RESULTS / "ablation_params_per_image.csv")
s = summarize(rows, ["param", "value"])
write_csv(s, RESULTS / "ablation_params.csv")
report.append("## Parameter sensitivity (case 1)\n\n" + md_table(
    s, ["param", "value", "psnr", "ssim", "time"], FMT))

# 3) sigma mismatch, Haar vs PCA
rows = []
for group in ["haar", "pca"]:
    for f in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        rows += evaluate(dict(group=group, sigma_factor=str(f)), [1, 2],
                         lambda g=group: haar(group=g), sigma_factor=f)
write_csv(rows, RESULTS / "ablation_sigma_per_image.csv")
s = summarize(rows, ["case", "group", "sigma_factor"])
write_csv(s, RESULTS / "ablation_sigma.csv")
report.append("## Sensitivity to the noise level (sigma given = factor x nominal)\n\n" + md_table(
    s, ["case", "group", "sigma_factor", "psnr", "ssim"], FMT))

# 4) runtime breakdown on 512x512x3
clean = imgs["kodim23"][:512, 128:640]
noisy = add_channel_noise(clean, CASES[1][0], seed=1)
rows = []
for label, kw in [(f"Haar-tSVD (global, {pick_device(DEVICE).type})", {}), ("Haar-tSVD (global, torch CPU)", dict(device="cpu")),
                  ("Haar-tSVD (global, NumPy CPU)", dict(device="numpy")),
                  ("Global t-SVD + PCA", dict(group="pca")),
                  ("Local t-SVD + Haar", dict(bases="local")),
                  ("Local t-SVD + PCA", dict(bases="local", group="pca"))]:
    haar_tsvd(noisy, 20, haar(**kw))  # warm-up (numba compile cache)
    ts = [haar_tsvd(noisy, 20, haar(**kw)).timings for _ in range(3)]
    m = {k: float(np.mean([t[k] for t in ts])) for k in ts[0]}
    if "filter+aggregate" in m:
        m["filter"], m["aggregate"] = m.pop("filter+aggregate"), float("nan")  # fused on device
    rows.append(dict(method=label, **m))
t0 = time.perf_counter()
cbm3d(noisy, 20)
rows.append(dict(method="CBM3D (bm3d package)", search=float("nan"), bases=float("nan"),
                 filter=float("nan"), aggregate=float("nan"), total=time.perf_counter() - t0))
write_csv(rows, RESULTS / "timing.csv")
report.append("## Runtime on a 512x512x3 image (seconds)\n\n" + md_table(
    rows, ["method", "search", "bases", "filter", "aggregate", "total"],
    {k: "{:.2f}" for k in ["search", "bases", "filter", "aggregate", "total"]}))

(RESULTS / "ablations.md").write_text("\n\n".join(report) + "\n")
print("\n\n".join(report))
