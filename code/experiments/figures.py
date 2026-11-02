"""Figures for the README: visual comparisons and sigma-sensitivity plot."""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import RESULTS  # noqa: E402
from haartsvd.utils import load, psnr  # noqa: E402

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#555",
                     "ytick.color": "#555"})


def comparison(prefix, crop, out):
    panels = [p for p in ["clean", "noisy", "CBM3D", "Local-tSVD+PCA", "Haar-tSVD"]
              if (RESULTS / "images" / f"{prefix}_{p}.png").exists()]
    clean = load(RESULTS / "images" / f"{prefix}_clean.png")
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.3))
    for ax, p in zip(axes, panels):
        im = load(RESULTS / "images" / f"{prefix}_{p}.png")
        (y0, y1), (x0, x1) = crop
        ax.imshow(im[y0:y1, x0:x1].astype(np.uint8))
        title = p if p == "clean" else f"{p}\n{psnr(im, clean):.2f} dB"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def sigma_plot():
    rows = list(csv.DictReader(open(RESULTS / "ablation_sigma.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, case in zip(axes, ["1", "2"]):
        for group, color, label in [("haar", BLUE, "Haar (fixed)"), ("pca", ORANGE, "PCA (per group)")]:
            rs = sorted((r for r in rows if r["case"] == case and r["group"] == group),
                        key=lambda r: float(r["sigma_factor"]))
            x = [float(r["sigma_factor"]) for r in rs]
            y = [float(r["psnr"]) for r in rs]
            ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=label)
        ax.set_title(f"Kodak case {case}", fontsize=10)
        ax.set_xlabel("sigma given / nominal sigma")
        ax.grid(axis="y", color="#e5e5e5")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "sigma_sensitivity.png", dpi=130)
    plt.close(fig)


comparison("kodak2_kodim23", ((100, 356), (250, 506)), "kodak_case2_kodim23.png")
comparison("kodak1_kodim23", ((100, 356), (250, 506)), "kodak_case1_kodim23.png")
names = sorted(p.name[len("polyu_"):-len("_clean.png")] for p in (RESULTS / "images").glob("polyu_*_clean.png"))
if names:
    comparison(f"polyu_{names[0]}", ((128, 384), (128, 384)), "polyu_example.png")
sigma_plot()
print("figures written to", FIG)
