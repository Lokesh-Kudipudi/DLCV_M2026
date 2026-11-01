"""Build markdown tables for report.md from results/*.csv.

Every 'this work' number comes from a CSV; paper numbers are hard-coded and labelled [22].
Usage: python scripts/fill_report_tables.py [--results results] [--out results/report_tables.md]
"""
import argparse
import csv
from pathlib import Path

# Reported in [22], Table IV (PSNR / SSIM) and Table I (PSNR; time in s)
PAPER_IV = {
    "CBM3D": {"k1": "33.51 / 0.893", "k2": "28.90 / 0.780", "polyu": "38.69 / 0.970"},
    "MSt-SVD": {"k1": "33.99 / 0.906", "k2": "29.12 / 0.770", "polyu": "38.85 / 0.971"},
    "Haar-tSVD": {"k1": "34.13 / 0.910", "k2": "29.16 / 0.780", "polyu": "38.78 / 0.969"},
}
OURS_TO_PAPER = {"CBM3D": "CBM3D", "Local-tSVD+PCA": "MSt-SVD", "Haar-tSVD": "Haar-tSVD"}
PAPER_I = [("CC15", "37.86", "37.89", "37.90"), ("HighISO", "40.05", "40.13", "40.15"),
           ("SIDD-val", "34.35", "34.39", "34.38"), ("Time (s)", "6.95", "6.13", "6.02")]


def rd(path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def cell(r, nd=2, ssim=True):
    s = f"{fl(r['psnr']):.{nd}f}"
    return s + (f" / {fl(r['ssim']):.3f}" if ssim and r.get("ssim") not in (None, "", "nan") else "")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = Path(a.results)
    out = []

    def section(title, body):
        out.append(f"### {title}\n\n{body if body else '*missing CSV: run the experiment first*'}\n")

    # 5.1 -------------------------------------------------------------------
    ks, ps = rd(R / "kodak_summary.csv"), rd(R / "polyu_summary.csv")
    pc = rd(R / "polyu_cbm3d_summary.csv")  # CBM3D on PolyU with its own kappa (polyu_extra.py)
    if pc:
        ps = [r for r in ps or [] if r["method"] != "CBM3D"] + pc
    rows = []
    for ours, paper in OURS_TO_PAPER.items():
        g = lambda rs, key, val: next((r for r in rs or [] if r[key] == val), None)
        row = [paper if ours != "Local-tSVD+PCA" else "MSt-SVD / Local-tSVD+PCA"]
        for case, key in (("1", "k1"), ("2", "k2")):
            r = next((r for r in ks or [] if r["method"] == ours and r["case"] == case), None)
            row += [PAPER_IV[paper][key], cell(r) if r else "n/a"]
        r = g(ps, "method", ours)
        row += [PAPER_IV[paper]["polyu"], cell(r) if r else "n/a"]
        rows.append(row)
    section("5.1 Paper Table IV vs this work (PSNR / SSIM)", table(
        ["Method", "Kodak-1 [22]", "Kodak-1 (this work)", "Kodak-2 [22]", "Kodak-2 (this work)",
         "PolyU [22]", "PolyU (this work)"], rows) if ks or ps else None)

    # 5.2 / 6.1 ---------------------------------------------------------------
    at = rd(R / "ablation_transforms.csv")
    if at:
        rows = []
        for case in sorted({r["case"] for r in at}):
            d = {r["bases"]: r for r in at if r["case"] == case and r["group"] == "haar"}
            if all(b in d for b in ("local", "global", "reused")):
                rows.append([f"Kodak subset, case {case}"] + [cell(d[b]) for b in ("local", "global", "reused")]
                            + [f"{fl(d['local']['time']):.2f} / {fl(d['global']['time']):.2f} / {fl(d['reused']['time']):.2f}"])
        rows += [[f"{n} [22]", l, g, r, ""] for n, l, g, r in PAPER_I[:3]]
        rows += [["Time (s) [22]", PAPER_I[3][1], PAPER_I[3][2], PAPER_I[3][3], ""]]
        section("5.2 Locally learned / globally learned / reused bases (Haar group transform)", table(
            ["Dataset", "Local", "Global", "Reused", "Time local / global / reused (s)"], rows))
        section("6.1 Group transform x bases (all rows)", table(
            ["case", "group", "bases", "PSNR", "SSIM", "time (s)", "n"],
            [[r["case"], r["group"], r["bases"], f"{fl(r['psnr']):.2f}", f"{fl(r['ssim']):.4f}",
              f"{fl(r['time']):.2f}", r["n"]] for r in at]))
    else:
        section("5.2 / 6.1 transforms", None)

    # 5.3 -------------------------------------------------------------------
    sg = rd(R / "ablation_sigma.csv")
    if sg:
        rows = []
        for case in sorted({r["case"] for r in sg}):
            for grp in sorted({r["group"] for r in sg}):
                rs = sorted((r for r in sg if r["case"] == case and r["group"] == grp),
                            key=lambda r: fl(r["sigma_factor"]))
                if not rs:
                    continue
                peak = max(fl(r["psnr"]) for r in rs)
                rows.append([case, grp] + [f"{fl(r['psnr']):.2f} ({fl(r['psnr']) - peak:+.2f})" for r in rs])
        facs = sorted({fl(r["sigma_factor"]) for r in sg})
        section("5.3 Sensitivity to sigma: PSNR (drop from own peak) by factor", table(
            ["case", "group"] + [f"f={f:g}" for f in facs], rows))
    else:
        section("5.3 sigma sensitivity", None)

    # 6.2 -------------------------------------------------------------------
    kt = rd(R / "polyu_kappa_tuning.csv")
    body = None
    if kt:
        best = max(kt, key=lambda r: fl(r["psnr"]))
        body = table(["kappa", "val PSNR"], [[r["kappa"], f"{fl(r['psnr']):.2f}"] for r in kt])
        body += f"\n\nChosen kappa = {best['kappa']}" + (
            "  **(edge of the grid: widen KAPPAS and rerun)**" if best is kt[0] or best is kt[-1] else "")
    section("6.2 PolyU kappa tuning (validation = first 10 images)", body)
    kc = rd(R / "polyu_cbm3d_kappa_tuning.csv")
    if kc:
        best = max(kc, key=lambda r: fl(r["psnr"]))
        body = table(["kappa", "val PSNR (CBM3D)"], [[r["kappa"], f"{fl(r['psnr']):.2f}"] for r in kc])
        body += f"\n\nChosen kappa for CBM3D = {best['kappa']}" + (
            "  **(edge of the grid)**" if best is kc[0] or best is kc[-1] else "")
        section("6.2 PolyU kappa tuning for CBM3D", body)
    if ps:
        body = table(["method", "PSNR", "SSIM", "time (s)", "n"],
                     [[r["method"], f"{fl(r['psnr']):.2f}", f"{fl(r['ssim']):.4f}",
                       "-" if "oracle" in r["method"] or r["method"] == "Noisy input" else f"{fl(r['time']):.2f}",
                       r["n"]] for r in ps])
        e = next((r for r in ps if r["method"] == "Haar-tSVD"), None)
        o = next((r for r in ps if "oracle" in r["method"]), None)
        if e and o:
            body += f"\n\nOracle-sigma gap (oracle - estimated) = {fl(o['psnr']) - fl(e['psnr']):.2f} dB"
        ox = rd(R / "polyu_oracle_ext.csv")
        if e and ox:
            m = sum(fl(r["psnr"]) for r in ox) / len(ox)
            n = sum(r["best_sigma"] != "<=15" for r in ox)
            body += (f"\n\nExtended oracle grid (adds 18-32): {m:.2f} dB, gap {m - fl(e['psnr']):.2f} dB; "
                     f"best sigma above 15 on {n}/{len(ox)} images")
        section("6.2 PolyU test set (90 images)", body)

    # 6.3 / 6.4 ---------------------------------------------------------------
    tm = rd(R / "timing.csv")
    section("6.3 Runtime on a 512x512x3 image (s; empty = not recorded or fused)", table(
        ["method", "search", "bases", "filter", "aggregate", "total"],
        [[r["method"]] + ["" if r[k] in ("", "nan") else f"{fl(r[k]):.2f}"
                          for k in ("search", "bases", "filter", "aggregate", "total")] for r in tm]) if tm else None)
    ap_ = rd(R / "ablation_params.csv")
    section("6.4 Parameter sensitivity (case 1)", table(
        ["param", "value", "PSNR", "SSIM", "time (s)"],
        [[r["param"], r["value"], f"{fl(r['psnr']):.2f}", f"{fl(r['ssim']):.4f}", f"{fl(r['time']):.2f}"]
         for r in ap_]) if ap_ else None)

    # Final phase: noise estimators and SIDD (noise_est.py, sidd.py) --------------------
    nk = rd(R / "noise_est_kodak.csv")
    if nk:
        rows = []
        for case in sorted({r["case"] for r in nk}):
            for est in ("mad", "pca"):
                rs = [r for r in nk if r["case"] == case and r["estimator"] == est]
                if rs:
                    true = " / ".join(f"{fl(rs[0][f'true_{c}']):g}" for c in "rgb")
                    mean = " / ".join(f"{sum(fl(r[f'est_{c}']) for r in rs) / len(rs):.2f}" for c in "rgb")
                    rows.append([case, est, true, mean, f"{100 * sum(fl(r['rel_err']) for r in rs) / len(rs):.1f} %"])
        section("F.1 Noise estimators on Kodak (per-channel sigma, mean over 24 images)", table(
            ["case", "estimator", "true R / G / B", "estimated R / G / B", "mean rel. error"], rows))
    pe = rd(R / "polyu_est_summary.csv")
    if pe:
        body = table(["method", "rule", "kappa", "floor", "val PSNR", "test PSNR", "SSIM", "time (s)", "n"],
                     [[r["method"], r["rule"], r["kappa"], r["floor"], f"{fl(r['val_psnr']):.2f}",
                       f"{fl(r['psnr']):.2f}", f"{fl(r['ssim']):.4f}", f"{fl(r['time']):.2f}", r["n"]] for r in pe])
        ox = rd(R / "polyu_oracle_ext.csv")
        if ox:
            body += f"\n\nExtended oracle (Haar-tSVD, same 90 images): {sum(fl(r['psnr']) for r in ox) / len(ox):.2f} dB"
        section("F.2 PolyU test set with each sigma rule (tuned on the 10 validation images)", body)
    pt = rd(R / "polyu_est_tuning.csv")
    if pt:
        section("F.2b PolyU validation tuning of the sigma rules", table(
            ["rule", "kappa", "floor", "val PSNR"],
            [[r["rule"], r["kappa"], r["floor"], f"{fl(r['val_psnr']):.2f}"] for r in pt]))
    sd = rd(R / "sidd_summary.csv")
    if sd:
        section("F.3 SIDD sRGB validation (1280 blocks; nothing tuned on SIDD)", table(
            ["method", "rule", "kappa", "PSNR", "SSIM", "time (s)", "n"],
            [[r["method"], r["rule"], r["kappa"], f"{fl(r['psnr']):.2f}",
              "-" if r["ssim"] in ("", "nan") else f"{fl(r['ssim']):.4f}",
              "-" if "oracle" in r["method"] or r["method"] == "Noisy input" else f"{fl(r['time']):.2f}", r["n"]]
             for r in sd]) + "\n\nReported in [22] (SIDD): CBM3D 34.74 / 0.922, MSt-SVD 34.38 / 0.901, "
            "Haar-tSVD 35.05 / 0.914. CBM3D time is per block with parallel workers, not comparable.")

    text = "\n".join(out)
    dest = Path(a.out) if a.out else R / "report_tables.md"
    dest.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
