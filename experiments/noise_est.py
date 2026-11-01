"""Noise-level estimators for the final phase (report Sec. 3.1 and 8).

--part kodak   Accuracy of the MAD and the PCA estimator of Chen et al. [39] against the known
               per-channel noise std on Kodak (case 1 and 2). CPU only, about a minute.
--part polyu   Compare sigma rules on PolyU with Haar-tSVD, with every parameter tuned on the
               same 10 validation images as polyu.py and results on the other 90 (test):
                 mad        sigma = kappa * MAD                 (re-tuned on a grid that goes past 24)
                 mad+floor  sigma = max(kappa * MAD, floor)     (MAD is 0 on 4 val / 9 test images)
                 pca        sigma = kappa * PCA estimate [39]
               With --cbm3d, CBM3D is also run with the rule that is best on validation, scaled by its
               own factor s (kappa and floor both times s) tuned on validation (parallel CPU workers).

Outputs (results/): noise_est_kodak.csv, polyu_est_tuning.csv, polyu_est_per_image.csv,
polyu_est_summary.csv (+ polyu_est_cbm3d_tuning.csv with --cbm3d)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # CBM3D workers: one thread each

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from common import RESULTS, cbm3d, haar, md_table, write_csv  # noqa: E402
from haartsvd import haar_tsvd  # noqa: E402
from haartsvd.utils import (ESTIMATORS, add_channel_noise, estimate_sigma_channels, kodak,  # noqa: E402
                            polyu, psnr, ssim)

N_VAL = 10  # same split as polyu.py
GRIDS = {
    "mad": [12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0],
    "pca": [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0],
}
FLOORS = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
CBM3D_SCALE = [0.4, 0.5, 0.67, 0.8, 1.0, 1.25]  # CBM3D sigma = s x (Haar-tSVD sigma rule): kappa and floor scaled
CASES = {1: (15, 10, 20), 2: (40, 30, 50)}
DATA = None  # filled in main(); inherited by forked CBM3D workers


def part_kodak():
    rows = []
    for case, sig in CASES.items():
        for i, (name, clean) in enumerate(kodak()):
            noisy = add_channel_noise(clean, sig, 1000 * case + i)  # same seeds as kodak.py
            for est in ("mad", "pca"):
                e = estimate_sigma_channels(noisy, est)
                rows.append(dict(case=case, image=name, estimator=est,
                                 **{f"est_{c}": v for c, v in zip("rgb", e)},
                                 **{f"true_{c}": v for c, v in zip("rgb", sig)},
                                 rel_err=float(np.mean([abs(a - b) / b for a, b in zip(e, sig)]))))
    write_csv(rows, RESULTS / "noise_est_kodak.csv")
    summ = []
    for case in CASES:
        for est in ("mad", "pca"):
            rs = [r for r in rows if r["case"] == case and r["estimator"] == est]
            summ.append(dict(case=case, estimator=est,
                             **{c: float(np.mean([r[f"est_{c}"] for r in rs])) for c in "rgb"},
                             rel_err=float(np.mean([r["rel_err"] for r in rs]))))
    print(md_table(summ, ["case", "estimator", "r", "g", "b", "rel_err"],
                   {"r": "{:.2f}", "g": "{:.2f}", "b": "{:.2f}", "rel_err": "{:.3f}"}))


class Haar:
    """Haar-tSVD outputs cached by (image index, sigma), so repeated sigmas cost nothing."""

    def __init__(self):
        self.cache = {}

    def __call__(self, i, sigma):
        key = (i, round(sigma, 4))
        if key not in self.cache:
            name, noisy, clean = DATA[i]
            t0 = time.perf_counter()
            out = haar_tsvd(noisy, sigma, haar()).image if sigma > 0 else noisy.astype(np.float32)
            dt = time.perf_counter() - t0
            self.cache[key] = (psnr(out, clean), ssim(out, clean), dt)
        return self.cache[key]


def sigma_of(rule, est, kappa, floor):
    return max(kappa * est[rule.split("+")[0]], floor if rule.endswith("+floor") else 0.0)


def _cbm3d_job(job):
    i, sigma = job
    name, noisy, clean = DATA[i]
    t0 = time.perf_counter()
    out = cbm3d(noisy, sigma) if sigma > 0 else noisy.astype(np.float32)
    return i, sigma, psnr(out, clean), ssim(out, clean), time.perf_counter() - t0


def part_polyu(use_cbm3d, workers):
    val, test = range(N_VAL), range(N_VAL, len(DATA))
    est = {i: {k: f(DATA[i][1]) for k, f in ESTIMATORS.items()} for i in range(len(DATA))}
    H = Haar()

    # 1) tune each rule on the validation images
    tuning, chosen = [], {}
    for rule in ("mad", "pca"):
        for k in GRIDS[rule]:
            p = float(np.mean([H(i, sigma_of(rule, est[i], k, 0))[0] for i in val]))
            tuning.append(dict(rule=rule, kappa=k, floor=0.0, val_psnr=p))
            print(f"{rule:9s} kappa {k:5.1f}: val PSNR {p:.2f}", flush=True)
        best = max((t for t in tuning if t["rule"] == rule), key=lambda t: t["val_psnr"])
        edge = best["kappa"] in (GRIDS[rule][0], GRIDS[rule][-1])
        chosen[rule] = best
        print(f"{rule}: kappa {best['kappa']}" + ("  ** edge of the grid **" if edge else ""), flush=True)
    k_mad = chosen["mad"]["kappa"]
    for f in FLOORS:
        p = float(np.mean([H(i, sigma_of("mad+floor", est[i], k_mad, f))[0] for i in val]))
        tuning.append(dict(rule="mad+floor", kappa=k_mad, floor=f, val_psnr=p))
        print(f"mad+floor kappa {k_mad} floor {f:4.1f}: val PSNR {p:.2f}", flush=True)
    chosen["mad+floor"] = max((t for t in tuning if t["rule"] == "mad+floor"), key=lambda t: t["val_psnr"])
    write_csv(tuning, RESULTS / "polyu_est_tuning.csv")

    # 2) test images, Haar-tSVD with each rule's chosen parameters
    rows = []
    for rule, c in chosen.items():
        for i in test:
            sg = sigma_of(rule, est[i], c["kappa"], c["floor"])
            p, s, dt = H(i, sg)
            rows.append(dict(image=DATA[i][0], method="Haar-tSVD", rule=rule, kappa=c["kappa"],
                             floor=c["floor"], mad=est[i]["mad"], pca=est[i]["pca"], sigma=sg,
                             psnr=p, ssim=s, time=dt))
        print(f"test {rule}: {np.mean([r['psnr'] for r in rows if r['rule'] == rule]):.2f} dB", flush=True)

    # 3) optional: CBM3D with the rule that is best on validation, kappa re-tuned for CBM3D
    if use_cbm3d:
        rule = max(chosen, key=lambda r: chosen[r]["val_psnr"])
        c = chosen[rule]
        grid = [(round(c["kappa"] * s, 3), round(c["floor"] * s, 3)) for s in CBM3D_SCALE]
        jobs = sorted({(i, sigma_of(rule, est[i], k, fl)) for k, fl in grid for i in val})
        print(f"CBM3D: rule {rule}, (kappa, floor) grid {grid}, {len(jobs)} val runs, {workers} workers", flush=True)
        with mp.get_context("fork").Pool(workers) as pool:
            res = {(i, round(sg, 4)): r for i, sg, *r in pool.imap_unordered(_cbm3d_job, jobs)}
        ct = [dict(rule=rule, scale=s, kappa=k, floor=fl, val_psnr=float(np.mean(
            [res[(i, round(sigma_of(rule, est[i], k, fl), 4))][0] for i in val])))
              for s, (k, fl) in zip(CBM3D_SCALE, grid)]
        write_csv(ct, RESULTS / "polyu_est_cbm3d_tuning.csv")
        bt = max(ct, key=lambda t: t["val_psnr"])
        kb, fb = bt["kappa"], bt["floor"]
        print(f"CBM3D scale {bt['scale']}: kappa {kb}, floor {fb}",
              "(edge of grid)" if bt["scale"] in (CBM3D_SCALE[0], CBM3D_SCALE[-1]) else "", flush=True)
        jobs = [(i, sigma_of(rule, est[i], kb, fb)) for i in test]
        with mp.get_context("fork").Pool(workers) as pool:
            for n, (i, sg, p, s, dt) in enumerate(pool.imap_unordered(_cbm3d_job, jobs), 1):
                rows.append(dict(image=DATA[i][0], method="CBM3D", rule=rule, kappa=kb, floor=fb,
                                 mad=est[i]["mad"], pca=est[i]["pca"], sigma=sg, psnr=p, ssim=s, time=dt))
                print(f"[{n}/{len(jobs)}] CBM3D {DATA[i][0]} sigma={sg:.2f} {p:.2f}", flush=True)

    write_csv(rows, RESULTS / "polyu_est_per_image.csv")
    summ = []
    for (m, rule) in dict.fromkeys((r["method"], r["rule"]) for r in rows):
        rs = [r for r in rows if r["method"] == m and r["rule"] == rule]
        val_p = chosen[rule]["val_psnr"] if m == "Haar-tSVD" else float("nan")
        summ.append(dict(method=m, rule=rule, kappa=rs[0]["kappa"], floor=rs[0]["floor"], val_psnr=val_p,
                         psnr=float(np.mean([r["psnr"] for r in rs])), ssim=float(np.mean([r["ssim"] for r in rs])),
                         time=float(np.mean([r["time"] for r in rs if r["sigma"] > 0])), n=len(rs)))
    write_csv(summ, RESULTS / "polyu_est_summary.csv")
    print(md_table(summ, ["method", "rule", "kappa", "floor", "val_psnr", "psnr", "ssim", "time", "n"],
                   {"val_psnr": "{:.2f}", "psnr": "{:.2f}", "ssim": "{:.4f}", "time": "{:.2f}"}))


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["kodak", "polyu"], required=True)
    ap.add_argument("--cbm3d", action="store_true", help="also run CBM3D with the best rule (slow, CPU)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N PolyU images (smoke test)")
    a = ap.parse_args()
    if a.part == "kodak":
        part_kodak()
    else:
        DATA = list(polyu())[:a.limit]
        part_polyu(a.cbm3d, a.workers)


if __name__ == "__main__":
    main()
