"""SIDD sRGB validation (40 images x 32 blocks of 256x256 = 1280 blocks), report Sec. 8.

No parameter is tuned on SIDD: the sigma rule (estimator, kappa, floor) and CBM3D's kappa are read
from results/polyu_est_summary.csv (tuned on the 10 PolyU validation images by noise_est.py), or
given with --rule / --kappa / --floor / --cbm3d-kappa. Data: `SIDD=1 bash scripts/setup_data.sh`.

Haar-tSVD (and Local-tSVD+PCA) run in the main process on the GPU; CBM3D runs in parallel CPU
workers. Resumable: rows are appended to results/sidd_per_image.csv as they finish, so after a
disconnect rerun the same command and finished (block, method) pairs are skipped.
--oracle adds Haar-tSVD with the best sigma per block from ORACLE (uses the ground truth; upper bound).

Outputs (results/): sidd_per_image.csv, sidd_summary.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # CBM3D workers: one thread each

import argparse  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from common import METHODS, RESULTS, cbm3d, md_table, write_csv  # noqa: E402
from haartsvd.utils import ESTIMATORS, psnr, sidd, ssim  # noqa: E402

ORACLE = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40]
FIELDS = ["image", "method", "rule", "kappa", "floor", "mad", "pca", "sigma", "psnr", "ssim", "time"]
DATA = None  # filled in main(); inherited by forked CBM3D workers


def _read(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sigma_of(est, rule, kappa, floor):
    return max(kappa * est[rule.split("+")[0]], floor if rule.endswith("+floor") else 0.0)


def _cbm3d_job(job):
    i, sigma = job
    name, noisy, clean = DATA[i]
    t0 = time.perf_counter()
    out = cbm3d(noisy, sigma) if sigma > 0 else noisy.astype(np.float32)
    return i, sigma, psnr(out, clean), ssim(out, clean), time.perf_counter() - t0


def tuned_params(a):
    """Sigma rule from the PolyU validation tuning, unless given on the command line."""
    summ = _read(RESULTS / "polyu_est_summary.csv")
    haar = [r for r in summ if r["method"] == "Haar-tSVD"]
    cb = next((r for r in summ if r["method"] == "CBM3D"), None)
    if a.rule is None:
        if not haar:
            raise SystemExit("no polyu_est_summary.csv: run noise_est.py --part polyu first, or pass --rule/--kappa")
        best = max(haar, key=lambda r: float(r["val_psnr"]))
        a.rule, a.kappa, a.floor = best["rule"], float(best["kappa"]), float(best["floor"])
    elif a.kappa is None:
        r = next((r for r in haar if r["rule"] == a.rule), None)
        if r is None:
            raise SystemExit(f"no tuned kappa for rule {a.rule}: pass --kappa")
        a.kappa, a.floor = float(r["kappa"]), float(r["floor"])
    if a.cbm3d_kappa is None and cb is not None and cb["rule"] == a.rule:
        a.cbm3d_kappa, a.cbm3d_floor = float(cb["kappa"]), float(cb["floor"])
    if a.cbm3d_floor is None:
        a.cbm3d_floor = a.floor
    print(f"rule {a.rule}, kappa {a.kappa}, floor {a.floor}, CBM3D kappa {a.cbm3d_kappa}, "
          f"floor {a.cbm3d_floor}", flush=True)


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["Haar-tSVD", "CBM3D"], choices=list(METHODS))
    ap.add_argument("--rule", choices=["mad", "mad+floor", "pca"], default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--cbm3d-kappa", type=float, default=None)
    ap.add_argument("--cbm3d-floor", type=float, default=None)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N blocks (smoke test)")
    a = ap.parse_args()
    tuned_params(a)
    if "CBM3D" in a.methods and a.cbm3d_kappa is None:
        raise SystemExit("CBM3D kappa unknown: run noise_est.py --part polyu --cbm3d, or pass --cbm3d-kappa")

    DATA = list(sidd(a.limit))
    est = [{k: f(noisy) for k, f in ESTIMATORS.items()} for _, noisy, _ in DATA]
    path = RESULTS / "sidd_per_image.csv"
    done = {(r["image"], r["method"]) for r in _read(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()

        def emit(i, method, kappa, sigma, p, s, dt, floor=a.floor):
            w.writerow(dict(image=DATA[i][0], method=method, rule=a.rule, kappa=kappa, floor=floor,
                            mad=est[i]["mad"], pca=est[i]["pca"], sigma=sigma, psnr=p, ssim=s, time=dt))
            f.flush()

        # GPU methods in this process
        for m in [m for m in a.methods if m != "CBM3D"]:
            todo = [i for i in range(len(DATA)) if (DATA[i][0], m) not in done]
            print(f"{m}: {len(todo)} blocks left", flush=True)
            for n, i in enumerate(todo, 1):
                name, noisy, clean = DATA[i]
                sg = sigma_of(est[i], a.rule, a.kappa, a.floor)
                t0 = time.perf_counter()
                out = METHODS[m](noisy, sg) if sg > 0 else noisy
                dt = time.perf_counter() - t0
                emit(i, m, a.kappa, sg, psnr(out, clean), ssim(out, clean), dt)
                if n % 100 == 0:
                    print(f"  [{n}/{len(todo)}]", flush=True)

        if a.oracle:
            m = "Haar-tSVD (oracle sigma)"
            todo = [i for i in range(len(DATA)) if (DATA[i][0], m) not in done]
            print(f"{m}: {len(todo)} blocks left", flush=True)
            for i in todo:
                name, noisy, clean = DATA[i]
                best = max(((psnr(o := METHODS["Haar-tSVD"](noisy, sg), clean), sg, o) for sg in ORACLE),
                           key=lambda t: t[0])
                emit(i, m, "", best[1], best[0], ssim(best[2], clean), 0.0)

        # CBM3D in parallel CPU workers
        if "CBM3D" in a.methods:
            jobs = [(i, sigma_of(est[i], a.rule, a.cbm3d_kappa, a.cbm3d_floor))
                    for i in range(len(DATA)) if (DATA[i][0], "CBM3D") not in done]
            print(f"CBM3D: {len(jobs)} blocks left, {a.workers} workers", flush=True)
            with mp.get_context("fork").Pool(a.workers) as pool:
                for n, (i, sg, p, s, dt) in enumerate(pool.imap_unordered(_cbm3d_job, jobs), 1):
                    emit(i, "CBM3D", a.cbm3d_kappa, sg, p, s, dt, a.cbm3d_floor)
                    if n % 50 == 0:
                        print(f"  [{n}/{len(jobs)}] CBM3D", flush=True)

    rows = _read(path)
    summ = []
    for m in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == m]
        summ.append(dict(method=m, rule=rs[0]["rule"], kappa=rs[0]["kappa"],
                         psnr=float(np.mean([float(r["psnr"]) for r in rs])),
                         ssim=float(np.mean([float(r["ssim"]) for r in rs])),
                         time=float(np.mean([float(r["time"]) for r in rs])), n=len(rs)))
    summ.insert(0, dict(method="Noisy input", rule="", kappa="",
                        psnr=float(np.mean([psnr(y, c) for _, y, c in DATA])), ssim=float("nan"), time=0.0,
                        n=len(DATA)))
    write_csv(summ, RESULTS / "sidd_summary.csv")
    print(md_table(summ, ["method", "rule", "kappa", "psnr", "ssim", "time", "n"],
                   {"psnr": "{:.2f}", "ssim": "{:.4f}", "time": "{:.2f}"}))


if __name__ == "__main__":
    main()
