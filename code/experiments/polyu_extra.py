"""Extra PolyU runs that do not redo the Haar-tSVD / Local-tSVD+PCA test set of polyu.py.

--part cbm3d   Tune kappa for CBM3D on the 10 validation images (its own kappa, not the one
               tuned for Haar-tSVD), then run CBM3D on the 90 test images with
               sigma = kappa_cbm3d * MAD. CPU only; images run in parallel worker processes.
               Resumable: rows are appended to the CSVs as they finish, so after a Colab
               disconnect just rerun the same command and finished images are skipped.
--part oracle  Extend the Haar-tSVD oracle-sigma grid above 15 (the estimated sigma is 17.8
               for most test images, above the old grid's maximum), and combine it with the
               oracle rows already in polyu_per_image.csv. Uses the GPU if available.

Outputs (results/): polyu_cbm3d_val.csv, polyu_cbm3d_kappa_tuning.csv,
polyu_cbm3d_per_image.csv, polyu_cbm3d_summary.csv, polyu_oracle_ext.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker; parallelism is across images

import argparse  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from common import RESULTS, cbm3d, haar, md_table, write_csv  # noqa: E402
from haartsvd.utils import estimate_sigma, polyu, psnr, ssim  # noqa: E402

N_VAL = 10  # same split as polyu.py
KAPPAS_CBM3D = [8.0, 12.0, 16.0, 24.0, 32.0]
ORACLE_EXT = [18, 20, 24, 28, 32]
DATA = None  # filled in main(); inherited by forked workers


def _cbm3d_job(job):
    i, kappa, sigma = job
    name, noisy, clean = DATA[i]
    t0 = time.perf_counter()
    # sigma = 0 happens when the MAD of the HH band is exactly 0; bm3d cannot run with
    # sigma 0, so the image is returned unchanged (Haar-tSVD at sigma 0 also leaves it as is).
    out = cbm3d(noisy, sigma) if sigma > 0 else noisy.astype(np.float32)
    dt = time.perf_counter() - t0
    return dict(image=name, kappa=kappa, sigma=sigma, psnr=psnr(out, clean),
                ssim=ssim(out, clean), time=dt)


def _read(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _run_pool(jobs, path, workers):
    """Run CBM3D jobs in parallel, appending each finished row to `path`."""
    fields = ["image", "kappa", "sigma", "psnr", "ssim", "time"]
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f, mp.get_context("fork").Pool(workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for n, r in enumerate(pool.imap_unordered(_cbm3d_job, jobs), 1):
            w.writerow(r)
            f.flush()
            print(f"[{n}/{len(jobs)}] {r['image']} kappa={r['kappa']} sigma={r['sigma']:.2f} "
                  f"{r['psnr']:.2f} {r['ssim']:.4f} {r['time']:.1f}s", flush=True)


def part_cbm3d(workers):
    val_idx = range(N_VAL)
    test_idx = range(N_VAL, len(DATA))
    mad = {i: estimate_sigma(DATA[i][1]) for i in range(len(DATA))}

    # 1) kappa tuning on the validation images
    vpath = RESULTS / "polyu_cbm3d_val.csv"
    done = {(r["image"], float(r["kappa"])) for r in _read(vpath)}
    jobs = [(i, k, k * mad[i]) for k in KAPPAS_CBM3D for i in val_idx if (DATA[i][0], k) not in done]
    print(f"CBM3D kappa tuning: {len(jobs)} runs left, {workers} workers", flush=True)
    if jobs:
        _run_pool(jobs, vpath, workers)
    vrows = _read(vpath)
    tune = [dict(kappa=k, psnr=float(np.mean([float(r["psnr"]) for r in vrows if float(r["kappa"]) == k])))
            for k in KAPPAS_CBM3D]
    write_csv(tune, RESULTS / "polyu_cbm3d_kappa_tuning.csv")
    for t in tune:
        print(f"kappa {t['kappa']}: val PSNR {t['psnr']:.2f}")
    best = max(tune, key=lambda t: t["psnr"])
    kappa = best["kappa"]
    edge = best is tune[0] or best is tune[-1]
    print(f"chosen kappa for CBM3D {kappa}" + ("  ** edge of the grid **" if edge else ""), flush=True)

    # 2) test images
    tpath = RESULTS / "polyu_cbm3d_per_image.csv"
    old = [r for r in _read(tpath) if float(r["kappa"]) == kappa]
    if len(old) != len(_read(tpath)):  # rows from a different kappa: start the test file again
        tpath.unlink()
        old = []
    done = {r["image"] for r in old}
    jobs = [(i, kappa, kappa * mad[i]) for i in test_idx if DATA[i][0] not in done]
    print(f"CBM3D test: {len(jobs)} images left", flush=True)
    if jobs:
        _run_pool(jobs, tpath, workers)
    rows = _read(tpath)
    summ = [dict(method="CBM3D", kappa=kappa, psnr=float(np.mean([float(r["psnr"]) for r in rows])),
                 ssim=float(np.mean([float(r["ssim"]) for r in rows])),
                 time=float(np.mean([float(r["time"]) for r in rows if float(r["sigma"]) > 0])),
                 n=len(rows))]
    write_csv(summ, RESULTS / "polyu_cbm3d_summary.csv")
    print(md_table(summ, ["method", "kappa", "psnr", "ssim", "time", "n"],
                   {"psnr": "{:.2f}", "ssim": "{:.4f}", "time": "{:.2f}"}))


def part_oracle():
    from haartsvd import haar_tsvd
    old = {r["image"]: (float(r["psnr"]), float(r["ssim"]), "<=15")
           for r in _read(RESULTS / "polyu_per_image.csv") if "oracle" in r["method"]}
    if not old:
        raise SystemExit("no oracle rows in polyu_per_image.csv: run polyu.py without --no-oracle first")
    rows = []
    for name, noisy, clean in DATA[N_VAL:]:
        best = old[name]
        for sg in ORACLE_EXT:
            out = haar_tsvd(noisy, sg, haar()).image
            p = psnr(out, clean)
            if p > best[0]:
                best = (p, ssim(out, clean), str(sg))
        rows.append(dict(image=name, psnr_old=old[name][0], psnr=best[0], ssim=best[1], best_sigma=best[2]))
        print(f"{name} oracle old {old[name][0]:.2f} new {best[0]:.2f} (sigma {best[2]})", flush=True)
    write_csv(rows, RESULTS / "polyu_oracle_ext.csv")
    above = sum(r["best_sigma"] != "<=15" for r in rows)
    print(f"oracle (grid <= 15): {np.mean([r['psnr_old'] for r in rows]):.2f} dB; "
          f"extended grid: {np.mean([r['psnr'] for r in rows]):.2f} dB / "
          f"{np.mean([r['ssim'] for r in rows]):.4f}; best sigma > 15 on {above}/{len(rows)} images")


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["cbm3d", "oracle"], required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    DATA = list(polyu())
    if a.part == "cbm3d":
        part_cbm3d(a.workers)
    else:
        part_oracle()


if __name__ == "__main__":
    main()
