# Haar-tSVD — to-do (experiments → progress report)

Work through the phases in order. Each step lists the command, what it produces and the check
that must pass before moving on. `[x]` = done already, `[ ]` = to do.

---

## Phase 0 — Already in place
- [x] Implementation: `haartsvd/` (GCP matching, global/local/reused t-SVD bases, Haar / PCA /
      identity group transform, hard threshold, aggregation, MAD noise estimator)
- [x] Backends: NumPy (reference), torch real-form path for shared bases (MPS/CUDA/CPU),
      torch complex path for local bases + PCA (CUDA/CPU), torch block matcher (CUDA)
- [x] Datasets downloaded: `data/kodak` (24), `data/polyu` (100 pairs)
- [x] Threshold scale `TAU_SCALE = 0.45` tuned on kodim01 / kodim13 (`experiments/common.py`)
- [x] Verified on MPS: same PSNR as NumPy, filter stage 2.97 s → 0.42 s (kodim01)

## Phase 1 — Environment
- [x] **CUDA machine / Colab (Tesla T4):** torch 2.11.0+cu130, `pip install -r requirements.txt`
- [x] Backend check: `HAARTSVD_DEVICE=cuda python scripts/check_device.py` → `OK`
      (CUDA vs NumPy ≤ 0.0002 dB on all four variants, both matchers)
- [ ] Optional: **Mac (MPS)** `uv sync` and rerun timing (report 4.3 MPS number is from dev notes only)

## Phase 2 — Experiments
One command runs everything (`PYTHON=python` on a non-uv machine):
```bash
scripts/run_all.sh cuda          # or: mps / cpu / auto;  FAST=1 skips Local-tSVD+PCA on full sets
```

| # | Command | Outputs (in `results/`) | Status |
|---|---|---|---|
| 2.1 | `python experiments/kodak.py` | `kodak_per_image.csv`, `kodak_summary.csv`, `images/kodak*_kodim23_*` | [x] 144 rows |
| 2.2 | `python experiments/polyu.py` | `polyu_kappa_tuning.csv`, `polyu_per_image.csv`, `polyu_summary.csv` | [x] κ = 24 (grid widened twice) |
| 2.3 | `python experiments/ablations.py` | `ablation_transforms.csv`, `ablation_params.csv`, `ablation_sigma.csv`, `timing.csv`, `ablations.md` | [x] CUDA timing row |
| 2.4 | `python experiments/figures.py` | `figures/kodak_case{1,2}_kodim23.png`, `figures/polyu_example.png`, `figures/sigma_sensitivity.png` | [x] |
| 2.5 | `python experiments/polyu_extra.py --part cbm3d` | `polyu_cbm3d_{val,kappa_tuning,per_image,summary}.csv` | [x] κ_CBM3D = 16, 38.52 dB |
| 2.6 | `python experiments/polyu_extra.py --part oracle` | `polyu_oracle_ext.csv` | [x] 39.18 dB, gap 0.42 dB |
| 2.7 | `python scripts/fill_report_tables.py` | `report_tables.md` | [ ] run once CSVs are local |

Sanity checks on the numbers:
- [x] Kodak case 1 Haar-tSVD 33.99 dB (paper 34.13), beats CBM3D (33.69). Case 2 does **not** (28.76 vs 29.07).
- [x] κ grid widened: {1..4} → {2..12} → {8..24}; κ = 24 at the edge but flat (+0.01 dB from 20).
      `KAPPAS` in `polyu.py` updated to the final grid.
- [x] Oracle-σ gap noted: 0.25 dB (grid ≤ 15), 0.42 dB with extended grid (lower bound; 16 images pick σ = 32).
- [x] CUDA row in `timing.csv`.
- [ ] Optional: copy the Colab `results/*.csv`, `results/images/`, `figures/*.png` into this repo
      (only `kodak.log` / `polyu.log` are here; needed for step 2.7 and Phase 5 comparisons)

## Phase 3 — Further literature review
- [x] Noise-level estimation (MAD, Liu et al., Pyatykh et al., Chen et al.) — report 3.1
- [x] Real-noise modelling (Foi, Mäkitalo & Foi, Nam et al.) — report 3.2
- [x] GCP-ID / MSt-SVD in detail — report 3.3
- [x] Zero-shot / self-supervised baselines — report 3.4
- [x] Noise2VST [44] (ICCV 2025) and Pixel2Pixel [45] (TPAMI 47(6)) verified; ZS-N2N noted as not in Table IV
- [x] Evaluation protocols — report 3.5
- [x] Comparison table — report 3.6

## Phase 4 — Progress report (`report.md`)
- [x] Draft written, sections 1, 3–8 + references
- [x] Section 2 added (problem + RQ1–RQ3 recap)
- [x] Draft/verify notes removed; cited ppr references [1]–[33] repeated in the reference list
- [x] Arithmetic of every derived number (gaps, ratios, drops) re-checked; two wording fixes
- [ ] Optional: cross-check tables against `results/report_tables.md` (needs Colab CSVs, step 2.7)
- [x] PDF exported: `pandoc report.md -o report.pdf --pdf-engine=typst -V papersize=a4 -V margin.x=2cm -V margin.y=2cm -V fontsize=10pt`
- [ ] Optional: embed figures once `figures/*.png` are copied from Colab (currently referenced by filename only)

Report checklist:
- [x] Every number comes from the Colab CSVs (not copied to this repo; copy them to make it checkable locally)
- [x] Paper numbers labelled "reported in [22]", ours "this work"
- [x] Hardware and seeds stated (report 4.4)
- [x] What was **not** done stated: DND, SIDD, A-/RA-Haar-tSVD (report 7)

## Phase 5 — Final phase

### Code (done here, smoke-tested on the M4)
- [x] PCA noise estimator of Chen et al. [39]: `estimate_sigma_pca` in `haartsvd/utils.py`
      (Kodak check: within ~1 of the true per-channel sigma in both cases, same as MAD; 0.07 s/image)
- [x] `experiments/noise_est.py`: `--part kodak` (estimator accuracy), `--part polyu [--cbm3d]`
      (rules mad / mad+floor / pca, all tuned on the 10 validation images, results on the 90 test images)
- [x] SIDD loader `sidd()` + `experiments/sidd.py` (resumable; sigma rule taken from the PolyU tuning,
      nothing tuned on SIDD; `--oracle` optional)
- [x] `SIDD=1 bash scripts/setup_data.sh` (440 MB zip from the Restormer mirror; official server is down)
- [x] `scripts/run_final.sh` runs everything; `fill_report_tables.py` has sections F.1–F.3

### Run on the VPS (CUDA + many CPU cores)
```bash
git clone -b final-phase https://github.com/Lokesh-Kudipudi/DLCV_M2026.git && cd DLCV_M2026
pip install torch --index-url https://download.pytorch.org/whl/cu124 && pip install -r requirements.txt
bash scripts/setup_data.sh                      # Kodak + PolyU (data/ is git-ignored)
PYTHON=python scripts/run_final.sh              # + FULL=1 to also regenerate the progress-report CSVs
git add results && git commit -m "Final-phase results" && git push
```
Expected time on a T4-class GPU + 16–32 cores: noise_est polyu ~15 min (+ CBM3D ~15 min),
SIDD Haar-tSVD ~10 min, SIDD CBM3D ~10–20 min; FULL=1 adds ~2–3 h (CBM3D on Kodak/PolyU is serial).
- [ ] Run on the VPS and push `results/`
- [ ] Pull here; write report sections from `results/report_tables.md` (F.1–F.3)

### Still not implemented
- [ ] DND submission (needs registration on the DND site; server-side evaluation)
- [ ] Case-specific `TAU_SCALE` for high noise (Kodak case 2 peaks at f = 1.2), report as tuned, not blind
- [ ] sigma sweep / own `TAU_SCALE` for Local-tSVD+PCA (MSt-SVD stand-in is 0.4–1.7 dB below paper)
- [ ] Eigenvalue-rank adaptive sigma (paper Eq. 23) without the CNN
- [ ] Optional: FCN refinement of the mean patch (RA-Haar-tSVD)
