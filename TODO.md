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
- [ ] **Mac (MPS):** `uv sync`
- [ ] **CUDA machine / Colab:**
      `pip install torch --index-url https://download.pytorch.org/whl/cu124 && pip install -r requirements.txt`
- [ ] Data (only if `data/` is missing): `bash scripts/setup_data.sh` → expect 24 and 200 files
- [ ] Backend check: `HAARTSVD_DEVICE=cuda python scripts/check_device.py` (or `mps`)
      → must print `OK`. The CUDA path has **not been tested yet**; fix any mismatch before Phase 2.

## Phase 2 — Experiments
One command runs everything (`PYTHON=python` on a non-uv machine):
```bash
scripts/run_all.sh cuda          # or: mps / cpu / auto;  FAST=1 skips Local-tSVD+PCA on full sets
```
Or step by step (`HAARTSVD_DEVICE` picks the device):

| # | Command | Outputs (in `results/`) | Check |
|---|---|---|---|
| 2.1 | `python experiments/kodak.py` | `kodak_per_image.csv`, `kodak_summary.csv`, `images/kodak*_kodim23_*` | 144 rows (24 imgs × 2 cases × 3 methods) |
| 2.2 | `python experiments/polyu.py` | `polyu_kappa_tuning.csv`, `polyu_per_image.csv`, `polyu_summary.csv` | chosen κ printed; 90 test images |
| 2.3 | `python experiments/ablations.py` | `ablation_transforms.csv`, `ablation_params.csv`, `ablation_sigma.csv`, `timing.csv`, `ablations.md` | no NaN PSNR; timing table has device row |
| 2.4 | `python experiments/figures.py` | `figures/kodak_case{1,2}_kodim23.png`, `figures/polyu_example.png`, `figures/sigma_sensitivity.png` | open each figure, check crops and labels |

Sanity checks on the numbers:
- [ ] Kodak case 1 Haar-tSVD is roughly 33–34.5 dB (paper 34.13); it should beat CBM3D, as in the paper.
- [ ] If κ is chosen at the edge of the grid (1.0 or 4.0), widen `KAPPAS` in `polyu.py` and rerun 2.2.
- [ ] Note the oracle-σ gap (oracle − estimated σ) on PolyU; it goes into the discussion.
- [ ] Optional: on CUDA, rerun 2.3 so `timing.csv` has a CUDA row next to the MPS/CPU rows.

## Phase 3 — Further literature review (for the report)
Summarise each item in 2–4 sentences: what it does, how it relates to Haar-tSVD, and what we take from it.
- [ ] **Noise-level estimation** (replaces the paper's CNN): Donoho & Johnstone 1994 (MAD, what we use);
      Liu, Tanaka & Okutomi 2013 (weak-texture patches, PCA); Chen, Zhu & Heng ICCV 2015 (PCA eigenvalue
      statistics). Explain why MAD underestimates correlated real sRGB noise, which motivates κ.
- [ ] **Real-noise modelling:** Foi et al. 2008 (Poisson–Gaussian); Mäkitalo & Foi 2013 (generalised
      Anscombe VST); camera-pipeline effects (demosaicking, JPEG) that make sRGB noise spatially and
      channel-correlated, which matters for the AWGN assumption behind Eq. (12).
- [ ] **GCP / predecessors in detail:** Kong et al., GCP-ID (TIP 2025) and MSt-SVD (TIP 2019): what changes
      in Haar-tSVD (no local bases, fixed Haar).
- [ ] **Zero-shot / self-supervised denoisers** used as baselines in Table IV of the paper: Noise2VST,
      ZS-N2N (Mansour & Heckel CVPR 2023), Pixel2Pixel, Blind2Unblind. Position them as "no paired data,
      but per-image optimisation", against our "no optimisation at all".
- [ ] **Evaluation protocols:** SIDD/DND benchmark protocols; per-channel SSIM vs. luminance SSIM
      (the authors' code averages per-channel SSIM, and so do we).
- [ ] Add a comparison table (method, needs training?, iterative?, group transform, noise estimate).

## Phase 4 — Write the progress report (`report.md`)
Match the style and length of `group11-ppr.md` (same author block, numbered sections, numbered
references), then export to PDF (e.g. `pandoc report.md -o report.pdf`).

Outline and where each part's content comes from:
1. **Abstract.** Problem, what was implemented, headline numbers from `kodak_summary.csv` / `polyu_summary.csv`.
2. **Introduction / progress since the preliminary report.** Short; point to the ppr for the problem statement.
3. **Further literature review.** From Phase 3.
4. **Implementation.**
   - Pipeline as built: GCP matching, global bases, Haar, threshold, aggregation.
   - Design choices and deviations from the paper:
     - orthonormal channel FFT, so τ needs a scale (`TAU_SCALE = 0.45`, tuned on 2 images);
     - search radius 20 and step 4, taken from the authors' code (the paper says W < 20);
     - MAD × κ in place of the CNN estimator.
   - Backends: NumPy reference, MPS real-form path (orthogonal 192×192 matrix), CUDA complex path, GPU matcher.
5. **Results replicated from the literature.** Each subsection puts the paper's numbers next to ours:
   - 5.1 Paper Table IV (Kodak case 1/2, PolyU: CBM3D, MSt-SVD, Haar-tSVD) vs ours: `kodak_summary.csv`,
     `polyu_summary.csv`. Use Local-tSVD+PCA as the stand-in for MSt-SVD and say so.
   - 5.2 Paper Table I (locally-learned / globally-learned / reused bases) vs ours:
     `ablation_transforms.csv` (group = haar, bases = local / global / reused).
   - 5.3 Paper Fig. 6 (Haar vs PCA σ-sensitivity): `ablation_sigma.csv`, `figures/sigma_sensitivity.png`.
     Does Haar degrade faster than PCA when σ is wrong?
6. **Preliminary results of our own.**
   - RQ1 (Haar vs PCA quality): `ablation_transforms.csv`.
   - RQ2 (bases sharing / reuse): same file.
   - RQ3 (speed): `timing.csv`. Give stage-wise times, MPS vs CPU (vs CUDA), and vs CBM3D.
   - Parameter sensitivity: `ablation_params.csv` (ps, K, search radius, matching rule, aggregation weights).
   - PolyU: κ tuning curve, estimated-σ vs oracle-σ gap.
   - Visual results: `figures/*.png`.
7. **Discussion and open issues.**
   - Gap to the paper's numbers and likely causes (τ normalisation, SSIM details, noise realisation).
   - Behaviour at high noise (case 2).
   - Real-noise σ estimation.
8. **Plan for the final phase.** From Phase 5 below.
9. **References.** Keep the ppr numbering and add the new ones.

Report checklist:
- [ ] Every number in the report is traceable to a CSV in `results/` (no hand-typed results).
- [ ] Paper numbers are labelled "reported in [22]", ours "this work".
- [ ] State the hardware (CPU model, GPU / MPS) and the seeds (Kodak noise seed = 1000·case + index).
- [ ] State what was **not** done: DND, SIDD, A-/RA-Haar-tSVD.

## Phase 5 — After the progress report (final phase)
- [ ] SIDD validation (`ValidationNoisyBlocksSrgb.mat`) and a DND submission
- [ ] PCA-based noise estimator (Chen et al. 2015) to close the oracle-σ gap
- [ ] Eigenvalue-rank adaptive σ (paper Eq. 23) without the CNN
- [ ] Optional: FCN refinement of the mean patch (RA-Haar-tSVD)
