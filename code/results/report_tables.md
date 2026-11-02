### 5.1 Paper Table IV vs this work (PSNR / SSIM)

| Method | Kodak-1 [22] | Kodak-1 (this work) | Kodak-2 [22] | Kodak-2 (this work) | PolyU [22] | PolyU (this work) |
|---|---|---|---|---|---|---|
| CBM3D | 33.51 / 0.893 | 33.69 / 0.895 | 28.90 / 0.780 | 29.07 / 0.782 | 38.69 / 0.970 | 38.30 / 0.962 |
| MSt-SVD / Local-tSVD+PCA | 33.99 / 0.906 | 33.57 / 0.888 | 29.12 / 0.770 | 27.44 / 0.653 | 38.85 / 0.971 | 37.98 / 0.960 |
| Haar-tSVD | 34.13 / 0.910 | 33.99 / 0.906 | 29.16 / 0.780 | 28.76 / 0.741 | 38.78 / 0.969 | 38.47 / 0.964 |

### 5.2 Locally learned / globally learned / reused bases (Haar group transform)

| Dataset | Local | Global | Reused | Time local / global / reused (s) |
|---|---|---|---|---|
| Kodak subset, case 1 | 34.38 / 0.914 | 34.37 / 0.916 | 34.39 / 0.917 | 0.31 / 0.70 / 0.19 |
| Kodak subset, case 2 | 28.72 / 0.736 | 29.05 / 0.762 | 29.06 / 0.762 | 0.31 / 0.63 / 0.19 |
| CC15 [22] | 37.86 | 37.89 | 37.90 |  |
| HighISO [22] | 40.05 | 40.13 | 40.15 |  |
| SIDD-val [22] | 34.35 | 34.39 | 34.38 |  |
| Time (s) [22] | 6.95 | 6.13 | 6.02 |  |

### 6.1 Group transform x bases (all rows)

| case | group | bases | PSNR | SSIM | time (s) | n |
|---|---|---|---|---|---|---|
| 1 | haar | global | 34.37 | 0.9163 | 0.70 | 6 |
| 2 | haar | global | 29.05 | 0.7622 | 0.63 | 6 |
| 1 | haar | local | 34.38 | 0.9138 | 0.31 | 6 |
| 2 | haar | local | 28.72 | 0.7364 | 0.31 | 6 |
| 1 | haar | reused | 34.39 | 0.9167 | 0.19 | 6 |
| 2 | haar | reused | 29.06 | 0.7625 | 0.19 | 6 |
| 1 | pca | global | 34.07 | 0.9021 | 0.77 | 6 |
| 2 | pca | global | 28.15 | 0.7005 | 0.77 | 6 |
| 1 | pca | local | 33.94 | 0.8965 | 0.37 | 6 |
| 2 | pca | local | 27.64 | 0.6700 | 0.37 | 6 |
| 1 | pca | reused | 34.11 | 0.9031 | 0.33 | 6 |
| 2 | pca | reused | 28.18 | 0.7014 | 0.34 | 6 |
| 1 | none | global | 33.72 | 0.9056 | 0.63 | 6 |
| 2 | none | global | 27.84 | 0.7255 | 0.63 | 6 |

### 5.3 Sensitivity to sigma: PSNR (drop from own peak) by factor

| case | group | f=0.6 | f=0.8 | f=1 | f=1.2 | f=1.5 | f=2 |
|---|---|---|---|---|---|---|---|
| 1 | haar | 29.77 (-4.60) | 33.09 (-1.28) | 34.37 (+0.00) | 34.23 (-0.14) | 33.38 (-0.99) | 32.10 (-2.27) |
| 1 | pca | 29.35 (-5.18) | 32.35 (-2.19) | 34.07 (-0.46) | 34.54 (+0.00) | 34.25 (-0.28) | 33.40 (-1.14) |
| 2 | haar | 21.80 (-7.71) | 26.31 (-3.20) | 29.05 (-0.45) | 29.50 (+0.00) | 28.93 (-0.57) | 28.05 (-1.45) |
| 2 | pca | 21.34 (-8.33) | 25.15 (-4.52) | 28.15 (-1.52) | 29.48 (-0.19) | 29.67 (+0.00) | 29.14 (-0.52) |

### 6.2 PolyU kappa tuning (validation = first 10 images)

| kappa | val PSNR |
|---|---|
| 8.0 | 35.86 |
| 10.0 | 36.02 |
| 12.0 | 36.15 |
| 14.0 | 36.24 |
| 16.0 | 36.31 |
| 20.0 | 36.37 |
| 24.0 | 36.39 |

Chosen kappa = 24.0  **(edge of the grid: widen KAPPAS and rerun)**

### 6.2 PolyU kappa tuning for CBM3D

| kappa | val PSNR (CBM3D) |
|---|---|
| 8.0 | 35.90 |
| 12.0 | 36.18 |
| 16.0 | 36.29 |
| 24.0 | 36.29 |
| 32.0 | 36.21 |

Chosen kappa for CBM3D = 24.0

### 6.2 PolyU test set (90 images)

| method | PSNR | SSIM | time (s) | n |
|---|---|---|---|---|
| Noisy input | 35.86 | nan | - | 90 |
| Haar-tSVD | 38.47 | 0.9636 | 0.46 | 90 |
| Local-tSVD+PCA | 37.98 | 0.9600 | 0.27 | 90 |
| Haar-tSVD (oracle sigma) | 38.82 | 0.9689 | - | 90 |
| CBM3D | 38.30 | 0.9623 | 104.34 | 90 |

Oracle-sigma gap (oracle - estimated) = 0.35 dB

Extended oracle grid (adds 18-32): 38.99 dB, gap 0.52 dB; best sigma above 15 on 78/90 images

### 6.3 Runtime on a 512x512x3 image (s; empty = not recorded or fused)

| method | search | bases | filter | aggregate | total |
|---|---|---|---|---|---|
| Haar-tSVD (global, cuda) | 0.19 | 0.29 | 0.03 |  | 0.51 |
| Haar-tSVD (global, torch CPU) | 1.06 | 0.29 | 2.25 |  | 3.60 |
| Haar-tSVD (global, NumPy CPU) | 0.88 | 0.30 | 9.74 | 0.80 | 11.72 |
| Global t-SVD + PCA | 0.10 | 0.29 | 0.12 |  | 0.51 |
| Local t-SVD + Haar | 0.10 | 0.00 | 0.10 |  | 0.21 |
| Local t-SVD + PCA | 0.10 | 0.00 | 0.14 |  | 0.25 |
| CBM3D (bm3d package) |  |  |  |  | 14.65 |

### 6.4 Parameter sensitivity (case 1)

| param | value | PSNR | SSIM | time (s) |
|---|---|---|---|---|
| ps | 6 | 34.29 | 0.9122 | 0.39 |
| ps | 8 | 34.37 | 0.9163 | 0.64 |
| ps | 10 | 34.35 | 0.9172 | 1.03 |
| K | 8 | 33.82 | 0.8992 | 0.61 |
| K | 16 | 34.21 | 0.9106 | 0.62 |
| K | 32 | 34.37 | 0.9163 | 0.64 |
| K | 64 | 34.39 | 0.9187 | 0.67 |
| search radius | 5 | 34.08 | 0.9099 | 0.49 |
| search radius | 10 | 34.29 | 0.9145 | 0.53 |
| search radius | 15 | 34.35 | 0.9158 | 0.57 |
| search radius | 20 | 34.37 | 0.9163 | 0.63 |
| matching | gcp | 34.37 | 0.9163 | 0.63 |
| matching | avg | 34.34 | 0.9161 | 0.56 |
| matching | green | 34.36 | 0.9162 | 0.56 |
| aggregation | uniform | 34.37 | 0.9163 | 0.63 |
| aggregation | sparse | 34.39 | 0.9171 | 0.63 |

### F.1 Noise estimators on Kodak (per-channel sigma, mean over 24 images)

| case | estimator | true R / G / B | estimated R / G / B | mean rel. error |
|---|---|---|---|---|
| 1 | mad | 15 / 10 / 20 | 16.02 / 11.22 / 20.82 | 7.7 % |
| 1 | pca | 15 / 10 / 20 | 15.98 / 11.32 / 20.74 | 7.8 % |
| 2 | mad | 40 / 30 / 50 | 40.50 / 30.65 / 50.46 | 1.5 % |
| 2 | pca | 40 / 30 / 50 | 40.39 / 30.54 / 50.41 | 1.2 % |

### F.2 PolyU test set with each sigma rule (tuned on the 10 validation images)

| method | rule | kappa | floor | val PSNR | test PSNR | SSIM | time (s) | n |
|---|---|---|---|---|---|---|---|---|
| Haar-tSVD | mad | 24.0 | 0.0 | 36.39 | 38.47 | 0.9636 | 0.42 | 90 |
| Haar-tSVD | pca | 20.0 | 0.0 | 37.03 | 38.72 | 0.9668 | 0.42 | 90 |
| Haar-tSVD | mad+floor | 24.0 | 16.0 | 37.08 | 38.84 | 0.9682 | 0.42 | 90 |
| CBM3D | mad+floor | 19.2 | 12.8 | nan | 38.64 | 0.9680 | 92.55 | 90 |

Extended oracle (Haar-tSVD, same 90 images): 38.99 dB

### F.2b PolyU validation tuning of the sigma rules

| rule | kappa | floor | val PSNR |
|---|---|---|---|
| mad | 12.0 | 0.0 | 36.15 |
| mad | 16.0 | 0.0 | 36.31 |
| mad | 20.0 | 0.0 | 36.37 |
| mad | 24.0 | 0.0 | 36.39 |
| mad | 28.0 | 0.0 | 36.39 |
| mad | 32.0 | 0.0 | 36.38 |
| mad | 40.0 | 0.0 | 36.34 |
| pca | 4.0 | 0.0 | 35.99 |
| pca | 6.0 | 0.0 | 36.41 |
| pca | 8.0 | 0.0 | 36.70 |
| pca | 10.0 | 0.0 | 36.87 |
| pca | 12.0 | 0.0 | 36.97 |
| pca | 16.0 | 0.0 | 37.03 |
| pca | 20.0 | 0.0 | 37.03 |
| pca | 24.0 | 0.0 | 37.02 |
| pca | 32.0 | 0.0 | 36.96 |
| mad+floor | 24.0 | 0.0 | 36.39 |
| mad+floor | 24.0 | 4.0 | 36.82 |
| mad+floor | 24.0 | 8.0 | 37.02 |
| mad+floor | 24.0 | 12.0 | 37.07 |
| mad+floor | 24.0 | 16.0 | 37.08 |
| mad+floor | 24.0 | 20.0 | 37.07 |
| mad+floor | 24.0 | 24.0 | 37.05 |

### F.3 SIDD sRGB validation (1280 blocks; nothing tuned on SIDD)

| method | rule | kappa | PSNR | SSIM | time (s) | n |
|---|---|---|---|---|---|---|
| Noisy input |  |  | 23.66 | - | - | 1280 |
| Haar-tSVD | mad+floor | 24.0 | 33.56 | 0.8822 | 0.12 | 1280 |
| Haar-tSVD (oracle sigma) | mad+floor |  | 33.38 | 0.7544 | - | 1280 |
| CBM3D | mad+floor | 19.2 | 33.31 | 0.8673 | 34.40 | 1280 |

Reported in [22] (SIDD): CBM3D 34.74 / 0.922, MSt-SVD 34.38 / 0.901, Haar-tSVD 35.05 / 0.914. CBM3D time is per block with parallel workers, not comparable.
