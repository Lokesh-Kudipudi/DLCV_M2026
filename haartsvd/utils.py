"""Noise estimation, metrics and dataset loading."""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

DATA = Path(__file__).resolve().parent.parent / "data"


def estimate_sigma(img):
    """Donoho-Johnstone MAD estimate from the finest diagonal Haar band, averaged over channels."""
    x = img.astype(np.float64)
    H, W = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
    x = x[:H, :W]
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2.0
    sig = np.median(np.abs(hh - np.median(hh, axis=(0, 1))), axis=(0, 1)) / 0.6745
    return float(np.mean(sig))


def _pca_sigma_channel(x, ps):
    """Chen, Zhu & Heng (ICCV 2015): sigma^2 is the mean of the redundant eigenvalues of the
    patch covariance, where the redundant set is the largest tail {lambda_1..lambda_m} (ascending)
    whose mean is also its median (as many eigenvalues above the mean as below it)."""
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(x, (ps, ps))  # (H', W', ps, ps), all overlapping patches
    d = ps * ps
    S, m, n = np.zeros((d, d)), np.zeros(d), 0
    for r in range(0, win.shape[0], 64):  # accumulate in row blocks to bound memory
        blk = win[r:r + 64].reshape(-1, d)
        S += blk.T @ blk
        m += blk.sum(0)
        n += len(blk)
    m /= n
    lam = np.sort(np.linalg.eigvalsh(S / n - np.outer(m, m)))  # ascending
    for cnt in range(d, 0, -1):
        tail = lam[:cnt]
        tau = tail.mean()
        if np.sum(tail > tau) == np.sum(tail < tau):
            break
    return float(np.sqrt(max(tau, 0.0)))


def estimate_sigma_pca(img, ps=8):
    """PCA noise estimate of Chen et al. [39], per channel, averaged over channels."""
    x = img.astype(np.float64)
    return float(np.mean([_pca_sigma_channel(x[..., i], ps) for i in range(x.shape[2])]))


def estimate_sigma_channels(img, method="mad", ps=8):
    """Per-channel sigma estimates (used to check the estimators against a known noise level)."""
    x = img.astype(np.float64)
    if method == "pca":
        return [_pca_sigma_channel(x[..., i], ps) for i in range(x.shape[2])]
    return [estimate_sigma(x[..., i:i + 1]) for i in range(x.shape[2])]


ESTIMATORS = {"mad": estimate_sigma, "pca": estimate_sigma_pca}


def psnr(x, ref):
    mse = np.mean((np.clip(x, 0, 255).astype(np.float64) - ref.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)


def ssim(x, ref):
    """Per-channel SSIM (Gaussian window, sigma 1.5) averaged over RGB, as in the authors' code."""
    x = np.clip(x, 0, 255).astype(np.float64)
    ref = ref.astype(np.float64)
    return float(np.mean([
        structural_similarity(x[..., i], ref[..., i], data_range=255, gaussian_weights=True,
                              sigma=1.5, use_sample_covariance=False)
        for i in range(x.shape[2])]))


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def save(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8)).save(path)


def kodak():
    return [(p.stem, load(p)) for p in sorted((DATA / "kodak").glob("kodim*.png"))]


def polyu():
    """Yields (name, noisy, clean) for the 100 cropped PolyU pairs."""
    for real in sorted((DATA / "polyu").glob("*_real.JPG")):
        name = real.name[: -len("_real.JPG")]
        yield name, load(real), load(real.with_name(name + "_mean.JPG"))


def sidd(limit=None):
    """Yields (name, noisy, clean) for the SIDD sRGB validation set: 40 images x 32 blocks of 256x256."""
    from scipy.io import loadmat
    noisy = loadmat(DATA / "sidd" / "ValidationNoisyBlocksSrgb.mat")["ValidationNoisyBlocksSrgb"]
    clean = loadmat(DATA / "sidd" / "ValidationGtBlocksSrgb.mat")["ValidationGtBlocksSrgb"]
    n = 0
    for i in range(noisy.shape[0]):
        for j in range(noisy.shape[1]):
            if limit is not None and n >= limit:
                return
            yield f"{i:02d}_{j:02d}", noisy[i, j].astype(np.float32), clean[i, j].astype(np.float32)
            n += 1


def add_channel_noise(clean, sigmas, seed):
    """Independent Gaussian noise per RGB channel (Kodak case 1/2 protocol)."""
    rng = np.random.default_rng(seed)
    return clean + rng.standard_normal(clean.shape).astype(np.float32) * np.asarray(sigmas, np.float32)
