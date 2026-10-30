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


def add_channel_noise(clean, sigmas, seed):
    """Independent Gaussian noise per RGB channel (Kodak case 1/2 protocol)."""
    rng = np.random.default_rng(seed)
    return clean + rng.standard_normal(clean.shape).astype(np.float32) * np.asarray(sigmas, np.float32)
