"""Denoise one image with Haar-tSVD.

    uv run python denoise.py noisy.png out.png            # sigma estimated (MAD x kappa)
    uv run python denoise.py noisy.png out.png --sigma 20
"""
import argparse

from haartsvd import Params, haar_tsvd
from haartsvd.utils import estimate_sigma, load, save

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("output")
ap.add_argument("--sigma", type=float, help="noise std on the 0-255 scale (default: estimated)")
ap.add_argument("--kappa", type=float, default=2.0, help="sigma = kappa * MAD estimate when --sigma is not given")
ap.add_argument("--tau-scale", type=float, default=0.45)
ap.add_argument("--ps", type=int, default=8)
ap.add_argument("--K", type=int, default=32)
ap.add_argument("--sr", type=int, default=20)
ap.add_argument("--step", type=int, default=4)
args = ap.parse_args()

img = load(args.input)
sigma = args.sigma if args.sigma is not None else args.kappa * estimate_sigma(img)
r = haar_tsvd(img, sigma, Params(ps=args.ps, K=args.K, sr=args.sr, step=args.step, tau_scale=args.tau_scale))
save(r.image, args.output)
print(f"sigma={sigma:.2f}  time={r.timings['total']:.2f}s  -> {args.output}")
