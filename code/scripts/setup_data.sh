#!/usr/bin/env bash
# Download the public datasets into data/ (Kodak 24, PolyU cropped 100 pairs).
#   SIDD=1 bash scripts/setup_data.sh   # also SIDD sRGB validation blocks (440 MB zip, ~500 MB on disk)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/kodak data/polyu
for i in $(seq -w 1 24); do
  [[ -f data/kodak/kodim$i.png ]] || curl -sfL -o data/kodak/kodim$i.png "https://r0k.us/graphics/kodak/kodak/kodim$i.png" &
done
wait
curl -s "https://api.github.com/repos/csjunxu/PolyU-Real-World-Noisy-Images-Dataset/contents/CroppedImages" \
  | python3 -c "import sys,json; [print(x['download_url']) for x in json.load(sys.stdin)]" \
  | (cd data/polyu && xargs -n 1 -P 16 curl -sfLO)
if [[ "${SIDD:-0}" == "1" && ! -f data/sidd/ValidationGtBlocksSrgb.mat ]]; then
  mkdir -p data/sidd
  python3 -m pip install -q gdown lmdb >/dev/null 2>&1 || true
  # 1) Google Drive: test.zip of the Restormer repo (the official SIDD server is offline), which holds
  #    the unmodified ValidationNoisyBlocksSrgb.mat / ValidationGtBlocksSrgb.mat under test/SIDD/.
  #    Often fails with "quota exceeded".
  ID=11vfqV-lqousZTuAit1Qkqghiv_taY0KZ
  python3 -m gdown "$ID" -O data/sidd/test.zip || true
  # 2) Fallback: the same 1280 blocks as lossless PNGs in BasicSR/NAFNet LMDBs on Hugging Face.
  if ! python3 -c "import zipfile,sys; sys.exit(not zipfile.is_zipfile('data/sidd/test.zip'))" 2>/dev/null; then
    echo "Google Drive failed; using the Hugging Face LMDB copy"
    HF=https://huggingface.co/datasets/talib-sid/sidd-val/resolve/main/val
    for d in input_crops gt_crops; do
      mkdir -p data/sidd/$d.lmdb
      curl -fL -o data/sidd/$d.lmdb/data.mdb "$HF/$d.lmdb/data.mdb"
    done
  fi
  python3 - <<'PY'
import os, shutil, zipfile
import numpy as np
from scipy.io import savemat
D = "data/sidd/"
if zipfile.is_zipfile(D + "test.zip"):
    with zipfile.ZipFile(D + "test.zip") as z:
        for n in z.namelist():
            if n.startswith("test/SIDD/") and n.endswith(".mat"):
                with z.open(n) as src, open(D + n.split("/")[-1], "wb") as dst:
                    shutil.copyfileobj(src, dst)
else:
    # Rebuild the two (40, 32, 256, 256, 3) uint8 arrays; block k -> (k // 32, k % 32). Noisy and
    # clean are paired by the same key, so the averages do not depend on that order.
    import io
    import lmdb
    from PIL import Image
    out = {}
    for d, key in (("input_crops", "ValidationNoisyBlocksSrgb"), ("gt_crops", "ValidationGtBlocksSrgb")):
        arr = np.zeros((40, 32, 256, 256, 3), np.uint8)
        with lmdb.open(D + d + ".lmdb", readonly=True, lock=False, subdir=True).begin() as txn:
            for k in range(1280):
                buf = txn.get(f"ValidationBlocksSrgb_{k}".encode()) or txn.get(f"ValidationBlocksSrgb_{k}.png".encode())
                arr[k // 32, k % 32] = np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
        savemat(D + key + ".mat", {key: arr}, do_compression=False)
        out[key] = arr
        shutil.rmtree(D + d + ".lmdb")
    mse = np.mean((out["ValidationNoisyBlocksSrgb"].astype(float) - out["ValidationGtBlocksSrgb"]) ** 2, axis=(2, 3, 4))
    print(f"SIDD val noisy PSNR {np.mean(10 * np.log10(255 ** 2 / mse)):.2f} dB (expected about 23.7)")
if os.path.exists(D + "test.zip"):
    os.remove(D + "test.zip")
PY
  echo "sidd: $(ls data/sidd)"
fi
echo "kodak: $(ls data/kodak | wc -l) files, polyu: $(ls data/polyu | wc -l) files (expect 24 and 200)"
