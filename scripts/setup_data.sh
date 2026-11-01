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
  # The official SIDD server is offline; this is the test.zip of the Restormer repo, which holds the
  # unmodified ValidationNoisyBlocksSrgb.mat / ValidationGtBlocksSrgb.mat under test/SIDD/.
  # gdown handles Google Drive's confirmation page; plain curl is the fallback.
  mkdir -p data/sidd
  ID=11vfqV-lqousZTuAit1Qkqghiv_taY0KZ
  python3 -m pip install -q gdown >/dev/null 2>&1 || true
  python3 -m gdown "$ID" -O data/sidd/test.zip \
    || curl -fL -o data/sidd/test.zip "https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t"
  python3 - <<'PY'
import shutil, sys, zipfile
path = "data/sidd/test.zip"
if not zipfile.is_zipfile(path):
    sys.exit("SIDD download is not a zip (Google Drive quota or confirmation page?). First bytes:\n"
             + open(path, "rb").read(400).decode("utf-8", "replace"))
with zipfile.ZipFile(path) as z:
    for n in z.namelist():
        if n.startswith("test/SIDD/") and n.endswith(".mat"):
            with z.open(n) as src, open("data/sidd/" + n.split("/")[-1], "wb") as dst:
                shutil.copyfileobj(src, dst)
PY
  rm data/sidd/test.zip
  echo "sidd: $(ls data/sidd)"
fi
echo "kodak: $(ls data/kodak | wc -l) files, polyu: $(ls data/polyu | wc -l) files (expect 24 and 200)"
