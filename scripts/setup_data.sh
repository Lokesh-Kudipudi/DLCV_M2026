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
  mkdir -p data/sidd
  curl -fL -o data/sidd/test.zip \
    "https://drive.usercontent.google.com/download?id=11vfqV-lqousZTuAit1Qkqghiv_taY0KZ&export=download&confirm=t"
  unzip -j -o data/sidd/test.zip 'test/SIDD/*.mat' -d data/sidd
  rm data/sidd/test.zip
  echo "sidd: $(ls data/sidd)"
fi
echo "kodak: $(ls data/kodak | wc -l) files, polyu: $(ls data/polyu | wc -l) files (expect 24 and 200)"
