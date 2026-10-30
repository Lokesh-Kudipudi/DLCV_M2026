#!/usr/bin/env bash
# Download the public datasets into data/ (Kodak 24, PolyU cropped 100 pairs).
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
echo "kodak: $(ls data/kodak | wc -l) files, polyu: $(ls data/polyu | wc -l) files (expect 24 and 200)"
