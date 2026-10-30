#!/usr/bin/env bash
# Run every experiment and figure, logging to results/.
#   scripts/run_all.sh            # auto device: cuda > mps > cpu
#   scripts/run_all.sh cuda       # force CUDA
#   scripts/run_all.sh mps        # Apple GPU
#   FAST=1 scripts/run_all.sh     # skip the slow Local-tSVD+PCA baseline on the full sets
set -euo pipefail
cd "$(dirname "$0")/.."
export HAARTSVD_DEVICE="${1:-auto}"
PY="${PYTHON:-uv run python}"
mkdir -p results

METHODS=("Haar-tSVD" "CBM3D" "Local-tSVD+PCA")
[[ "${FAST:-0}" == "1" ]] && METHODS=("Haar-tSVD" "CBM3D")

$PY scripts/check_device.py
$PY experiments/kodak.py --methods "${METHODS[@]}" 2>&1 | tee results/kodak.log
$PY experiments/polyu.py --methods "${METHODS[@]}" 2>&1 | tee results/polyu.log
$PY experiments/ablations.py 2>&1 | tee results/ablations.log
$PY experiments/figures.py 2>&1 | tee results/figures.log
