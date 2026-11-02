#!/usr/bin/env bash
# Final-phase experiments (noise estimators, PolyU, SIDD), logging to results/.
# Meant for a CUDA machine with many CPU cores (CBM3D runs in parallel CPU workers).
#   PYTHON=python scripts/run_final.sh          # after: pip install -r requirements.txt
#   FULL=1 PYTHON=python scripts/run_final.sh   # also rerun the progress-report experiments (run_all.sh)
#   ORACLE=1 ...                                # also the oracle-sigma run on SIDD (~11x Haar-tSVD time)
# Resumable where it matters (sidd.py); commit results/ afterwards (data/ is git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
export HAARTSVD_DEVICE="${HAARTSVD_DEVICE:-cuda}"
PY="${PYTHON:-uv run python}"
mkdir -p results

SIDD=1 bash scripts/setup_data.sh || echo "** SIDD download failed: continuing without SIDD (rerun this script later) **"
$PY scripts/check_device.py
[[ "${FULL:-0}" == "1" ]] && PYTHON="$PY" bash scripts/run_all.sh "$HAARTSVD_DEVICE"

$PY experiments/noise_est.py --part kodak 2>&1 | tee results/noise_est_kodak.log
$PY experiments/noise_est.py --part polyu --cbm3d 2>&1 | tee results/noise_est_polyu.log
if [[ -f data/sidd/ValidationGtBlocksSrgb.mat ]]; then
  $PY experiments/sidd.py ${ORACLE:+--oracle} 2>&1 | tee results/sidd.log
else
  echo "** no SIDD data: skipped sidd.py **"
fi
$PY scripts/fill_report_tables.py > /dev/null || true
echo "done: git add results && git commit -m 'Final-phase results' && git push"
