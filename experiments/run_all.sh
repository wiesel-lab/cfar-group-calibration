#!/usr/bin/env bash
# Reproduces every result of the paper from the raw images.
#   bash experiments/run_all.sh [workers]
# Stage 1 (scene fits) is the slow part: about 50 CPU-minutes per planting.
# Every stage skips work that is already cached, so it can be restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-4}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

bash data/download_data.sh
for s in 0 1 2; do
  python experiments/build_scenes.py --seed $s --workers "$W"
  python experiments/evaluate.py --seed $s --workers "$W"
done
python experiments/table1.py
python experiments/figure1.py
python experiments/section5_numbers.py | tee results/section5_numbers.txt
