#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-ecg_arrhythmia_combined_seed0}"
mkdir -p outputs/results outputs/console
python3 -u train.py configs/datasets/ecg_arrhythmia_combined.toml "$RUN_ID" \
  2>&1 | tee "outputs/console/${RUN_ID}.log"
