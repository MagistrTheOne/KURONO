#!/usr/bin/env bash
set -euo pipefail

python train_s1.py \
  --steps "${STEPS:-50}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --frames "${FRAMES:-65}" \
  --height "${HEIGHT:-256}" \
  --width "${WIDTH:-256}" \
  --precision "${PRECISION:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --mock-vae
