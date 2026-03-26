#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-${KURONO_DATA_PATH:-}}"
: "${DATA_PATH:?Set DATA_PATH or KURONO_DATA_PATH to a video file or directory}"

python train_s1.py \
  --steps "${STEPS:-50}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --frames "${FRAMES:-65}" \
  --height "${HEIGHT:-256}" \
  --width "${WIDTH:-256}" \
  --precision "${PRECISION:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --data-path "${DATA_PATH}" \
  --mock-vae
