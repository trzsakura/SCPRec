#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_DIR:?Set DATASET_DIR to a supported saved dataset}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/experts}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
mkdir -p "${CHECKPOINT_DIR}"

python -u sasrec.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_path "${CHECKPOINT_DIR}/sasrec.pt" \
  --device "${DEVICE}" \
  --validation_fraction "${VALIDATION_FRACTION}" \
  --seed "${SEED}"

python -u lightgcn.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_path "${CHECKPOINT_DIR}/lightgcn.pt" \
  --device "${DEVICE}" \
  --validation_fraction "${VALIDATION_FRACTION}" \
  --seed "${SEED}"

python -u fmlprec.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_path "${CHECKPOINT_DIR}/fmlprec.pt" \
  --device "${DEVICE}" \
  --validation_fraction "${VALIDATION_FRACTION}" \
  --seed "${SEED}"

python -u ease.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_path "${CHECKPOINT_DIR}/ease.pt" \
  --fit_device "${DEVICE}" \
  --validation_fraction "${VALIDATION_FRACTION}" \
  --seed "${SEED}"
