#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-cegsp_p5a_affine_adapter_opt350m_$(date +%Y%m%d_%H%M%S)}"
cd /root/tqgsp-work
/opt/conda/bin/python cegsp_p5a_affine_adapter_feasibility_4090.py \
  --model facebook/opt-350m \
  --run-id "${RUN_ID}" \
  --layers 13 \
  --seq-len 128 \
  --batch-size 2 \
  --fit-batches 8 \
  --val-batches 8 \
  --untouched-batches 8 \
  --c4-untouched-batches 8 \
  --group-size 128 \
  --threshold-factor 0.75 \
  --max-edits 64 \
  --grad-batches 1 \
  --dtype bf16 \
  --seed 20260828 \
  --out-dir /root/tqgsp-runs
