#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work
RUN_ID=CEGSP-V2-P4R2-OPT350M-STRICT-EDITMATCH-SCOPE-OFFSET2
RUN_DIR=/root/tqgsp-runs/$RUN_ID
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"

{
  echo START_P4R2_FAIRNESS $(date -Is)
  echo RUN_ID=$RUN_ID
  echo PURPOSE="Strict edit matching near CEGSP 768 changes + QAT update scope check; CEGSP unchanged."
  /opt/conda/bin/python cegsp_v2_p4r_fairness_4090.py \
    --run-id "$RUN_ID" \
    --model facebook/opt-350m \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
    --seq-len 128 \
    --batch-size 1 \
    --fit-batches 8 \
    --val-batches 8 \
    --untouched-batches 64 \
    --fit-token-offset 8192 \
    --val-token-offset 8192 \
    --c4-untouched-batches 32 \
    --c4-token-offset 16384 \
    --group-size 128 \
    --threshold-factor 0.7 \
    --max-edits 64 \
    --grad-batches 1 \
    --layer-topk 6 \
    --etas 3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,1.5e-4,2e-4,3e-4,5e-4,7e-4,1e-3 \
    --dtype bf16 \
    --seed 20260828 \
    --out-dir /root/tqgsp-runs
  echo DONE_P4R2_FAIRNESS $(date -Is)
} 2>&1 | tee -a "$LOG"
