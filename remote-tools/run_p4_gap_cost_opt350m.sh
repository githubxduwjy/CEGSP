#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work
RUN_ID=CEGSP-V2-P4-OPT350M-GAP-COST-OFFSET2
RUN_DIR=/root/tqgsp-runs/$RUN_ID
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"

{
  echo START_P4_GAP_COST $(date -Is)
  echo RUN_ID=$RUN_ID
  echo PURPOSE="Matched direct / CEGSP / one-step QAT / small-step QAT gap-cost closure; no new CEGSP module."
  /opt/conda/bin/python cegsp_v2_p4_gap_cost_4090.py \
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
    --score-layers 13,17,14,19 \
    --score-candidates 32 \
    --qat-etas 0.0,0.003,0.01,0.03,0.1 \
    --qat-steps 1,10,50 \
    --dtype bf16 \
    --seed 20260828 \
    --out-dir /root/tqgsp-runs
  echo DONE_P4_GAP_COST $(date -Is)
} 2>&1 | tee -a "$LOG"
