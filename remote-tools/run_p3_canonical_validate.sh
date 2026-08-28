#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work
mkdir -p /root/tqgsp-runs/CEGSP-V2-P3-CANONICAL-VALIDATE
LOG=/root/tqgsp-runs/CEGSP-V2-P3-CANONICAL-VALIDATE/console.log

{
  echo START_P3_CANONICAL_VALIDATE $(date -Is)
  echo P3_RULE="Direct ternary -> one quantized-point CE gradient -> Q/K -> support relocation primary -> small fixed top-k -> untouched evaluation"

  echo P3A_START $(date -Is)
  /opt/conda/bin/python cegsp_ce_gradient_4090.py \
    --run-id CEGSP-V2-P3A-OPT350M-CANONICAL-OFFSET2 \
    --model facebook/opt-350m \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
    --seq-len 128 \
    --batch-size 1 \
    --fit-batches 8 \
    --val-batches 8 \
    --untouched-batches 64 \
    --c4-untouched-batches 32 \
    --fit-token-offset 8192 \
    --val-token-offset 8192 \
    --c4-token-offset 16384 \
    --k-sweep 6 \
    --support-topk 4 \
    --signflip-topk 4 \
    --max-edits 64 \
    --grad-batches 1 \
    --random-control-repeats 1 \
    --dtype bf16 \
    --out-dir /root/tqgsp-runs
  echo P3A_DONE $(date -Is)

  echo P3B_START $(date -Is)
  TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /opt/conda/bin/python cegsp_ce_gradient_4090.py \
    --run-id CEGSP-V2-P3B-PYTHIA1B-CANONICAL-OFFSET1 \
    --model EleutherAI/pythia-1b \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --seq-len 128 \
    --batch-size 1 \
    --fit-batches 8 \
    --val-batches 8 \
    --untouched-batches 32 \
    --c4-untouched-batches 0 \
    --fit-token-offset 4096 \
    --val-token-offset 4096 \
    --k-sweep 4 \
    --support-topk 4 \
    --signflip-topk 4 \
    --max-edits 64 \
    --grad-batches 1 \
    --random-control-repeats 1 \
    --dtype bf16 \
    --out-dir /root/tqgsp-runs
  echo P3B_DONE $(date -Is)

  echo DONE_P3_CANONICAL_VALIDATE $(date -Is)
} 2>&1 | tee -a "$LOG"
