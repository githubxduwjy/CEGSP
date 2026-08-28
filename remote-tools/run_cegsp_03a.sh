#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work

RUNID="CEGSP-03A-C4TRANSFER"
mkdir -p "/root/tqgsp-runs/${RUNID}"

echo "START ${RUNID}"
CUDA_VISIBLE_DEVICES=0 \
  /opt/conda/bin/python /root/tqgsp-work/cegsp_ce_gradient_4090.py \
    --run-id "${RUNID}" \
    --model facebook/opt-350m \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
    --seq-len 128 \
    --batch-size 2 \
    --fit-batches 8 \
    --val-batches 8 \
    --untouched-batches 8 \
    --c4-untouched-batches 8 \
    --max-edits 64 \
    --grad-batches 1 \
    --support-topk 6 \
    --signflip-topk 6 \
    --k-sweep 4,6 \
    --fit-token-offset 0 \
    --val-token-offset 0 \
    --c4-token-offset 0 \
    --out-dir /root/tqgsp-runs \
  2>&1 | tee "/root/tqgsp-runs/${RUNID}/console.log"
echo "DONE ${RUNID}"
