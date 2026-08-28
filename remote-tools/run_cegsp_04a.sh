#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work

run_one() {
  local edits="$1"
  local runid="CEGSP-04A-E${edits}"
  mkdir -p "/root/tqgsp-runs/${runid}"
  echo "START ${runid} max_edits=${edits}"
  CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/bin/python /root/tqgsp-work/cegsp_ce_gradient_4090.py \
      --run-id "${runid}" \
      --model facebook/opt-350m \
      --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
      --seq-len 128 \
      --batch-size 2 \
      --fit-batches 8 \
      --val-batches 8 \
      --untouched-batches 8 \
      --c4-untouched-batches 8 \
      --max-edits "${edits}" \
      --grad-batches 1 \
      --support-topk 6 \
      --signflip-topk 6 \
      --k-sweep 4,6 \
      --fit-token-offset 0 \
      --val-token-offset 0 \
      --c4-token-offset 0 \
      --out-dir /root/tqgsp-runs \
    2>&1 | tee "/root/tqgsp-runs/${runid}/console.log"
  echo "DONE ${runid}"
}

run_one 16
run_one 32
run_one 64
run_one 128
