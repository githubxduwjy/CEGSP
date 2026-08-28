#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work

run_opt350m() {
  local runid="CEGSP-05B-OPT350M-O0-U32-RANDOM"
  mkdir -p "/root/tqgsp-runs/${runid}"
  echo "START ${runid}"
  CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/bin/python /root/tqgsp-work/cegsp_ce_gradient_4090.py \
      --run-id "${runid}" \
      --model facebook/opt-350m \
      --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
      --seq-len 128 \
      --batch-size 2 \
      --fit-batches 8 \
      --val-batches 8 \
      --untouched-batches 32 \
      --c4-untouched-batches 32 \
      --max-edits 64 \
      --grad-batches 1 \
      --support-topk 6 \
      --signflip-topk 6 \
      --k-sweep 4,6 \
      --random-control-repeats 3 \
      --fit-token-offset 0 \
      --val-token-offset 0 \
      --c4-token-offset 0 \
      --out-dir /root/tqgsp-runs \
    2>&1 | tee "/root/tqgsp-runs/${runid}/console.log"
  echo "DONE ${runid}"
}

run_opt125m() {
  local runid="CEGSP-05B-OPT125M-O0-U32-RANDOM"
  mkdir -p "/root/tqgsp-runs/${runid}"
  echo "START ${runid}"
  CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/bin/python /root/tqgsp-work/cegsp_ce_gradient_4090.py \
      --run-id "${runid}" \
      --model facebook/opt-125m \
      --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
      --seq-len 128 \
      --batch-size 2 \
      --fit-batches 8 \
      --val-batches 8 \
      --untouched-batches 32 \
      --c4-untouched-batches 32 \
      --max-edits 64 \
      --grad-batches 1 \
      --support-topk 3 \
      --signflip-topk 3 \
      --k-sweep 2,3 \
      --random-control-repeats 3 \
      --fit-token-offset 0 \
      --val-token-offset 0 \
      --c4-token-offset 0 \
      --out-dir /root/tqgsp-runs \
    2>&1 | tee "/root/tqgsp-runs/${runid}/console.log"
  echo "DONE ${runid}"
}

run_opt350m
run_opt125m
