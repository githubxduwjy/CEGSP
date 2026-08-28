#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work

run_one() {
  local runid="$1"
  local fitoff="$2"
  local valoff="$3"
  local c4off="$4"
  mkdir -p "/root/tqgsp-runs/${runid}"
  echo "START ${runid} fit=${fitoff} val=${valoff} c4=${c4off}"
  CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/bin/python /root/tqgsp-work/cegsp_ce_gradient_4090.py \
      --run-id "${runid}" \
      --model facebook/opt-125m \
      --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
      --seq-len 128 \
      --batch-size 2 \
      --fit-batches 8 \
      --val-batches 8 \
      --untouched-batches 8 \
      --c4-untouched-batches 8 \
      --max-edits 64 \
      --grad-batches 1 \
      --support-topk 3 \
      --signflip-topk 3 \
      --k-sweep 2,3 \
      --fit-token-offset "${fitoff}" \
      --val-token-offset "${valoff}" \
      --c4-token-offset "${c4off}" \
      --out-dir /root/tqgsp-runs \
    2>&1 | tee "/root/tqgsp-runs/${runid}/console.log"
  echo "DONE ${runid}"
}

run_one CEGSP-04B-OPT125M-O0 0 0 0
run_one CEGSP-04B-OPT125M-O1 4096 4096 4096
run_one CEGSP-04B-OPT125M-O2 8192 8192 8192
