#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work

run_one() {
  local runid="$1"
  local fitoff="$2"
  local valoff="$3"
  mkdir -p "/root/tqgsp-runs/${runid}"
  echo "START ${runid} fit=${fitoff} val=${valoff}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/bin/python /root/tqgsp-work/cegsp_ce_gradient_4090.py \
      --run-id "${runid}" \
      --model facebook/opt-350m \
      --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
      --seq-len 128 \
      --batch-size 2 \
      --fit-batches 8 \
      --val-batches 8 \
      --untouched-batches 8 \
      --max-edits 64 \
      --grad-batches 1 \
      --support-topk 6 \
      --signflip-topk 6 \
      --k-sweep 4,6,8,12 \
      --fit-token-offset "${fitoff}" \
      --val-token-offset "${valoff}" \
      --out-dir /root/tqgsp-runs \
    2>&1 | tee "/root/tqgsp-runs/${runid}/console.log"
  echo "DONE ${runid}"
}

run_one CEGSP-02A-O0 0 0
run_one CEGSP-02A-O1 4096 4096
run_one CEGSP-02A-O2 8192 8192
