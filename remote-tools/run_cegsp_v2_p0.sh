#!/usr/bin/env bash
set -euo pipefail

cd /root/tqgsp-work

runid="CEGSP-V2-P0-OPT125M"
mkdir -p "/root/tqgsp-runs/${runid}"
CUDA_VISIBLE_DEVICES=0 \
  /opt/conda/bin/python /root/tqgsp-work/cegsp_v2_p0_gap_4090.py \
    --run-id "${runid}" \
    --model facebook/opt-125m \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --seq-len 128 \
    --batch-size 2 \
    --fit-batches 8 \
    --val-batches 8 \
    --untouched-batches 16 \
    --group-size 128 \
    --threshold-factor 0.7 \
    --max-edits 64 \
    --grad-batches 1 \
    --layer-topk 3 \
    --score-layers 0,6,11 \
    --score-candidates 32 \
    --qat-etas 0.0,0.01,0.03,0.1,0.3,1.0 \
    --qat-steps 1,4 \
    --dtype bf16 \
    --out-dir /root/tqgsp-runs \
  2>&1 | tee "/root/tqgsp-runs/${runid}/console.log"
