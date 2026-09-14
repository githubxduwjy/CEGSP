#!/usr/bin/env bash
set -euo pipefail

cd /root/CEGSP-P11-P12/remote-tools

for off in 0 1024 2048 3072 4096; do
  run="CEGSP-E1-QPOINT-VS-FPPOINT-OPT350M-20260910-42196-offset${off}"
  out="/root/tqgsp-runs/${run}"
  mkdir -p "${out}"
  screen -dmS "cegsp_e1_off${off}_42196" bash -lc \
    "cd /root/CEGSP-P11-P12/remote-tools && CUDA_VISIBLE_DEVICES=0 python3 -u cegsp_e1_quantized_vs_fp_gradient_opt350m.py --run-id ${run} --offsets ${off} --fit-batches 8 --val-batches 8 --w2-batches 8 --c4-batches 8 --single-sample 512 --seq-len 128 --batch-size 2 --grad-batches 1 2>&1 | tee ${out}/screen.log"
done

screen -ls
