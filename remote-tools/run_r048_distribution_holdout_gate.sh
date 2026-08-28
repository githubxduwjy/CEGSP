#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/root/models/Llama-2-7b-hf}"
out_dir="${2:-/root/PT2-LLM-official/aris-runs/r048_distribution_holdout_gate_20260824}"
script_path="${3:-/root/PT2-LLM-official/r048_distribution_holdout_gate.py}"

cd /root/PT2-LLM-official
export PT2_DATA_ROOT=/root/PT2-LLM/data
export CUDA_VISIBLE_DEVICES=0

/root/PT2-LLM/venv/bin/python "$script_path" \
  --model "$model_path" \
  --output-dir "$out_dir" \
  --calib-nsamples 8 \
  --gate-nsamples 4 \
  --test-nsamples 4 \
  --seqlen 2048 \
  --blocksize 128 \
  --seed 0 \
  --validation-fraction 0.25 \
  --max-steps 4 \
  --window-layers 0,1 \
  --mean-epsilon 0.0 \
  --cvar-epsilon 0.0
