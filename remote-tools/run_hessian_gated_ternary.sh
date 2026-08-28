#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:?output directory required}"
script_path="${2:-/root/PT2-LLM-official/hessian_gated_ternary_diagnostics.py}"
nsamples="${3:-12}"
tokens_per_sample="${4:-128}"

mkdir -p "$output_dir"
cd /root/PT2-LLM-official
export PYTHONPATH=/root/PT2-LLM-official
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PT2_DATA_ROOT=/root/PT2-LLM/data

/root/PT2-LLM/venv/bin/python "$script_path" \
  --model /root/models/Llama-2-7b-hf \
  --output-dir "$output_dir" \
  --layers 0,10,20,31 \
  --nsamples "$nsamples" \
  --tokens-per-sample "$tokens_per_sample" \
  --blocks-per-module 2 \
  --blocksize 128 \
  --max-steps 4 \
  2>&1 | tee "$output_dir/run.log"
