#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/root/models/Llama-2-7b-hf}"
root_dir="${2:-/root/PT2-LLM-official/aris-runs/r054_cancellation_audit_20260824}"
score_script="${3:-/root/PT2-LLM-official/r048_distribution_holdout_gate.py}"
analyze_script="${4:-/root/PT2-LLM-official/r054_analyze_cancellation.py}"

cd /root/PT2-LLM-official
export PT2_DATA_ROOT=/root/PT2-LLM/data
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$root_dir"

for first_layer in 0 10 20 30; do
  second_layer=$((first_layer + 1))
  output_dir="$root_dir/layers_${first_layer}_${second_layer}"
  /root/PT2-LLM/venv/bin/python "$score_script" \
    --model "$model_path" \
    --output-dir "$output_dir" \
    --calib-nsamples 8 \
    --gate-nsamples 8 \
    --test-nsamples 8 \
    --score-start 72 \
    --seqlen 2048 \
    --blocksize 128 \
    --seed 0 \
    --validation-fraction 0.25 \
    --max-steps 4 \
    --window-layers "${first_layer},${second_layer}" \
    --mean-epsilon 0.0 \
    --cvar-epsilon 0.0 \
    --compute-cancellation
done

/root/PT2-LLM/venv/bin/python "$analyze_script" --root "$root_dir"
