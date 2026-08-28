#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/PT2-LLM-official/aris-runs/r058_checkpoint_veto_seed1_20260824"
MODEL="/root/models/Llama-2-7b-hf"
PY="/root/PT2-LLM/venv/bin/python"

export PYTHONPATH="/root/PT2-LLM-official:/root/PT2-LLM-official/remote-tools:${PYTHONPATH:-}"
export PT2_DATA_ROOT="/root/PT2-LLM/data"

mkdir -p "$ROOT"

"$PY" /root/PT2-LLM-official/remote-tools/r048_distribution_holdout_gate.py \
  --model "$MODEL" \
  --output-dir "$ROOT" \
  --calib-nsamples 8 \
  --gate-nsamples 8 \
  --test-nsamples 8 \
  --score-start 120 \
  --seqlen 2048 \
  --blocksize 128 \
  --seed 1 \
  --validation-fraction 0.25 \
  --max-steps 4 \
  --window-layers "10,11" \
  --mean-epsilon 0 \
  --cvar-epsilon 0

"$PY" /root/PT2-LLM-official/remote-tools/r058_analyze_checkpoint_veto.py \
  --metrics "$ROOT/metrics.json" \
  --output "$ROOT/r058_summary.json"
