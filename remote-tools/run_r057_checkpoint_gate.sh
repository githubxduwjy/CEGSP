#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/PT2-LLM-official/aris-runs/r057_checkpoint_gate_20260824"
MODEL="/root/models/Llama-2-7b-hf"
PY="/root/PT2-LLM/venv/bin/python"

mkdir -p "$ROOT"
for FIRST in 10 30; do
  SECOND=$((FIRST + 1))
  "$PY" /root/PT2-LLM-official/remote-tools/r048_distribution_holdout_gate.py \
    --model "$MODEL" \
    --output-dir "$ROOT/layers_${FIRST}_${SECOND}" \
    --calib-nsamples 8 \
    --gate-nsamples 8 \
    --test-nsamples 8 \
    --score-start 88 \
    --seqlen 2048 \
    --blocksize 128 \
    --seed 0 \
    --validation-fraction 0.25 \
    --max-steps 4 \
    --window-layers "${FIRST},${SECOND}" \
    --mean-epsilon 0 \
    --cvar-epsilon 0
done

"$PY" /root/PT2-LLM-official/remote-tools/r057_analyze_checkpoint_gate.py --root "$ROOT"
