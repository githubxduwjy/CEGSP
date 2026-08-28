#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/PT2-LLM-official/aris-runs/r057a_hparam_gate_20260824"
MODEL="/root/models/Llama-2-7b-hf"
PY="/root/PT2-LLM/venv/bin/python"
RUNNER="/root/PT2-LLM-official/remote-tools/r048_distribution_holdout_gate.py"

export PYTHONPATH="/root/PT2-LLM-official:/root/PT2-LLM-official/remote-tools:${PYTHONPATH:-}"
export PT2_DATA_ROOT="/root/PT2-LLM/data"

mkdir -p "$ROOT"

CONFIG_IDS=(H0 H1 H2 H3 H4)
NSAMPLES=(8 8 8 16 8)
BLOCKSIZES=(128 128 128 128 64)
STEPS=(4 2 8 4 4)

for INDEX in "${!CONFIG_IDS[@]}"; do
  CONFIG_ID="${CONFIG_IDS[$INDEX]}"
  if [[ -s "$ROOT/$CONFIG_ID/metrics.json" ]]; then
    echo "Skipping completed configuration $CONFIG_ID"
    continue
  fi
  "$PY" "$RUNNER" \
    --model "$MODEL" \
    --output-dir "$ROOT/$CONFIG_ID" \
    --calib-nsamples "${NSAMPLES[$INDEX]}" \
    --gate-nsamples 8 \
    --test-nsamples 8 \
    --score-start 88 \
    --seqlen 2048 \
    --blocksize "${BLOCKSIZES[$INDEX]}" \
    --seed 0 \
    --validation-fraction 0.25 \
    --max-steps "${STEPS[$INDEX]}" \
    --window-layers "10,11" \
    --mean-epsilon 0 \
    --cvar-epsilon 0
done

"$PY" /root/PT2-LLM-official/remote-tools/r057a_analyze_hparams.py --root "$ROOT"
