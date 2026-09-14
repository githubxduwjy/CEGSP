#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Reproduce the APG validation-selected patch-growth experiments.

Use the large-model ordinary-affine runner by default. For strong PT2 runs,
call the PT2 replication scripts listed in docs/SCRIPT_INDEX.md with a prepared
sidecar/checkpoint.
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/root/Llama-2-7b-hf}"
OUT_DIR="${OUT_DIR:-results/table4_apg}"

"${PYTHON_BIN}" remote-tools/cegsp_e1_cross_model_apg.py \
  --model "${MODEL_PATH}" \
  --out-dir "${OUT_DIR}"

