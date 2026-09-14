#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOH'
Reproduce the Table 3 mechanism experiment: quantized-point vs full-precision-point task-gradient ranking.

Environment variables:
  MODEL_PATH      OPT-350M path or Hugging Face id (default: facebook/opt-350m)
  OUT_DIR         output root (default: results/table3_opt350m)
  RUN_ID          run id (default: TABLE3-OPT350M)
  PYTHON          Python executable (default: python)

The experiment keeps Q0, CPSR moves, task objective, and patch size fixed; it changes only the gradient evaluation point.
EOH
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-facebook/opt-350m}"
OUT_DIR="${OUT_DIR:-results/table3_opt350m}"
RUN_ID="${RUN_ID:-TABLE3-OPT350M}"

"${PYTHON_BIN}" examples/table3_opt350m.py \
  --model "${MODEL_PATH}" \
  --run-id "${RUN_ID}" \
  --out-dir "${OUT_DIR}"
