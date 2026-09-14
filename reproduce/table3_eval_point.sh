#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Reproduce Table 3: quantized-point vs full-precision-point gradient.

Environment variables:
  MODEL_PATH      OPT-350M path or Hugging Face id (default: facebook/opt-350m)
  OUT_DIR         output root (default: results/table3_eval_point)
  PYTHON          Python executable (default: python)

This command keeps Q0, CPSR moves, task objective, and patch size fixed; it
changes only the gradient evaluation point.
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-facebook/opt-350m}"
OUT_DIR="${OUT_DIR:-results/table3_eval_point}"

"${PYTHON_BIN}" remote-tools/cegsp_e1_quantized_vs_fp_gradient_opt350m.py \
  --model "${MODEL_PATH}" \
  --out-dir "${OUT_DIR}"

