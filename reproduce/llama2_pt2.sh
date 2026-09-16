#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOH'
Run the representative Llama-2-7B PT2 + TernRefine/APG example.

Required environment variables:
  MODEL_PATH        Llama-2-7B path or Hugging Face identifier
  SIDECAR_DIR       exported PT2 sidecar directory
  REFERENCE_JSON    frozen PT2 reference metrics JSON
  PT2_ROOT          external PT2 checkout
  PT2_DATA_ROOT     PT2 calibration data root

Optional environment variables:
  PT2_CHECKPOINT    frozen PT2 full-state checkpoint; if set, skips PT2 rebuild
  OUT_DIR           output root (default: results/llama2_pt2_refine)
  RUN_ID            run id (default: LLAMA2-PT2-REFINE)
  PYTHON            Python executable (default: python)

This example uses the paper protocol: Q/K scope, one fitting backward per split, fixed QGP ranking, APG K={1,2,4,8,16,32,64}, and no reranking.
EOH
  exit 0
fi

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${SIDECAR_DIR:?SIDECAR_DIR is required}"
: "${REFERENCE_JSON:?REFERENCE_JSON is required}"
: "${PT2_ROOT:?PT2_ROOT is required}"
: "${PT2_DATA_ROOT:?PT2_DATA_ROOT is required}"

PYTHON_BIN="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-results/llama2_pt2_refine}"
RUN_ID="${RUN_ID:-LLAMA2-PT2-REFINE}"

CMD=(
  "${PYTHON_BIN}" examples/llama2_pt2_refine.py
  --model "${MODEL_PATH}"
  --sidecar-dir "${SIDECAR_DIR}"
  --reference-json "${REFERENCE_JSON}"
  --run-id "${RUN_ID}"
  --out-dir "${OUT_DIR}"
  --pt2-root "${PT2_ROOT}"
  --pt2-data-root "${PT2_DATA_ROOT}"
)

if [[ -n "${PT2_CHECKPOINT:-}" ]]; then
  CMD+=(--pt2-checkpoint "${PT2_CHECKPOINT}")
fi

"${CMD[@]}"
