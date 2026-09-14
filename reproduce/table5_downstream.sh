#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Reproduce Table 5: standardized 0-shot downstream benchmark.

Set MODEL_FAMILY=llama or qwen and provide the checkpoint/sidecar/patch paths
expected by the corresponding downstream runner.
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"

case "${MODEL_FAMILY}" in
  llama)
    "${PYTHON_BIN}" remote-tools/cegsp_e2_llama_lm_eval_downstream.py "$@"
    ;;
  qwen)
    "${PYTHON_BIN}" remote-tools/cegsp_e2_qwen_lm_eval_downstream.py "$@"
    ;;
  *)
    echo "MODEL_FAMILY must be llama or qwen" >&2
    exit 2
    ;;
esac

