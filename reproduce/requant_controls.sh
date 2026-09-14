#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run reconstruction-guided ReQuant comparison controls.

Usage:
  MODE=style bash reproduce/requant_controls.sh
  MODE=faithful bash reproduce/requant_controls.sh
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON:-python}"
MODE="${MODE:-style}"

case "${MODE}" in
  style)
    "${PYTHON_BIN}" remote-tools/cegsp_requant_comparison_4090.py "$@"
    ;;
  faithful)
    "${PYTHON_BIN}" remote-tools/cegsp_requant_faithful_4090.py "$@"
    ;;
  *)
    echo "MODE must be style or faithful" >&2
    exit 2
    ;;
esac

