#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-check}"
MODEL_KEY="${2:-llama}"

ROOT="${CEGSP_REMOTE_ROOT:-/root/CEGSP-code}"
PYTHON_BIN="${CEGSP_PYTHON:-/opt/conda/bin/python}"
OUT_ROOT="${CEGSP_OUT_ROOT:-/root/tqgsp-runs}"
PORT_TAG="${P8_PORT_TAG:-42079}"
TASKS="${P8_TASKS:-all6}"
MAX_EXAMPLES="${P8_MAX_EXAMPLES:-128}"
EVAL_BATCH_SIZE="${P8_EVAL_BATCH_SIZE:-8}"

case "${MODEL_KEY}" in
  llama|llama2|llama2-7b)
    MODEL_PATH="${LLAMA_MODEL_PATH:-/root/Llama-2-7b-hf}"
    RUN_ID="cegsp_p8a_downstream_llama2_7b_a100_20260901_${PORT_TAG}"
    SCREEN_NAME="p8a_downstream_llama_${PORT_TAG}"
    ;;
  qwen|qwen3|qwen3-8b)
    MODEL_PATH="${QWEN_MODEL_PATH:-/root/Qwen3-8B}"
    RUN_ID="cegsp_p8a_downstream_qwen3_8b_a100_20260901_${PORT_TAG}"
    SCREEN_NAME="p8a_downstream_qwen_${PORT_TAG}"
    ;;
  *)
    echo "unknown MODEL_KEY=${MODEL_KEY}; use llama or qwen" >&2
    exit 2
    ;;
esac

RUN_DIR="${OUT_ROOT}/${RUN_ID}"
LOG_PATH="${RUN_DIR}/screen.log"

check_common() {
  test -d "${ROOT}" || { echo "missing code root: ${ROOT}" >&2; exit 3; }
  test -f "${ROOT}/remote-tools/cegsp_p8_downstream.py" || { echo "missing P8 script" >&2; exit 3; }
  test -f "${ROOT}/remote-tools/cegsp_p7_a100_scaling.py" || { echo "missing P7 helper script" >&2; exit 3; }
  test -e "${MODEL_PATH}" || { echo "missing model path: ${MODEL_PATH}" >&2; exit 4; }
  "${PYTHON_BIN}" -m py_compile \
    "${ROOT}/remote-tools/cegsp_p8_downstream.py" \
    "${ROOT}/remote-tools/cegsp_p7_a100_scaling.py"
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
for name in ["torch", "transformers", "datasets"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing python package: {name}")
print("python_package_check=ok")
print("lm_eval_installed=", importlib.util.find_spec("lm_eval") is not None)
PY
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  echo "model_key=${MODEL_KEY}"
  echo "model_path=${MODEL_PATH}"
  echo "tasks=${TASKS}"
  echo "run_id=${RUN_ID}"
  echo "screen_name=${SCREEN_NAME}"
  echo "log_path=${LOG_PATH}"
}

case "${ACTION}" in
  check)
    check_common
    ;;
  launch)
    check_common
    mkdir -p "${RUN_DIR}"
    screen -dmS "${SCREEN_NAME}" bash -lc "cd '${ROOT}/remote-tools' && CUDA_VISIBLE_DEVICES=0 '${PYTHON_BIN}' cegsp_p8_downstream.py --model '${MODEL_PATH}' --run-id '${RUN_ID}' --tasks '${TASKS}' --max-examples '${MAX_EXAMPLES}' --eval-batch-size '${EVAL_BATCH_SIZE}' --out-dir '${OUT_ROOT}' 2>&1 | tee '${LOG_PATH}'"
    screen -ls
    echo "launched=${SCREEN_NAME}"
    ;;
  *)
    echo "usage: $0 [check|launch] [llama|qwen]" >&2
    exit 2
    ;;
esac
