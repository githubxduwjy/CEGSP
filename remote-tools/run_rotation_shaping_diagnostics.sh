#!/usr/bin/env bash
set -uo pipefail

run_dir="${1:?run directory required}"
script_path="${2:-/root/PT2-LLM-official/rotation_shaping_diagnostics.py}"
python_bin=/root/PT2-LLM/venv/bin/python
mkdir -p "$run_dir"
start_iso="$(date -Is)"
start_epoch="$(date +%s)"
printf 'status\tRUNNING\nstart\t%s\n' "$start_iso" > "$run_dir/status.tsv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 > "$run_dir/gpu.csv" 2>&1 &
monitor_pid=$!
cd /root/PT2-LLM-official
set +e
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PT2_DATA_ROOT=/root/PT2-LLM/data \
  "$python_bin" "$script_path" \
    --model /root/models/Llama-2-7b-hf \
    --output-dir "$run_dir" \
    --layers 0,10,20,31 --nsamples 8 --tokens-per-sample 128 \
    --blocks-per-module 2 --blocksize 128 --seed 0 \
    > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
run_rc=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
end_iso="$(date -Is)"
end_epoch="$(date +%s)"
elapsed="$((end_epoch - start_epoch))"
status=FAILED
if [ "$run_rc" -eq 0 ]; then status=COMPLETED; fi
peak_mib="$(awk -F, 'BEGIN{m=0} {gsub(/ /,"",$3); if (($3+0)>m) m=$3+0} END{print m}' "$run_dir/gpu.csv")"
printf 'status\t%s\nstart\t%s\nend\t%s\nelapsed_seconds\t%s\nexit_code\t%s\npeak_gpu_mib\t%s\n' \
  "$status" "$start_iso" "$end_iso" "$elapsed" "$run_rc" "$peak_mib" > "$run_dir/status.tsv"
exit "$run_rc"
