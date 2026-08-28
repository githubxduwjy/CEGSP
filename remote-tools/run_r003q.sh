#!/usr/bin/env bash
set -uo pipefail

run_dir="${1:?run directory required}"
project_dir=/root/PT2-LLM
mkdir -p "$run_dir"

start_iso="$(date -Is)"
start_epoch="$(date +%s)"
printf 'run_id\tR003Q\nstatus\tRUNNING\nstart\t%s\ncommand\t./venv/bin/python hf_ppl.py\n' "$start_iso" > "$run_dir/status.tsv"

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 > "$run_dir/gpu.csv" 2>&1 &
monitor_pid=$!

cd "$project_dir"
set +e
./venv/bin/python hf_ppl.py > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
run_rc=$?
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

end_iso="$(date -Is)"
end_epoch="$(date +%s)"
elapsed="$((end_epoch - start_epoch))"
status=FAILED
if [ "$run_rc" -eq 0 ]; then
  status=COMPLETED
fi

peak_mib="$(awk -F, 'BEGIN{m=0} {gsub(/ /,"",$3); if (($3+0)>m) m=$3+0} END{print m}' "$run_dir/gpu.csv")"
printf 'run_id\tR003Q\nstatus\t%s\nstart\t%s\nend\t%s\nelapsed_seconds\t%s\nexit_code\t%s\npeak_gpu_mib\t%s\ncommand\t./venv/bin/python hf_ppl.py\n' \
  "$status" "$start_iso" "$end_iso" "$elapsed" "$run_rc" "$peak_mib" > "$run_dir/status.tsv"

exit "$run_rc"
