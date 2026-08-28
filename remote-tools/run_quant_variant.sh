#!/usr/bin/env bash
set -uo pipefail

run_id="${1:?run id required}"
run_dir="${2:?run directory required}"
method="${3:?method required}"
ssr_mode="${4:?ssr mode required}"
project_dir=/root/PT2-LLM

case "$method" in
  atq|atq-itf|atq-aga|ternary-init) ;;
  *) printf 'Unsupported method: %s\n' "$method" >&2; exit 64 ;;
esac
case "$ssr_mode" in
  on) ssr_arg=--ssr ;;
  off) ssr_arg= ;;
  *) printf 'Unsupported SSR mode: %s\n' "$ssr_mode" >&2; exit 64 ;;
esac

command_text="./venv/bin/python quantize.py /root/PT2-LLM/qwen3-8b wikitext2 $method --nsamples 8 --calib_seqlen 512 --ppl_seqlen 2048 --blocksize 128 --device cuda:0 ${ssr_arg}"
mkdir -p "$run_dir"
start_iso="$(date -Is)"
start_epoch="$(date +%s)"
printf 'run_id\t%s\nstatus\tRUNNING\nstart\t%s\ncommand\t%s\n' \
  "$run_id" "$start_iso" "$command_text" > "$run_dir/status.tsv"

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 > "$run_dir/gpu.csv" 2>&1 &
monitor_pid=$!

cd "$project_dir"
set +e
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PT2_DATA_ROOT=/root/PT2-LLM/data \
  ./venv/bin/python quantize.py /root/PT2-LLM/qwen3-8b wikitext2 "$method" \
    --nsamples 8 --calib_seqlen 512 --ppl_seqlen 2048 --blocksize 128 --device cuda:0 $ssr_arg \
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
printf 'run_id\t%s\nstatus\t%s\nstart\t%s\nend\t%s\nelapsed_seconds\t%s\nexit_code\t%s\npeak_gpu_mib\t%s\ncommand\t%s\n' \
  "$run_id" "$status" "$start_iso" "$end_iso" "$elapsed" "$run_rc" "$peak_mib" "$command_text" > "$run_dir/status.tsv"
exit "$run_rc"
