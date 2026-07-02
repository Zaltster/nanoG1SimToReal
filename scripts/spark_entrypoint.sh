#!/usr/bin/env bash
set -uo pipefail

OUT="${NANOG1_OUTPUT_DIR:-/outputs}"
mkdir -p "${OUT}"
case "${OUT}" in
  ""|"/")
    echo "refusing to clear unsafe NANOG1_OUTPUT_DIR=${OUT}" >&2
    exit 2
    ;;
esac
find "${OUT}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
exec > >(tee -a "${OUT}/entrypoint.log") 2>&1
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/started_at.txt"
printf '{"time_unix":%s,"output_dir":"%s","event":"outputs_cleared"}\n' "$(date +%s)" "${OUT}" > "${OUT}/run_metadata.json"

otel_event() {
  local name="$1"
  local status="${2:-ok}"
  printf '{"time_unix_nano":%s,"name":"%s","status":"%s"}\n' "$(date +%s%N)" "${name}" "${status}" >> "${OUT}/otel.jsonl"
}

gpu_monitor() {
  echo "timestamp,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c" > "${OUT}/gpu_samples.csv"
  while true; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits >> "${OUT}/gpu_samples.csv" 2>> "${OUT}/gpu_samples.err" || true
    fi
    sleep "${NANOG1_GPU_SAMPLE_INTERVAL:-60}"
  done
}

supervisor_heartbeat() {
  local interval="${NANOG1_HEARTBEAT_INTERVAL:-30}"
  while true; do
    local now
    local train_mtime
    local setup_mtime
    local metric_mtime
    local progress_mtime
    now="$(date +%s)"
    train_mtime="$(stat -c %Y "${OUT}/train.log" 2>/dev/null || echo 0)"
    setup_mtime="$(stat -c %Y "${OUT}/setup.log" 2>/dev/null || echo 0)"
    metric_mtime="$(stat -c %Y "${OUT}/fall_metrics.jsonl" 2>/dev/null || echo 0)"
    progress_mtime="$(stat -c %Y "${OUT}/training_heartbeat.json" 2>/dev/null || echo 0)"
    cat > "${OUT}/supervisor_heartbeat.json.tmp" <<EOF
{"time_unix":${now},"setup_log_age_s":$((setup_mtime > 0 ? now - setup_mtime : -1)),"train_log_age_s":$((train_mtime > 0 ? now - train_mtime : -1)),"fall_metrics_age_s":$((metric_mtime > 0 ? now - metric_mtime : -1)),"training_heartbeat_age_s":$((progress_mtime > 0 ? now - progress_mtime : -1))}
EOF
    mv "${OUT}/supervisor_heartbeat.json.tmp" "${OUT}/supervisor_heartbeat.json"
    cat "${OUT}/supervisor_heartbeat.json" >> "${OUT}/supervisor_heartbeat.jsonl"
    sleep "${interval}"
  done
}

checkpoint_mirror() {
  while true; do
    python3 /app/scripts/mirror_latest_checkpoint.py \
      /app/vendor/PufferLib/checkpoints "${OUT}" \
      >> "${OUT}/checkpoint_mirror.log" 2>&1 || true
    sleep "${NANOG1_CHECKPOINT_MIRROR_INTERVAL:-120}"
  done
}

train_watchdog() {
  local stall_seconds="${NANOG1_STALL_SECONDS:-900}"
  local hard_stall_seconds="${NANOG1_HARD_STALL_SECONDS:-1800}"
  local interval="${NANOG1_WATCHDOG_INTERVAL:-30}"
  local gpu_active_threshold="${NANOG1_GPU_ACTIVE_THRESHOLD:-10}"
  local gpu_sample_interval="${NANOG1_GPU_SAMPLE_INTERVAL:-60}"
  while true; do
    sleep "${interval}"
    [ -f "${OUT}/finished_at.txt" ] && return 0
    python3 /app/scripts/mirror_latest_checkpoint.py \
      /app/vendor/PufferLib/checkpoints "${OUT}" \
      >> "${OUT}/checkpoint_mirror.log" 2>&1 || true

    set +e
    python3 /app/scripts/progress_watchdog_check.py "${OUT}" \
      --stall-seconds "${stall_seconds}" \
      --hard-stall-seconds "${hard_stall_seconds}" \
      --gpu-active-threshold "${gpu_active_threshold}" \
      --gpu-sample-interval "${gpu_sample_interval}" \
      >> "${OUT}/spark_watchdog.log" 2>&1
    watchdog_rc=$?
    set -e
    if [ "${watchdog_rc}" -eq 30 ]; then
      {
        cat "${OUT}/progress_watchdog.json" 2>/dev/null || true
        python3 /app/scripts/mirror_latest_checkpoint.py \
          /app/vendor/PufferLib/checkpoints "${OUT}" --archive || true
      } >> "${OUT}/spark_watchdog.log" 2>&1
      return 30
    fi
  done
}

otel_event entrypoint_start
gpu_monitor &
monitor_pid=$!
supervisor_heartbeat &
heartbeat_pid=$!
checkpoint_mirror &
mirror_pid=$!
trap 'kill "${monitor_pid:-}" "${heartbeat_pid:-}" "${mirror_pid:-}" "${watchdog_pid:-}" "${train_pid:-}" 2>/dev/null || true' EXIT

python3 /app/scripts/check_spark.py > "${OUT}/spark.json" 2>&1 || true
otel_event spark_probe_done

set +e
otel_event train_start
python3 /app/train_local.py \
  --stall-seconds "${NANOG1_STALL_SECONDS:-900}" \
  --checkpoint-mirror-seconds "${NANOG1_CHECKPOINT_MIRROR_INTERVAL:-120}" \
  "$@" &
train_pid=$!
train_watchdog &
watchdog_pid=$!

completed_pid=""
wait -n -p completed_pid "${train_pid}" "${watchdog_pid}"
first_rc=$?
if [ "${completed_pid}" = "${watchdog_pid}" ]; then
  train_rc=30
  otel_event watchdog_hard_stall "exit_${first_rc}"
  kill -INT "${train_pid}" 2>/dev/null || true
  sleep 60
  kill -TERM "${train_pid}" 2>/dev/null || true
  sleep 30
  kill -KILL "${train_pid}" 2>/dev/null || true
  wait "${train_pid}" 2>/dev/null || true
else
  train_rc="${first_rc}"
fi
set -e
otel_event train_end "exit_${train_rc}"

python3 /app/scripts/mirror_latest_checkpoint.py \
  /app/vendor/PufferLib/checkpoints "${OUT}" --archive \
  >> "${OUT}/checkpoint_mirror.log" 2>&1 || true

kill "${monitor_pid}" "${heartbeat_pid}" "${mirror_pid}" "${watchdog_pid}" 2>/dev/null || true
wait "${monitor_pid}" "${heartbeat_pid}" "${mirror_pid}" "${watchdog_pid}" 2>/dev/null || true

if [ -f "${OUT}/latest.bin" ]; then
  otel_event video_start
  set +e
  G1_DEMO_VIDEO_FRAMES="${NANOG1_DEMO_FRAMES:-300}" \
  G1_DEMO_VIDEO_FPS="${NANOG1_DEMO_FPS:-30}" \
    bash /app/scripts/record_demo_video.sh "${OUT}/latest.bin" "${OUT}/g1_demo_latest.mp4" \
      > "${OUT}/g1_demo_video.log" 2>&1
  video_rc=$?
  G1_DEMO_VIDEO_FRAMES="${NANOG1_DEMO_FRAMES:-300}" \
  G1_DEMO_VIDEO_FPS="${NANOG1_DEMO_FPS:-30}" \
    bash /app/scripts/record_pushed_demo_video.sh "${OUT}/latest.bin" "${OUT}/g1_demo_pushed.mp4" \
      >> "${OUT}/g1_demo_video.log" 2>&1
  pushed_video_rc=$?
  set -e
  otel_event video_end "exit_${video_rc}_pushed_${pushed_video_rc}"
  [ "${video_rc}" -eq 0 ] || cat "${OUT}/g1_demo_video.log"
  [ "${pushed_video_rc}" -eq 0 ] || cat "${OUT}/g1_demo_video.log"
else
  otel_event video_skipped "missing_latest_bin"
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/finished_at.txt"
echo "${train_rc}" > "${OUT}/train_exit_code.txt"

echo "artifact server: http://0.0.0.0:${NANOG1_ARTIFACT_PORT:-8787}/"
ls -lah "${OUT}" || true
otel_event artifact_server_start
python3 -m http.server "${NANOG1_ARTIFACT_PORT:-8787}" --directory "${OUT}"
