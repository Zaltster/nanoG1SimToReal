#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-artifacts/visual-command/1091043328-train}"
STEPS="${VISUAL_COMMAND_STEPS:-20000}"
BATCH_SIZE="${VISUAL_COMMAND_BATCH_SIZE:-2048}"
LR="${VISUAL_COMMAND_LR:-0.01}"
RESUME="${VISUAL_COMMAND_RESUME:-0}"
RESUME_FROM="${VISUAL_COMMAND_RESUME_FROM:-}"
TARGET_STOP_ACCURACY="${VISUAL_COMMAND_TARGET_STOP_ACCURACY:-0.985}"
MAX_SECONDS="${VISUAL_COMMAND_MAX_SECONDS:-0}"
LOG_INTERVAL="${VISUAL_COMMAND_LOG_INTERVAL:-250}"
CHECKPOINT_INTERVAL="${VISUAL_COMMAND_CHECKPOINT_INTERVAL:-1000}"
HEARTBEAT_INTERVAL="${VISUAL_COMMAND_HEARTBEAT_INTERVAL:-100}"
TARGET_MODE="${VISUAL_COMMAND_TARGET_MODE:-moving}"
TRAJECTORY_LEN="${VISUAL_COMMAND_TRAJECTORY_LEN:-16}"
MOVING_FRACTION="${VISUAL_COMMAND_MOVING_FRACTION:-0.75}"

cmd=(.venv/bin/python scripts/train_visual_command_policy.py \
  --base-walker artifacts/protected/2026-06-21/1091043328/checkpoint.bin \
  --encoder features \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size 8192 \
  --hidden-dim 64 \
  --learning-rate "$LR" \
  --target-mode "$TARGET_MODE" \
  --trajectory-len "$TRAJECTORY_LEN" \
  --moving-fraction "$MOVING_FRACTION" \
  --log-interval "$LOG_INTERVAL" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --heartbeat-interval "$HEARTBEAT_INTERVAL" \
  --target-stop-accuracy "$TARGET_STOP_ACCURACY" \
  --max-seconds "$MAX_SECONDS" \
  --out "$OUT")

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi

if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--resume-from "$RESUME_FROM")
fi

"${cmd[@]}"
