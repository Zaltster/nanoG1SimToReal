#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ckpt="${1:-assets/nanoG1.bin}"
out="${2:-outputs/g1_demo_pushed.mp4}"

G1_DEMO_PUSH_VIDEO=1 \
G1_DEMO_PUSH_FIRST="${G1_DEMO_PUSH_FIRST:-45}" \
G1_DEMO_PUSH_EVERY="${G1_DEMO_PUSH_EVERY:-90}" \
  bash scripts/record_demo_video.sh "${ckpt}" "${out}"
