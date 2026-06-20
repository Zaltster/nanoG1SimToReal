#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ckpt="${1:-assets/nanoG1.bin}"
out="${2:-outputs/g1_demo.mp4}"
frames="${G1_DEMO_VIDEO_FRAMES:-180}"
fps="${G1_DEMO_VIDEO_FPS:-30}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
[ -f "${ckpt}" ] || { echo "checkpoint not found: ${ckpt}"; exit 1; }

mkdir -p "$(dirname "${out}")"
bash web/build_demo.sh

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

rm -f frame_*.png
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null; then
  G1_DEMO_RECORD_DIR="${tmp}" G1_DEMO_RECORD_FRAMES="${frames}" xvfb-run -a ./build/g1demo "${ckpt}"
else
  G1_DEMO_RECORD_DIR="${tmp}" G1_DEMO_RECORD_FRAMES="${frames}" ./build/g1demo "${ckpt}"
fi
if [ ! -f "${tmp}/frame_0000.png" ] && [ -f "frame_0000.png" ]; then
  mv frame_*.png "${tmp}/"
fi
ffmpeg -y -framerate "${fps}" -i "${tmp}/frame_%04d.png" \
  -vf "format=yuv420p" -movflags +faststart "${out}"

echo "wrote ${out}"
