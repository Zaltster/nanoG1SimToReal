#!/bin/bash
# nanoG1 speedrun — train a Unitree G1 to walk in <60s, on one GPU, from scratch.
#
#   bash speedrun.sh
#
# Prereqs: uv (https://docs.astral.sh/uv), a Modal account (`modal token new`), git.
# The GPU run is on Modal (~$0.17 on an RTX PRO 6000). Everything else is local.
set -e
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv"; exit 1; }

echo "[1/4] python env (uv sync)…"
uv sync

echo "[2/4] engine fork (pinned G1-specialized PufferLib)…"
bash setup.sh

echo "[3/4] train on GPU via Modal — the <60s walk (writes assets/nanoG1.bin)…"
uv run modal run train.py

echo "[4/4] quality gate — does it actually walk?…"
uv run python eval.py assets/nanoG1.bin

cat <<'EOF'

✓ speedrun complete.
  See it move:   bash web/build_demo.sh && ./build/g1demo assets/nanoG1.bin
  Live demo:     https://g1-sub60-walk.vercel.app
  The policy:    https://huggingface.co/kingJulio/nanoG1
EOF
