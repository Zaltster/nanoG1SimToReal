#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec python3 live_g1_follow_dry_run.py "$@"
