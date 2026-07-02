#!/usr/bin/env python3
"""Print status for a visual-command training run."""
from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts" / "visual-command" / "1091043328-train"


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def read_last_jsonl(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def fmt(value: object, missing: str = "-") -> str:
    return missing if value is None or value == "" else str(value)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()

    run = args.run if args.run.is_absolute() else ROOT / args.run
    heartbeat = read_json(run / "visual_training_heartbeat.json")
    latest = read_json(run / "latest_checkpoint.json")
    metrics = read_last_jsonl(run / "metrics.jsonl")
    summary = read_json(run / "summary.json")

    now = time.time()
    heartbeat_age = now - float(heartbeat.get("time_unix", 0) or 0) if heartbeat else None
    print(f"run: {run}")
    print(f"heartbeat: event={fmt(heartbeat.get('event'))} step={fmt(heartbeat.get('step'))} age_s={heartbeat_age:.1f}" if heartbeat_age is not None else "heartbeat: missing")
    print(f"latest_checkpoint: step={fmt(latest.get('step'))} reason={fmt(latest.get('reason'))} score={fmt(latest.get('score'))}")
    print(f"latest_policy: {fmt(latest.get('latest_policy'))}")
    print(f"best_policy: {fmt(latest.get('best_policy') or summary.get('best_policy'))}")
    print(f"metrics: step={fmt(metrics.get('step'))} stop_accuracy={fmt(metrics.get('stop_accuracy'))} velocity_mae={fmt(metrics.get('velocity_mae'))} loss={fmt(metrics.get('loss'))}")
    if summary:
        print(f"summary: final_step={fmt(summary.get('final_step'))} stop_reason={fmt(summary.get('stop_reason'))}")


if __name__ == "__main__":
    main()
