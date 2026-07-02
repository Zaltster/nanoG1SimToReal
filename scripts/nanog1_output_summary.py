#!/usr/bin/env python3
"""Summarize nanoG1 run liveness and push-eval status from an outputs directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def fmt(value: object, default: str = "unknown") -> str:
    return default if value is None or value == "" else str(value)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
    watchdog = read_json(out / "progress_watchdog.json")
    latest = read_json(out / "latest_checkpoint.json")
    heartbeat = read_json(out / "training_heartbeat.json")
    eval_latest = read_json(out / "checkpoint_eval_latest.json")
    push_commands = eval_latest.get("push_commands")
    if not isinstance(push_commands, dict):
        push_commands = {}

    print(f"outputs: {out}")
    print(f"started: {fmt(read_text(out / 'started_at.txt'))}")
    print(f"finished: {fmt(read_text(out / 'finished_at.txt'), 'not finished')}")
    print(f"train_exit_code: {fmt(read_text(out / 'train_exit_code.txt'), 'not written')}")
    print(
        "watchdog: "
        f"action={fmt(watchdog.get('action'))} "
        f"checkpoint={fmt(watchdog.get('checkpoint_counter'))} "
        f"metrics_age_s={fmt(watchdog.get('metrics_age_s'))} "
        f"train_log_age_s={fmt(watchdog.get('train_log_age_s'))} "
        f"gpu_active={fmt(watchdog.get('gpu_active'))}"
    )
    reason = watchdog.get("reason")
    if reason:
        print(f"watchdog_reason: {reason}")
    print(
        "latest_checkpoint: "
        f"counter={fmt(latest.get('counter'))} "
        f"count={fmt(latest.get('checkpoint_count'))}"
    )
    print(
        "training_heartbeat: "
        f"event={fmt(heartbeat.get('event'))} "
        f"steps={fmt(heartbeat.get('steps'))} "
        f"perf={fmt(heartbeat.get('perf'))} "
        f"falls={fmt(heartbeat.get('falls'))}"
    )
    print(
        "push_eval: "
        f"counter={fmt(eval_latest.get('counter'))} "
        f"ok={fmt(eval_latest.get('ok'))} "
        f"pushes={fmt(eval_latest.get('pushes'))} "
        f"push_falls={fmt(eval_latest.get('push_falls'))} "
        f"push_perf={fmt(eval_latest.get('push_perf'))}"
    )
    for name in sorted(push_commands):
        command = push_commands.get(name)
        if not isinstance(command, dict):
            continue
        print(
            f"push_eval.{name}: "
            f"pushes={fmt(command.get('pushes'))} "
            f"falls={fmt(command.get('falls'))} "
            f"perf={fmt(command.get('perf'))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
