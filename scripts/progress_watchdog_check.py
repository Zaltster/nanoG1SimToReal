#!/usr/bin/env python3
"""Single-step liveness check for Spark training outputs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def file_mtime(path: Path) -> int:
    try:
        return int(path.stat().st_mtime)
    except FileNotFoundError:
        return 0


def read_json(path: Path) -> dict[str, object]:
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    tmp.replace(path)


def latest_gpu_util(path: Path) -> int:
    try:
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        return -1
    if len(rows) < 2 or len(rows[-1]) < 3:
        return -1
    try:
        return int(float(rows[-1][2].strip()))
    except ValueError:
        return -1


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(message + "\n")


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--hard-stall-seconds", type=int, default=1800)
    parser.add_argument("--gpu-active-threshold", type=int, default=10)
    parser.add_argument("--gpu-sample-interval", type=int, default=60)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    now = int(time.time())

    state_path = out / ".progress_watchdog_state.json"
    state = read_json(state_path)
    last_real_progress_ts = int(state.get("last_real_progress_ts", now))
    last_output_mtime = int(state.get("last_output_mtime", 0))
    last_ckpt_counter = int(state.get("last_ckpt_counter", 0))
    last_progress_source = str(state.get("last_progress_source", "watchdog_start"))
    metrics_stall_warned = bool(state.get("metrics_stall_warned", False))
    metrics_hard_stall_warned = bool(state.get("metrics_hard_stall_warned", False))

    metrics_mtime = file_mtime(out / "fall_metrics.jsonl")
    train_mtime = file_mtime(out / "train.log")
    output_mtime = max(metrics_mtime, train_mtime)

    checkpoint = read_json(out / "latest_checkpoint.json")
    ckpt_counter = int(checkpoint.get("counter", 0) or 0)
    ckpt_count = int(checkpoint.get("checkpoint_count", 0) or 0)

    gpu_util = latest_gpu_util(out / "gpu_samples.csv")
    gpu_mtime = file_mtime(out / "gpu_samples.csv")
    gpu_age = now - gpu_mtime if gpu_mtime else -1
    gpu_active = (
        gpu_util >= args.gpu_active_threshold
        and gpu_age >= 0
        and gpu_age <= args.gpu_sample_interval * 2 + 30
    )

    if output_mtime > last_output_mtime:
        last_output_mtime = output_mtime
        last_real_progress_ts = now
        last_progress_source = "metrics_or_log"
        metrics_stall_warned = False
        metrics_hard_stall_warned = False

    if ckpt_counter > last_ckpt_counter:
        last_ckpt_counter = ckpt_counter
        last_real_progress_ts = now
        last_progress_source = "checkpoint"

    progress_age = now - last_real_progress_ts
    metrics_age = now - metrics_mtime if metrics_mtime else -1
    train_log_age = now - train_mtime if train_mtime else -1
    output_age = now - output_mtime if output_mtime else -1
    action = "ok"
    reason = ""

    metrics_stale_while_checkpoints_advance = (
        output_mtime > 0
        and output_age >= args.stall_seconds
        and ckpt_counter > 0
        and last_progress_source == "checkpoint"
    )
    metrics_hard_stale_while_checkpoints_advance = (
        metrics_stale_while_checkpoints_advance
        and output_age >= args.hard_stall_seconds
    )

    if metrics_hard_stale_while_checkpoints_advance:
        reason = (
            f"metrics/log output hard-stalled for {output_age}s while checkpoints advanced "
            f"to {ckpt_counter}; keeping trainer alive but marking logging visibility stale"
        )
        if not metrics_hard_stall_warned:
            append_log(out / "spark_watchdog.log", f"{iso_now()} {reason}")
        metrics_stall_warned = True
        metrics_hard_stall_warned = True
        action = "metrics_hard_stale"
    elif metrics_stale_while_checkpoints_advance:
        reason = (
            f"metrics/log output stalled for {output_age}s, "
            f"but checkpoints are still advancing at counter {ckpt_counter}; keeping trainer alive"
        )
        if not metrics_stall_warned:
            append_log(out / "spark_watchdog.log", f"{iso_now()} {reason}")
        metrics_stall_warned = True
        action = "metrics_stale"
    elif progress_age >= args.stall_seconds:
        if gpu_active and progress_age < args.hard_stall_seconds:
            reason = (
                f"observable training progress stale for {progress_age}s, "
                f"but GPU is active ({gpu_util}%); waiting until hard stall {args.hard_stall_seconds}s"
            )
            append_log(out / "spark_watchdog.log", f"{iso_now()} {reason}")
            action = "gpu_grace"
        else:
            reason = f"training progress stalled for {progress_age}s after {last_progress_source}; interrupting trainer"
            append_log(out / "spark_watchdog.log", f"{iso_now()} {reason}")
            action = "stop"

    payload = {
        "time_unix": now,
        "action": action,
        "reason": reason,
        "last_progress_source": last_progress_source,
        "progress_age_s": progress_age,
        "last_output_mtime_unix": last_output_mtime,
        "checkpoint_counter": ckpt_counter,
        "checkpoint_count": ckpt_count,
        "metrics_age_s": metrics_age,
        "train_log_age_s": train_log_age,
        "output_age_s": output_age,
        "gpu_util_pct": gpu_util,
        "gpu_sample_age_s": gpu_age,
        "gpu_active": gpu_active,
        "stall_seconds": args.stall_seconds,
        "hard_stall_seconds": args.hard_stall_seconds,
    }
    write_json_atomic(out / "progress_watchdog.json", payload)
    with (out / "progress_watchdog.jsonl").open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")

    write_json_atomic(
        state_path,
        {
            "last_real_progress_ts": last_real_progress_ts,
            "last_output_mtime": last_output_mtime,
            "last_ckpt_counter": last_ckpt_counter,
            "last_progress_source": last_progress_source,
            "metrics_stall_warned": metrics_stall_warned,
            "metrics_hard_stall_warned": metrics_hard_stall_warned,
        },
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return {"ok": 0, "metrics_stale": 10, "metrics_hard_stale": 10, "gpu_grace": 20, "stop": 30}[action]


if __name__ == "__main__":
    raise SystemExit(main())
