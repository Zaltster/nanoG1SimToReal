#!/usr/bin/env python3
"""Regression tests for Spark training liveness decisions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "progress_watchdog_check.py"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n")


class ProgressWatchdogCheckTest(unittest.TestCase):
    def run_check(self, out: Path, *, stall: int = 1, hard: int = 10) -> tuple[int, dict[str, object]]:
        proc = subprocess.run(
            [
                sys.executable,
                str(CHECK),
                str(out),
                "--stall-seconds",
                str(stall),
                "--hard-stall-seconds",
                str(hard),
                "--gpu-sample-interval",
                "60",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.stderr, "")
        payload = json.loads(proc.stdout)
        return proc.returncode, payload

    def test_checkpoint_progress_warns_when_metrics_stale(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            metrics = out / "fall_metrics.jsonl"
            metrics.write_text("{}\n")
            old = time.time() - 5
            os.utime(metrics, (old, old))
            write_json(out / "latest_checkpoint.json", {"counter": 12, "checkpoint_count": 2})

            rc, payload = self.run_check(out, hard=30)

            self.assertEqual(rc, 10)
            self.assertEqual(payload["action"], "metrics_stale")
            self.assertEqual(payload["last_progress_source"], "checkpoint")
            self.assertEqual(payload["checkpoint_counter"], 12)
            self.assertIn("keeping trainer alive", (out / "spark_watchdog.log").read_text())

    def test_checkpoint_progress_keeps_running_when_metrics_hard_stale(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            metrics = out / "fall_metrics.jsonl"
            metrics.write_text("{}\n")
            old = time.time() - 30
            os.utime(metrics, (old, old))
            write_json(out / "latest_checkpoint.json", {"counter": 12, "checkpoint_count": 2})

            rc, payload = self.run_check(out, hard=10)

            self.assertEqual(rc, 10)
            self.assertEqual(payload["action"], "metrics_hard_stale")
            self.assertIn("keeping trainer alive", payload["reason"])

    def test_no_progress_stops_when_gpu_is_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            now = int(time.time())
            write_json(
                out / ".progress_watchdog_state.json",
                {
                    "last_real_progress_ts": now - 10,
                    "last_output_mtime": now - 20,
                    "last_ckpt_counter": 7,
                    "last_progress_source": "checkpoint",
                    "metrics_stall_warned": True,
                },
            )
            write_json(out / "latest_checkpoint.json", {"counter": 7, "checkpoint_count": 2})

            rc, payload = self.run_check(out)

            self.assertEqual(rc, 30)
            self.assertEqual(payload["action"], "stop")
            self.assertIn("interrupting trainer", payload["reason"])

    def test_gpu_activity_defers_stop_until_hard_stall(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            now = int(time.time())
            write_json(
                out / ".progress_watchdog_state.json",
                {
                    "last_real_progress_ts": now - 10,
                    "last_output_mtime": now - 20,
                    "last_ckpt_counter": 7,
                    "last_progress_source": "checkpoint",
                    "metrics_stall_warned": True,
                },
            )
            write_json(out / "latest_checkpoint.json", {"counter": 7, "checkpoint_count": 2})
            (out / "gpu_samples.csv").write_text(
                "timestamp,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c\n"
                "2026/06/18 19:31:03.001,NVIDIA GB10,96,1234,65536,210,74\n"
            )

            rc, payload = self.run_check(out, hard=60)

            self.assertEqual(rc, 20)
            self.assertEqual(payload["action"], "gpu_grace")
            self.assertTrue(payload["gpu_active"])


if __name__ == "__main__":
    unittest.main()
