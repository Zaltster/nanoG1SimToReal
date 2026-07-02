"""Train nanoG1 on a local CUDA machine such as a DGX Spark.

It prepares the pinned PufferLib engine checkout, extracts the frozen G1 model,
builds the specialized ``g1gpu`` target, runs training, and copies the
samples-to-walk checkpoint to ``assets/nanoG1.bin``.

Examples:
    python train_local.py --smoke
    NANOG1_NVCC_ARCH=sm_120 python train_local.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import configparser
import json
import os
import queue
import re
import signal
import shutil
import site
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import recipe as R


ROOT = Path(__file__).resolve().parent
PUFFER = ROOT / "vendor" / "PufferLib"
MODEL_DIR = ROOT / "envs" / "g1" / "model"
MODEL = MODEL_DIR / "g1.mjb"
SMOKE_TIMESTEPS = 10_000_000
OUTPUT_DIR = Path(os.environ["NANOG1_OUTPUT_DIR"]) if os.environ.get("NANOG1_OUTPUT_DIR") else None


def append_setup_log(text: str) -> None:
    if OUTPUT_DIR is None:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "setup.log").open("a") as f:
        f.write(text)


def emit_heartbeat(event: dict[str, object]) -> None:
    try:
        write_training_heartbeat(event)
    except Exception as exc:
        append_setup_log(f"heartbeat write failed: {exc}\n")


def discover_executable(name: str, patterns: list[str]) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for pattern in patterns:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            if path.is_file() and os.access(path, os.X_OK):
                os.environ["PATH"] = f"{path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
                return str(path)
    return None


def prepend_path_env(key: str, path: Path) -> None:
    current = os.environ.get(key, "")
    parts = [str(path)] + ([current] if current else [])
    os.environ[key] = os.pathsep.join(parts)


def ensure_cudnn_link_path() -> Path | None:
    roots = [Path(p) for p in site.getsitepackages()]
    roots.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "dist-packages")
    for root in roots:
        lib = root / "nvidia" / "cudnn" / "lib"
        if not lib.is_dir():
            continue
        candidates = sorted(p for p in lib.glob("libcudnn.so*") if p.is_file())
        if not candidates:
            continue
        link = lib / "libcudnn.so"
        if not link.exists():
            target = next((p for p in candidates if p.name != "libcudnn.so"), candidates[0])
            try:
                link.symlink_to(target.name)
            except OSError:
                pass
        prepend_path_env("LIBRARY_PATH", lib)
        prepend_path_env("LD_LIBRARY_PATH", lib)
        return lib
    return None


def run(cmd: list[str] | str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    shell = isinstance(cmd, str)
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"$ {printable}", flush=True)
    append_setup_log(f"$ {printable}\n")
    emit_heartbeat(
        {
            "event": "setup_command_start",
            "command": printable,
            "cwd": str(cwd) if cwd else str(Path.cwd()),
        }
    )
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        shell=shell,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        append_setup_log(line)
    rc = proc.wait()
    emit_heartbeat(
        {
            "event": "setup_command_end",
            "command": printable,
            "exit_code": rc,
        }
    )
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return subprocess.CompletedProcess(cmd, rc)


def ensure_cuda() -> None:
    if shutil.which("nvidia-smi") is None:
        sys.exit("nvidia-smi not found. Run this on the DGX Spark or another NVIDIA CUDA host.")
    run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if discover_executable("nvcc", ["/usr/local/cuda*/bin/nvcc", "/usr/local/cuda*/targets/*/bin/nvcc", "/usr/bin/nvcc"]) is None:
        sys.exit("nvcc not found. Install CUDA devel tooling or use Dockerfile.spark.")
    cudnn = ensure_cudnn_link_path()
    if cudnn:
        print(f"using cuDNN libraries from {cudnn}", flush=True)


def default_nvcc_arch() -> str:
    override = os.environ.get("NANOG1_NVCC_ARCH")
    if override:
        return override
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()[0].strip()
        major, minor = out.split(".", 1)
        return f"sm_{major}{minor}"
    except Exception:
        return "native"


def ensure_engine(skip_setup: bool) -> None:
    if not skip_setup:
        run(["bash", "setup.sh"], cwd=ROOT)
    if not (PUFFER / "build.sh").exists():
        sys.exit(f"{PUFFER} is missing or incomplete. Run: bash setup.sh")
    run([sys.executable, "-m", "pip", "install", "-e", str(PUFFER), "--no-deps"])


def ensure_model(force: bool) -> None:
    if force or not MODEL.exists():
        env = {**os.environ, "G1_MODEL_DIR": str(MODEL_DIR)}
        run([sys.executable, "tools/extract_g1_model.py"], cwd=ROOT, env=env)
    if not MODEL.exists():
        sys.exit(f"model extraction failed: {MODEL} not found")


def build_engine(nvcc_arch: str, skip_build: bool) -> None:
    if skip_build:
        return
    env = {
        **os.environ,
        "NVCC_ARCH": nvcc_arch,
        "G1_TASK_FLAGS": R.TASK_FLAGS,
        "PUFFER_TRAIN_FLAGS": R.TRAIN_FLAGS,
    }
    print(f"building g1gpu with NVCC_ARCH={nvcc_arch}", flush=True)
    run(["./build.sh", "g1gpu"], cwd=PUFFER, env=env)


def apply_overrides(ini: Path, overrides: str, total_timesteps: int) -> None:
    cp = configparser.ConfigParser()
    cp.read(ini)
    if total_timesteps > 0:
        cp["train"]["total_timesteps"] = str(total_timesteps)
    for pair in (p.strip() for p in overrides.split(",") if p.strip()):
        key, value = pair.split("=", 1)
        section, option = key.rsplit(".", 1)
        cp[section][option] = value
    with ini.open("w") as f:
        cp.write(f)


def checkpoint_points(ckpt_dir: Path) -> list[tuple[int, float, Path]]:
    indexed: list[tuple[int, float, Path]] = []
    for path in ckpt_dir.rglob("*.bin"):
        match = re.match(r"^(\d{16})\.bin$", path.name)
        if match:
            stat = path.stat()
            indexed.append((int(match.group(1)), stat.st_mtime, path))
    indexed.sort()
    return indexed


def steady_sps(ckpt_dir: Path) -> tuple[float | None, list[tuple[int, float]]]:
    pts: list[tuple[int, float]] = []
    for counter, mtime, _ in checkpoint_points(ckpt_dir):
        pts.append((counter, mtime))
    pts.sort()
    rates = sorted(
        (b[0] - a[0]) / (b[1] - a[1])
        for a, b in zip(pts, pts[1:])
        if b[1] > a[1] and b[0] > a[0]
    )
    return (rates[len(rates) // 2] if rates else None), pts


def perf_curve(log_env_dir: Path) -> tuple[list[int], list[float]]:
    if not log_env_dir.is_dir():
        return [], []
    for path in sorted(log_env_dir.glob("*.json")):
        try:
            metrics = json.loads(path.read_text()).get("metrics", {})
        except Exception:
            continue
        return (
            [int(x) for x in metrics.get("agent_steps", [])],
            [float(x) for x in metrics.get("env/perf", [])],
        )
    return [], []


def archive_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    with tarfile.open(dst, "w:gz") as tf:
        tf.add(src, arcname=src.name)
    print(f"archived {src} to {dst}", flush=True)


def current_run_started_unix(out_dir: Path | None) -> object:
    if out_dir is None:
        return None
    try:
        data = json.loads((out_dir / "run_metadata.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data.get("time_unix") if isinstance(data, dict) else None


def remove_stale_checkpoint_eval(out_dir: Path, current_counter: int) -> None:
    latest_eval = out_dir / "checkpoint_eval_latest.json"
    try:
        payload = json.loads(latest_eval.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if not isinstance(payload, dict):
        return

    run_started = current_run_started_unix(out_dir)
    eval_run_started = payload.get("run_started_unix")
    eval_counter = int(payload.get("counter", 0) or 0)
    stale_reasons = []
    if run_started is not None and eval_run_started != run_started:
        stale_reasons.append("run_started_unix_mismatch")
    if eval_counter > current_counter:
        stale_reasons.append("eval_counter_ahead_of_checkpoint")
    if not stale_reasons:
        return

    removed = []
    for path in [latest_eval, out_dir / "checkpoint_eval.jsonl", *out_dir.glob(".checkpoint_eval_*.bin")]:
        try:
            path.unlink()
            removed.append(path.name)
        except FileNotFoundError:
            pass
    print(
        "STALE_CHECKPOINT_EVAL_REMOVED "
        f"reason={','.join(stale_reasons)} "
        f"eval_counter={eval_counter} "
        f"current_checkpoint_counter={current_counter} "
        f"removed={','.join(removed)}",
        flush=True,
    )


def mirror_latest_checkpoint(ckpt_dir: Path, out_dir: Path | None, *, reason: str) -> dict[str, object] | None:
    if out_dir is None:
        return None
    pts = checkpoint_points(ckpt_dir)
    if not pts:
        return None
    counter, mtime, src = pts[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "latest.bin"
    tmp = out_dir / f".latest.bin.{os.getpid()}.tmp"
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    info = {
        "counter": counter,
        "source": str(src),
        "mtime_unix": round(mtime, 3),
        "mirrored_at_unix": round(time.time(), 3),
        "reason": reason,
    }
    info_tmp = out_dir / f".latest_checkpoint.json.{os.getpid()}.tmp"
    info_tmp.write_text(json.dumps(info, indent=2) + "\n")
    info_tmp.replace(out_dir / "latest_checkpoint.json")
    remove_stale_checkpoint_eval(out_dir, counter)
    return info


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
METRIC_RE = re.compile(r"\b(perf|score|episode_return|episode_length|vel_err|falls)\s+(-?[0-9.]+)")
STEPS_RE = re.compile(r"\bSteps\s+([0-9.]+)\s*([KMGB]?)")
CONV_CMD_RE = re.compile(r"CONV_EVAL cmd=(\w+)\s+falls=(\d+)\s+perf=([0-9.]+)\s+lin_err=([0-9.]+)\s+ang_err=([0-9.]+)")
PUSH_CMD_RE = re.compile(r"PUSH_EVAL cmd=(\w+)\s+falls=(\d+)\s+pushes=(\d+)\s+perf=([0-9.]+)")
RESULT_RE = re.compile(r"RESULT\s+(\w+)\s+falls=(\d+)(?:\s+pushes=(\d+))?\s+perf=([0-9.]+)")
DIAG_WALK_RE = re.compile(r"DIAG walk .* falls=(\d+) .* pelvis_z=([0-9.]+)")
DIAG_GAIT_RE = re.compile(r"action_jerk_rms=([0-9.]+)\s+leg_qvel_rms=([0-9.]+)")
DIAG_FOOT_SEP_RE = re.compile(r"DIAG FOOT_SEP: lateral_mean=([-0-9.]+)m lateral_min=([-0-9.]+)m under_0\.10m_pct=([0-9.]+)%")
EOF_SENTINEL = object()
EVAL_DEMO_READY = False
EVAL_BUILD_TIMEOUT_SECONDS = int(os.environ.get("NANOG1_EVAL_BUILD_TIMEOUT_SECONDS", "600"))
EVAL_MODE_TIMEOUT_SECONDS = int(os.environ.get("NANOG1_EVAL_MODE_TIMEOUT_SECONDS", "120"))
DEPLOY_MAX_FALLS = int(os.environ.get("NANOG1_DEPLOY_MAX_FALLS", "0"))
DEPLOY_MIN_FOOT_SEP_M = float(os.environ.get("NANOG1_DEPLOY_MIN_FOOT_SEP_M", "0.10"))


def parse_count(value: str, suffix: str) -> float:
    mult = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "B": 1e9}[suffix]
    return float(value) * mult


def append_fall_metric(metrics_path: Path | None, event: dict[str, object]) -> None:
    if metrics_path is None:
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def write_training_heartbeat(event: dict[str, object]) -> None:
    if OUTPUT_DIR is None:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"time_unix": round(time.time(), 3), **event}
    tmp = OUTPUT_DIR / f".training_heartbeat.json.{os.getpid()}.{time.time_ns()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(OUTPUT_DIR / "training_heartbeat.json")
    with (OUTPUT_DIR / "training_heartbeat.jsonl").open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def ensure_eval_demo() -> None:
    global EVAL_DEMO_READY
    if EVAL_DEMO_READY and (ROOT / "build" / "g1demo").exists():
        return
    proc = subprocess.run(
        ["bash", "web/build_demo.sh"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=EVAL_BUILD_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])
    EVAL_DEMO_READY = True


def run_demo_mode(ckpt: Path, mode_env: str) -> str:
    env = {**os.environ, mode_env: "1"}
    proc = subprocess.run(
        [str(ROOT / "build" / "g1demo"), str(ckpt)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=EVAL_MODE_TIMEOUT_SECONDS,
    )
    return proc.stdout


def run_checkpoint_eval_snapshot(snapshot: Path, *, counter: int | None, reason: str) -> dict[str, object] | None:
    try:
        return run_checkpoint_eval(snapshot, OUTPUT_DIR, counter=counter, reason=reason)
    finally:
        try:
            snapshot.unlink()
        except FileNotFoundError:
            pass


def parse_checkpoint_eval(conv: str, push: str, diag: str) -> dict[str, object]:
    payload: dict[str, object] = {"commands": {}, "push_commands": {}}
    commands: dict[str, object] = {}
    push_commands: dict[str, object] = {}
    for line in conv.splitlines():
        if match := CONV_CMD_RE.search(line):
            commands[match.group(1)] = {
                "falls": int(match.group(2)),
                "perf": float(match.group(3)),
                "lin_err": float(match.group(4)),
                "ang_err": float(match.group(5)),
            }
        elif match := RESULT_RE.search(line):
            if match.group(1) == "conv":
                payload["battery_falls"] = int(match.group(2))
                payload["battery_perf"] = float(match.group(4))
    for line in push.splitlines():
        if match := PUSH_CMD_RE.search(line):
            push_commands[match.group(1)] = {
                "falls": int(match.group(2)),
                "pushes": int(match.group(3)),
                "perf": float(match.group(4)),
            }
        elif match := RESULT_RE.search(line):
            if match.group(1) == "push":
                payload["push_falls"] = int(match.group(2))
                payload["pushes"] = int(match.group(3) or 0)
                payload["push_perf"] = float(match.group(4))
    for line in diag.splitlines():
        if match := DIAG_WALK_RE.search(line):
            payload["diag_falls"] = int(match.group(1))
            payload["pelvis_z"] = float(match.group(2))
        elif match := DIAG_GAIT_RE.search(line):
            payload["action_jerk_rms"] = float(match.group(1))
            payload["leg_qvel_rms"] = float(match.group(2))
        elif match := DIAG_FOOT_SEP_RE.search(line):
            payload["foot_sep_mean_m"] = float(match.group(1))
            payload["foot_sep_min_m"] = float(match.group(2))
            payload["foot_sep_under_0p10_pct"] = float(match.group(3))
    payload["commands"] = commands
    payload["push_commands"] = push_commands
    if "forward" in commands:
        payload["forward_perf"] = commands["forward"]["perf"]
        payload["forward_falls"] = commands["forward"]["falls"]
    if "stand" in commands:
        payload["stand_perf"] = commands["stand"]["perf"]
    if "forward" in push_commands:
        payload["push_forward_perf"] = push_commands["forward"]["perf"]
        payload["push_forward_falls"] = push_commands["forward"]["falls"]
    return payload


def checkpoint_eval_deployable(event: dict[str, object] | None) -> tuple[bool, str]:
    if not event:
        return False, "missing_eval"
    if not event.get("ok"):
        return False, str(event.get("error") or "eval_failed")[-200:]
    for key in ("battery_falls", "forward_falls", "diag_falls", "push_falls"):
        if int(event.get(key, 0) or 0) > DEPLOY_MAX_FALLS:
            return False, f"{key}={event.get(key)}"
    foot_sep = event.get("foot_sep_min_m")
    if foot_sep is None:
        return False, "missing_foot_sep"
    if float(foot_sep) < DEPLOY_MIN_FOOT_SEP_M:
        return False, f"foot_sep_min_m={foot_sep}"
    if float(event.get("foot_sep_under_0p10_pct", 0.0) or 0.0) > 0.0:
        return False, f"foot_sep_under_0p10_pct={event.get('foot_sep_under_0p10_pct')}"
    return True, ""


def run_checkpoint_eval(ckpt: Path, out_dir: Path | None, *, counter: int | None, reason: str) -> dict[str, object] | None:
    if out_dir is None or not ckpt.exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    event: dict[str, object] = {
        "time_unix": round(started, 3),
        "run_started_unix": current_run_started_unix(out_dir),
        "checkpoint": str(ckpt),
        "counter": counter,
        "reason": reason,
    }
    try:
        ensure_eval_demo()
        conv = run_demo_mode(ckpt, "G1_DEMO_EVAL")
        push = run_demo_mode(ckpt, "G1_DEMO_PUSH_EVAL")
        diag = run_demo_mode(ckpt, "G1_DEMO_DIAG")
        event.update(parse_checkpoint_eval(conv, push, diag))
        event["raw_tail"] = {
            "conv": conv.splitlines()[-8:],
            "push": push.splitlines()[-6:],
            "diag": diag.splitlines()[:4],
        }
        event["ok"] = True
    except Exception as exc:
        event["ok"] = False
        event["error"] = str(exc)[-4000:]
    event["wall_s"] = round(time.time() - started, 3)
    with (out_dir / "checkpoint_eval.jsonl").open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
    tmp = out_dir / f".checkpoint_eval_latest.json.{os.getpid()}.{time.time_ns()}.tmp"
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_dir / "checkpoint_eval_latest.json")
    write_training_heartbeat(
        {
            "event": "checkpoint_eval",
            "counter": counter,
            "reason": reason,
            "ok": event.get("ok"),
            "battery_perf": event.get("battery_perf"),
            "forward_perf": event.get("forward_perf"),
            "stand_perf": event.get("stand_perf"),
            "push_perf": event.get("push_perf"),
            "push_falls": event.get("push_falls"),
            "leg_qvel_rms": event.get("leg_qvel_rms"),
            "foot_sep_min_m": event.get("foot_sep_min_m"),
            "foot_sep_under_0p10_pct": event.get("foot_sep_under_0p10_pct"),
            "wall_s": event.get("wall_s"),
        }
    )
    return event


def request_training_stop(proc: subprocess.Popen[str], reason: str) -> int:
    print(f"{reason}; asking trainer to stop", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        rc = proc.poll()
        return rc if rc is not None else 1
    try:
        return proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("trainer did not exit after SIGINT; sending SIGTERM", flush=True)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("trainer did not exit after SIGTERM; killing", flush=True)
            os.killpg(proc.pid, signal.SIGKILL)
            return proc.wait()


def wait_for_training(
    env: dict[str, str],
    max_train_seconds: int,
    survival_threshold: float,
    post_survival_seconds: int,
    stall_seconds: int,
    checkpoint_mirror_seconds: int,
    checkpoint_eval_seconds: int,
) -> tuple[int, bool, dict[str, object]]:
    proc = subprocess.Popen(
        ["puffer", "train", "g1gpu"],
        cwd=PUFFER,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    q: queue.Queue[str | object] = queue.Queue()

    def reader() -> None:
        try:
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(EOF_SENTINEL)

    threading.Thread(target=reader, daemon=True).start()

    t0 = time.perf_counter()
    metrics_path = OUTPUT_DIR / "fall_metrics.jsonl" if OUTPUT_DIR else None
    last: dict[str, object] = {}
    survival_hit = False
    survival_hit_wall_s: float | None = None
    post_deadline: float | None = None
    stop_reason: str | None = None
    reader_done = False
    last_progress_at = t0
    last_progress_label = "trainer_start"
    last_steps: float | None = None
    progress_id = 0
    last_mirror_at = 0.0
    last_mirror_info: dict[str, object] | None = None
    last_mirror_counter: int | None = None
    last_eval_at = 0.0
    last_eval_counter: int | None = None
    eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-eval")
    eval_future: concurrent.futures.Future[dict[str, object] | None] | None = None
    eval_future_counter: int | None = None
    eval_started_at: float | None = None
    process_exit_seen_at: float | None = None
    raw_log_path = OUTPUT_DIR / "train.log" if OUTPUT_DIR else None
    raw_log = raw_log_path.open("a") if raw_log_path else None
    if raw_log:
        raw_log.write("$ puffer train g1gpu\n")
        raw_log.flush()
    write_training_heartbeat(
        {
            "event": "trainer_start",
            "progress_id": progress_id,
            "wall_s": 0.0,
            "last_progress_label": last_progress_label,
        }
    )

    try:
        while proc.poll() is None or not reader_done:
            try:
                line = q.get(timeout=0.5)
            except queue.Empty:
                line = ""
            if line is EOF_SENTINEL:
                reader_done = True
            elif line == "":
                pass
            else:
                assert isinstance(line, str)
                if raw_log:
                    raw_log.write(line)
                    raw_log.flush()
                else:
                    print(line, end="", flush=True)
                clean = ANSI_RE.sub("", line.replace("│", " "))
                values = {k: float(v) for k, v in METRIC_RE.findall(clean)}
                steps_match = STEPS_RE.search(clean)
                if steps_match:
                    steps_value = parse_count(steps_match.group(1), steps_match.group(2))
                    last["steps"] = steps_value
                    if last_steps is None or steps_value > last_steps:
                        last_steps = steps_value
                        last_progress_at = time.perf_counter()
                        last_progress_label = f"steps={steps_value:g}"
                        progress_id += 1
                        write_training_heartbeat(
                            {
                                "event": "steps_advanced",
                                "progress_id": progress_id,
                                "wall_s": round(last_progress_at - t0, 3),
                                "steps": steps_value,
                                "last_progress_label": last_progress_label,
                            }
                        )
                if values:
                    last.update(values)
                    last_progress_at = time.perf_counter()
                    last_progress_label = ",".join(sorted(values))
                    progress_id += 1
                    write_training_heartbeat(
                        {
                            "event": "metrics_advanced",
                            "progress_id": progress_id,
                            "wall_s": round(last_progress_at - t0, 3),
                            "steps": last.get("steps"),
                            "perf": last.get("perf"),
                            "episode_length": last.get("episode_length"),
                            "falls": last.get("falls"),
                            "last_progress_label": last_progress_label,
                        }
                    )
                    if "falls" in values:
                        falls = max(0.0, min(1.0, values["falls"]))
                        survival = 1.0 - falls
                        event = {
                            "wall_s": round(time.perf_counter() - t0, 3),
                            "steps": last.get("steps"),
                            "falls": round(falls, 6),
                            "fall_pct": round(100.0 * falls, 3),
                            "survival_pct": round(100.0 * survival, 3),
                            "perf": last.get("perf"),
                            "episode_length": last.get("episode_length"),
                        }
                        append_fall_metric(metrics_path, event)
                        print(
                            "FALL_METRIC "
                            f"survival_pct={event['survival_pct']:.3f} "
                            f"fall_pct={event['fall_pct']:.3f} "
                            f"perf={event.get('perf')} "
                            f"episode_length={event.get('episode_length')}",
                            flush=True,
                        )
                        if (
                            survival_threshold > 0
                            and not survival_hit
                            and survival >= survival_threshold
                        ):
                            survival_hit = True
                            survival_hit_wall_s = float(event["wall_s"])
                            if post_survival_seconds > 0:
                                post_deadline = time.perf_counter() + post_survival_seconds
                            append_fall_metric(metrics_path, {**event, "event": "survival_threshold_hit"})
                            print(
                                "SURVIVAL_THRESHOLD_HIT "
                                f"survival_pct={event['survival_pct']:.3f} "
                                f"continuing_for_s={post_survival_seconds}",
                                flush=True,
                            )

            now = time.perf_counter()
            if eval_future is not None and eval_future.done():
                try:
                    eval_event = eval_future.result()
                except Exception as exc:
                    eval_event = {"ok": False, "error": str(exc)[-4000:]}
                    if OUTPUT_DIR:
                        event = {
                            "time_unix": round(time.time(), 3),
                            "run_started_unix": current_run_started_unix(OUTPUT_DIR),
                            "counter": eval_future_counter,
                            "reason": "periodic",
                            **eval_event,
                            "wall_s": round(time.perf_counter() - (eval_started_at or time.perf_counter()), 3),
                        }
                        with (OUTPUT_DIR / "checkpoint_eval.jsonl").open("a") as f:
                            f.write(json.dumps(event, sort_keys=True) + "\n")
                        tmp = OUTPUT_DIR / f".checkpoint_eval_latest.json.{os.getpid()}.{time.time_ns()}.tmp"
                        tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
                        tmp.replace(OUTPUT_DIR / "checkpoint_eval_latest.json")
                if eval_event:
                    print(
                        "CHECKPOINT_EVAL "
                        f"counter={eval_future_counter} "
                        f"ok={eval_event.get('ok')} "
                        f"forward_perf={eval_event.get('forward_perf')} "
                        f"stand_perf={eval_event.get('stand_perf')} "
                        f"push_falls={eval_event.get('push_falls')} "
                        f"push_perf={eval_event.get('push_perf')} "
                        f"foot_sep_min_m={eval_event.get('foot_sep_min_m')}",
                        flush=True,
                    )
                eval_future = None
                eval_future_counter = None
                eval_started_at = None
            proc_rc = proc.poll()
            if proc_rc is not None:
                if process_exit_seen_at is None:
                    process_exit_seen_at = now
                    write_training_heartbeat(
                        {
                            "event": "trainer_process_exited",
                            "progress_id": progress_id,
                            "wall_s": round(now - t0, 3),
                            "exit_code": proc_rc,
                            "steps": last.get("steps"),
                            "last_progress_age_s": round(now - last_progress_at, 3),
                            "reader_done": reader_done,
                        }
                    )
                if reader_done or (now - process_exit_seen_at) >= 5.0:
                    if not reader_done:
                        write_training_heartbeat(
                            {
                                "event": "trainer_stdout_left_open",
                                "progress_id": progress_id,
                                "wall_s": round(now - t0, 3),
                                "exit_code": proc_rc,
                                "steps": last.get("steps"),
                                "last_progress_age_s": round(now - last_progress_at, 3),
                            }
                        )
                    break
            if checkpoint_mirror_seconds > 0 and OUTPUT_DIR and now - last_mirror_at >= checkpoint_mirror_seconds:
                info = mirror_latest_checkpoint(PUFFER / "checkpoints", OUTPUT_DIR, reason="periodic")
                if info:
                    checkpoint_counter = int(info["counter"])
                    if last_mirror_counter is None or checkpoint_counter > last_mirror_counter:
                        last_progress_at = time.perf_counter()
                        last_progress_label = f"checkpoint={checkpoint_counter}"
                        progress_id += 1
                        write_training_heartbeat(
                            {
                                "event": "checkpoint_advanced",
                                "progress_id": progress_id,
                                "wall_s": round(last_progress_at - t0, 3),
                                "steps": last.get("steps"),
                                "checkpoint_counter": checkpoint_counter,
                                "last_progress_label": last_progress_label,
                            }
                        )
                    last_mirror_counter = checkpoint_counter
                if info and info.get("counter") != (last_mirror_info or {}).get("counter"):
                    last_mirror_info = info
                    print(
                        "CHECKPOINT_MIRROR "
                        f"counter={info['counter']} "
                        f"path={OUTPUT_DIR / 'latest.bin'}",
                        flush=True,
                    )
                if (
                    info
                    and checkpoint_eval_seconds > 0
                    and (last_eval_counter != int(info["counter"]))
                    and now - last_eval_at >= checkpoint_eval_seconds
                    and eval_future is None
                ):
                    checkpoint_counter = int(info["counter"])
                    snapshot = OUTPUT_DIR / f".checkpoint_eval_{checkpoint_counter}_{os.getpid()}.bin"
                    shutil.copy2(OUTPUT_DIR / "latest.bin", snapshot)
                    last_eval_counter = checkpoint_counter
                    last_eval_at = time.perf_counter()
                    eval_started_at = last_eval_at
                    eval_future_counter = checkpoint_counter
                    write_training_heartbeat(
                        {
                            "event": "checkpoint_eval_started",
                            "progress_id": progress_id,
                            "wall_s": round(last_eval_at - t0, 3),
                            "steps": last.get("steps"),
                            "counter": checkpoint_counter,
                        }
                    )
                    eval_future = eval_executor.submit(
                        run_checkpoint_eval_snapshot,
                        snapshot,
                        counter=checkpoint_counter,
                        reason="periodic",
                    )
                last_mirror_at = now
            if max_train_seconds > 0 and now - t0 >= max_train_seconds:
                stop_reason = f"training time limit reached after {max_train_seconds}s"
                write_training_heartbeat(
                    {
                        "event": "stop_requested",
                        "progress_id": progress_id,
                        "wall_s": round(now - t0, 3),
                        "steps": last.get("steps"),
                        "stop_reason": stop_reason,
                        "last_progress_age_s": round(now - last_progress_at, 3),
                    }
                )
                request_training_stop(proc, stop_reason)
                break
            if stall_seconds > 0 and now - last_progress_at >= stall_seconds:
                stop_reason = (
                    f"training progress stalled for {stall_seconds}s "
                    f"after {last_progress_label}"
                )
                write_training_heartbeat(
                    {
                        "event": "stall_detected",
                        "progress_id": progress_id,
                        "wall_s": round(now - t0, 3),
                        "steps": last.get("steps"),
                        "stop_reason": stop_reason,
                        "last_progress_age_s": round(now - last_progress_at, 3),
                    }
                )
                request_training_stop(proc, stop_reason)
                break
            if post_deadline is not None and now >= post_deadline:
                stop_reason = (
                    f"post-survival training complete after {post_survival_seconds}s "
                    f"above survival threshold {survival_threshold:.3f}"
                )
                write_training_heartbeat(
                    {
                        "event": "post_survival_complete",
                        "progress_id": progress_id,
                        "wall_s": round(now - t0, 3),
                        "steps": last.get("steps"),
                        "stop_reason": stop_reason,
                        "last_progress_age_s": round(now - last_progress_at, 3),
                    }
                )
                request_training_stop(proc, stop_reason)
                break
    finally:
        eval_executor.shutdown(wait=False, cancel_futures=True)
        if raw_log:
            raw_log.close()

    if OUTPUT_DIR:
        info = mirror_latest_checkpoint(PUFFER / "checkpoints", OUTPUT_DIR, reason="trainer_exit")
        if info and info != last_mirror_info:
            last_mirror_info = info
    rc = proc.poll()
    if rc is None:
        rc = proc.wait()
    timed_out = stop_reason is not None
    info = {
        "survival_threshold": survival_threshold or None,
        "post_survival_seconds": post_survival_seconds or None,
        "survival_threshold_hit": survival_hit,
        "survival_hit_wall_s": round(survival_hit_wall_s, 1) if survival_hit_wall_s is not None else None,
        "stop_reason": stop_reason,
        "stall_seconds": stall_seconds or None,
        "checkpoint_mirror_seconds": checkpoint_mirror_seconds or None,
        "checkpoint_eval_seconds": checkpoint_eval_seconds or None,
        "last_progress_age_s": round(time.perf_counter() - last_progress_at, 1),
        "last_progress_label": last_progress_label,
        "progress_id": progress_id,
        "last_mirrored_checkpoint": last_mirror_info,
        "last_falls": last.get("falls"),
        "last_fall_pct": round(100.0 * float(last["falls"]), 3) if "falls" in last else None,
        "last_survival_pct": round(100.0 * (1.0 - float(last["falls"])), 3) if "falls" in last else None,
        "fall_metrics": str(metrics_path) if metrics_path else None,
    }
    return rc if stop_reason is None else 124, timed_out, info


def train_local(
    total_timesteps: int,
    smoke: bool,
    max_train_seconds: int,
    survival_threshold: float,
    post_survival_seconds: int,
    stall_seconds: int,
    checkpoint_mirror_seconds: int,
    checkpoint_eval_seconds: int,
) -> dict[str, object]:
    t0 = time.perf_counter()
    tt = SMOKE_TIMESTEPS if (smoke and total_timesteps == 0) else (total_timesteps or R.TOTAL_TIMESTEPS)
    env = {**os.environ, "G1_MODEL_PATH": str(MODEL)}
    emit_heartbeat({"event": "train_local_start", "progress_id": 0})

    run(["git", "log", "--oneline", "-1"], cwd=PUFFER)
    overrides = R.overrides_str() + (",base.checkpoint_interval=2" if smoke else "")
    emit_heartbeat({"event": "config_overrides_start", "progress_id": 0})
    apply_overrides(PUFFER / "config" / "g1gpu.ini", overrides, tt)
    shutil.rmtree(PUFFER / "checkpoints", ignore_errors=True)
    shutil.rmtree(PUFFER / "logs" / "g1gpu", ignore_errors=True)
    emit_heartbeat({"event": "trainer_launching", "progress_id": 0})

    print(f"training nanoG1 locally: {tt:,} steps (recipe baked)", flush=True)
    train_start = time.perf_counter()
    rc, timed_out, survival_info = wait_for_training(
        env,
        max_train_seconds=max_train_seconds,
        survival_threshold=survival_threshold,
        post_survival_seconds=post_survival_seconds,
        stall_seconds=stall_seconds,
        checkpoint_mirror_seconds=checkpoint_mirror_seconds,
        checkpoint_eval_seconds=checkpoint_eval_seconds,
    )
    train_s = time.perf_counter() - train_start

    sps, pts = steady_sps(PUFFER / "checkpoints")
    steps, perf = perf_curve(PUFFER / "logs" / "g1gpu")

    walk_counter = None
    if pts:
        target = int(R.WALK_SAMPLES * pts[-1][0] / tt)
        walk_counter = min((counter for counter, _ in pts), key=lambda counter: abs(counter - target))

    walk_path = None
    if walk_counter:
        for path in (PUFFER / "checkpoints").rglob(f"{walk_counter:016d}.bin"):
            walk_path = path
            break
    latest_path = None
    if pts:
        latest_counter = pts[-1][0]
        for path in (PUFFER / "checkpoints").rglob(f"{latest_counter:016d}.bin"):
            latest_path = path
            break
    export_path = latest_path or walk_path
    export_strategy = "latest" if latest_path else ("walk_samples" if walk_path else None)

    t_walk = R.WALK_SAMPLES / sps if sps else None
    perf_at_walk = None
    if steps and perf:
        i = min(range(len(steps)), key=lambda j: abs(steps[j] - R.WALK_SAMPLES))
        perf_at_walk = round(perf[i], 3)

    result = {
        "exit_code": rc,
        "total_timesteps": tt,
        "steady_sps": round(sps, 1) if sps else None,
        "physics_steps_per_s": round(sps * R.DECIMATION, 1) if sps else None,
        "T_walk_s": round(t_walk, 1) if t_walk else None,
        "walk_samples": R.WALK_SAMPLES,
        "perf_at_walk": perf_at_walk,
        "final_perf": round(perf[-1], 3) if perf else None,
        "train_wall_s": round(train_s, 1),
        "timed_out": timed_out,
        "max_train_seconds": max_train_seconds or None,
        "total_wall_s": round(time.perf_counter() - t0, 1),
        "walk_checkpoint": str(walk_path) if walk_path else None,
        "latest_checkpoint": str(latest_path) if latest_path else None,
        "export_checkpoint": str(export_path) if export_path else None,
        "export_strategy": export_strategy,
        **survival_info,
    }
    print("\n=== nanoG1 LOCAL RESULT ===")
    print(json.dumps(result, indent=2))
    print("=== END RESULT ===\n", flush=True)

    export_allowed = True
    if OUTPUT_DIR:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        archive_tree(PUFFER / "checkpoints", OUTPUT_DIR / "checkpoints.tar.gz")
        archive_tree(PUFFER / "logs" / "g1gpu", OUTPUT_DIR / "puffer_logs.tar.gz")
        info = mirror_latest_checkpoint(PUFFER / "checkpoints", OUTPUT_DIR, reason="final")
        if info:
            print(f"copied latest checkpoint to {OUTPUT_DIR / 'latest.bin'}", flush=True)
            eval_event = run_checkpoint_eval(
                OUTPUT_DIR / "latest.bin",
                OUTPUT_DIR,
                counter=int(info["counter"]),
                reason="final",
            )
            if eval_event:
                result["final_checkpoint_eval"] = {
                    "battery_perf": eval_event.get("battery_perf"),
                    "battery_falls": eval_event.get("battery_falls"),
                    "forward_perf": eval_event.get("forward_perf"),
                    "forward_falls": eval_event.get("forward_falls"),
                    "stand_perf": eval_event.get("stand_perf"),
                    "diag_falls": eval_event.get("diag_falls"),
                    "push_falls": eval_event.get("push_falls"),
                    "push_perf": eval_event.get("push_perf"),
                    "leg_qvel_rms": eval_event.get("leg_qvel_rms"),
                    "foot_sep_min_m": eval_event.get("foot_sep_min_m"),
                    "foot_sep_under_0p10_pct": eval_event.get("foot_sep_under_0p10_pct"),
                }
                export_allowed, blocker = checkpoint_eval_deployable(eval_event)
                result["final_checkpoint_deployable"] = export_allowed
                result["final_checkpoint_blocker"] = blocker or None
        if latest_path:
            latest_path = OUTPUT_DIR / "latest.bin"
        if export_path and export_allowed:
            shutil.copy2(export_path, OUTPUT_DIR / "nanoG1.bin")
            print(f"copied {export_strategy} checkpoint to {OUTPUT_DIR / 'nanoG1.bin'}", flush=True)
        elif export_path:
            print(
                f"not copying deploy artifact: final checkpoint failed deploy gate "
                f"({result.get('final_checkpoint_blocker')})",
                flush=True,
            )
        (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    if not smoke and export_path and export_allowed:
        out = ROOT / "assets" / "nanoG1.bin"
        out.parent.mkdir(exist_ok=True)
        shutil.copy2(export_path, out)
        print(f"wrote {out}", flush=True)
    elif not smoke and export_path:
        print(
            f"not writing {ROOT / 'assets' / 'nanoG1.bin'}: final checkpoint failed deploy gate "
            f"({result.get('final_checkpoint_blocker')})",
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nanoG1 locally on a DGX Spark/CUDA host")
    parser.add_argument("--smoke", action="store_true", help="Run a 10M-step stack validation")
    parser.add_argument("--full", action="store_true", help="Run full training even if --smoke was passed earlier")
    parser.add_argument("--total-timesteps", type=int, default=0, help="Override training budget")
    parser.add_argument("--max-train-seconds", type=int, default=0, help="Stop training after this many seconds and collect latest checkpoint")
    parser.add_argument("--stall-seconds", type=int, default=900, help="Stop and package if trainer progress does not advance for this many seconds")
    parser.add_argument("--checkpoint-mirror-seconds", type=int, default=120, help="Mirror latest checkpoint to NANOG1_OUTPUT_DIR/latest.bin this often during training")
    parser.add_argument("--checkpoint-eval-seconds", type=int, default=600, help="Run host-policy eval on the mirrored checkpoint this often; writes checkpoint_eval.jsonl")
    parser.add_argument("--survival-threshold", type=float, default=0.0, help="Stop 30m-style post-training timer once survival reaches this fraction, e.g. 0.98")
    parser.add_argument("--post-survival-seconds", type=int, default=0, help="Keep training this many seconds after --survival-threshold is reached")
    parser.add_argument("--nvcc-arch", default="", help="Override NVCC_ARCH, e.g. sm_120")
    parser.add_argument("--skip-setup", action="store_true", help="Do not run setup.sh before training")
    parser.add_argument("--skip-build", action="store_true", help="Do not rebuild the g1gpu engine")
    parser.add_argument("--force-model", action="store_true", help="Regenerate envs/g1/model/g1.mjb")
    args = parser.parse_args()

    emit_heartbeat({"event": "startup", "progress_id": 0})
    ensure_cuda()
    emit_heartbeat({"event": "cuda_ready", "progress_id": 0})
    ensure_engine(skip_setup=args.skip_setup)
    emit_heartbeat({"event": "engine_ready", "progress_id": 0})
    ensure_model(force=args.force_model)
    emit_heartbeat({"event": "model_ready", "progress_id": 0})
    build_engine(args.nvcc_arch or default_nvcc_arch(), skip_build=args.skip_build)
    emit_heartbeat({"event": "build_ready", "progress_id": 0})
    result = train_local(
        total_timesteps=args.total_timesteps,
        smoke=args.smoke and not args.full,
        max_train_seconds=args.max_train_seconds,
        survival_threshold=args.survival_threshold,
        post_survival_seconds=args.post_survival_seconds,
        stall_seconds=args.stall_seconds,
        checkpoint_mirror_seconds=args.checkpoint_mirror_seconds,
        checkpoint_eval_seconds=args.checkpoint_eval_seconds,
    )
    if result.get("exit_code") not in (0, 124):
        sys.exit(int(result["exit_code"]))


if __name__ == "__main__":
    main()
