#!/usr/bin/env python3
"""Live MuJoCo preview for real-camera G1 follow commands.

Polls the browser follow dashboard /target.json endpoint, feeds its vx/vy/wz
command into the nanoG1 walking policy, steps the MuJoCo G1 model locally, and
serves a browser preview. It never contacts or commands the physical robot.
"""
from __future__ import annotations

import argparse
import ctypes
import io
import json
import math
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "perception-sim" / "model-mujoco39" / "g1.mjb"
DEFAULT_BIN = ROOT / "assets" / "nanoG1.bin"
DEFAULT_LIB = ROOT / "deploy" / "libnanog1policy.dylib"
DEFAULT_DASHBOARD = "http://127.0.0.1:8094/target.json"

NU = 29
LEG_DOF = 12
CONTROL_DT = 0.02
PHASE_PERIOD = 40
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
HOME = np.array([
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
     0.0, 0.0, 0.0,
     0.20, 0.20, 0.0, 1.28, 0.0, 0.0, 0.0,
     0.20,-0.20, 0.0, 1.28, 0.0, 0.0, 0.0,
], dtype=np.float64)
CTRL_RANGE = np.array([
    (-2.5307,2.8798),(-0.5236,2.9671),(-2.7576,2.7576),(-0.087267,2.8798),(-0.87267,0.5236),(-0.2618,0.2618),
    (-2.5307,2.8798),(-0.5236,2.9671),(-2.7576,2.7576),(-0.087267,2.8798),(-0.87267,0.5236),(-0.2618,0.2618),
    (-2.618,2.618),(-0.52,0.52),(-0.52,0.52),
    (-3.0892,2.6704),(-1.5882,2.2515),(-2.618,2.618),(-1.0472,2.0944),(-1.97222,1.97222),(-1.61443,1.61443),(-1.61443,1.61443),
    (-3.0892,2.6704),(-2.2515,1.5882),(-2.618,2.618),(-1.0472,2.0944),(-1.97222,1.97222),(-1.61443,1.61443),(-1.61443,1.61443),
], dtype=np.float64)


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array([-2 * (x * z + w * y), -2 * (y * z - w * x), -(1 - 2 * (x * x + y * y))])


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_policy(lib_path: Path, bin_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.nn_init.restype = ctypes.c_int
    lib.nn_obs.restype = ctypes.c_int
    lib.nn_nu.restype = ctypes.c_int
    if lib.nn_init(str(bin_path).encode()) != 0:
        raise RuntimeError(f"policy load failed: {bin_path}")
    obs_n, nu = lib.nn_obs(), lib.nn_nu()
    if obs_n != 98 or nu != NU:
        raise RuntimeError(f"policy shape mismatch obs={obs_n} nu={nu}")
    obs_buf = (ctypes.c_float * obs_n)()
    act_buf = (ctypes.c_float * nu)()

    def infer(obs: np.ndarray) -> np.ndarray:
        obs_buf[:] = obs.astype(np.float32)
        lib.nn_infer(obs_buf, act_buf)
        return np.frombuffer(act_buf, dtype=np.float32).copy()

    return infer


def fetch_target(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


class PreviewState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg = b""
        self.state: dict[str, Any] = {"status": "starting"}
        self.stop = False

    def update(self, jpeg: bytes, state: dict[str, Any]) -> None:
        with self.lock:
            self.jpeg = jpeg
            self.state = state

    def snapshot(self) -> tuple[bytes, dict[str, Any]]:
        with self.lock:
            return self.jpeg, dict(self.state)


class MujocoPreview:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = mujoco.MjModel.from_binary_path(str(args.model))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=args.height, width=args.width)
        self.infer = load_policy(args.lib, args.bin)
        self.prev_action = np.zeros(NU)
        self.cmd = np.zeros(3)
        self.step = 0
        self.last_target: dict[str, Any] = {}
        self.last_fetch_error = ""
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0:
            self.data.qpos[:] = self.model.key_qpos[home_key]
            self.data.qvel[:] = 0.0
        self.data.qpos[7:7 + NU] = HOME
        self.data.ctrl[:] = HOME
        mujoco.mj_forward(self.model, self.data)
        self.prev_action[:] = 0.0
        self.cmd[:] = 0.0
        self.step = 0

    def command_from_dashboard(self) -> tuple[np.ndarray, bool, str]:
        try:
            payload = fetch_target(self.args.dashboard_url, self.args.dashboard_timeout)
            self.last_target = payload
            self.last_fetch_error = ""
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self.last_fetch_error = f"{type(exc).__name__}: {exc}"
            return np.zeros(3), False, "dashboard_fetch_failed"

        target_time = self.last_target.get("time_unix")
        if target_time is None or time.time() - float(target_time) > self.args.max_target_age:
            return np.zeros(3), False, "stale_dashboard_target"
        if self.args.require_go and not (self.last_target.get("motion_control") or {}).get("enabled", False):
            return np.zeros(3), False, "browser_stop"
        if (self.last_target.get("safety_stop") or {}).get("active", False):
            return np.zeros(3), False, "dashboard_safety_stop"
        raw = self.last_target.get("wanted_input_vx_vy_wz_stop")
        if not isinstance(raw, list) or len(raw) < 4:
            return np.zeros(3), False, "missing_command"
        if float(raw[3]) >= self.args.stop_threshold:
            return np.zeros(3), False, f"policy_stop:{float(raw[3]):.2f}"
        cmd = self.args.command_scale * np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64)
        limit = np.array([self.args.max_vx, self.args.max_vy, self.args.max_wz], dtype=np.float64)
        return np.clip(cmd, -limit, limit), True, "live_dashboard_command"

    def build_obs(self) -> np.ndarray:
        obs = np.zeros(98, dtype=np.float64)
        quat = self.data.qpos[3:7].copy()
        q = self.data.qpos[7:7 + NU].copy()
        dq = self.data.qvel[6:6 + NU].copy()
        obs[0:3] = ANG_VEL_SCALE * self.data.qvel[3:6]
        obs[3:6] = projected_gravity(quat)
        obs[6:9] = self.cmd
        obs[9:38] = q - HOME
        obs[38:67] = DOF_VEL_SCALE * dq
        obs[67:96] = self.prev_action
        phase = 2 * math.pi * ((self.step % PHASE_PERIOD) / PHASE_PERIOD)
        obs[96], obs[97] = math.sin(phase), math.cos(phase)
        return obs

    def control_step(self) -> dict[str, Any]:
        self.cmd, active, reason = self.command_from_dashboard()
        if active:
            act = self.infer(self.build_obs())
            target = HOME.copy()
            for i in range(NU):
                c = float(np.clip(act[i], -1.0, 1.0))
                if i >= LEG_DOF:
                    c = 0.0
                self.prev_action[i] = c
                target[i] = np.clip(HOME[i] + ACTION_SCALE * c, CTRL_RANGE[i, 0], CTRL_RANGE[i, 1])
        else:
            target = HOME.copy()
            self.prev_action[:] = 0.0
        self.data.ctrl[:] = target
        substeps = max(1, int(round(CONTROL_DT / self.model.opt.timestep)))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        self.step += 1
        if self.data.qpos[2] < 0.35 or not np.isfinite(self.data.qpos).all():
            self.reset()
            reason = "sim_fall_reset"
            active = False
        return {
            "active": active,
            "reason": reason,
            "cmd_vx_vy_wz": [float(x) for x in self.cmd],
            "leg_action_rms": float(np.sqrt(np.mean(self.prev_action[:LEG_DOF] ** 2))),
            "leg_target_delta_max_rad": float(np.max(np.abs(self.data.ctrl[:LEG_DOF] - HOME[:LEG_DOF]))),
        }

    def render(self, info: dict[str, Any]) -> bytes:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        yaw = yaw_from_quat(self.data.qpos[3:7])
        hx, hy = math.cos(yaw), math.sin(yaw)
        cam.lookat[:] = [self.data.qpos[0], self.data.qpos[1], 0.75]
        cam.distance = 3.2
        cam.azimuth = math.degrees(math.atan2(hy, hx)) + 180.0
        cam.elevation = -18.0
        self.renderer.update_scene(self.data, camera=cam)
        rgb = self.renderer.render()
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
            bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        except Exception:
            font = bold = None
        target = self.last_target
        motion = target.get("motion_control") or {}
        lines = [
            "MuJoCo preview only - physical robot is not commanded",
            f"sim active={info['active']} reason={info['reason']}",
            f"cmd vx={info['cmd_vx_vy_wz'][0]:+.3f} vy={info['cmd_vx_vy_wz'][1]:+.3f} wz={info['cmd_vx_vy_wz'][2]:+.3f}",
            f"leg action rms={info['leg_action_rms']:.3f} target delta={info['leg_target_delta_max_rad']:.3f} rad",
            f"browser_go={bool(motion.get('enabled', False))} require_go={self.args.require_go}",
            f"policy_source={target.get('policy_target_source')} wanted={target.get('wanted_input_vx_vy_wz_stop')}",
        ]
        draw.rectangle((0, 0, image.width, 168), fill=(255, 255, 255))
        draw.text((16, 12), lines[0], fill=(128, 26, 26), font=bold)
        y = 44
        for line in lines[1:]:
            draw.text((16, y), line, fill=(25, 35, 48), font=font)
            y += 22
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=88)
        return buf.getvalue()


def worker(shared: PreviewState, args: argparse.Namespace) -> None:
    sim = MujocoPreview(args)
    next_t = time.time()
    while not shared.stop:
        info = sim.control_step()
        jpeg = sim.render(info)
        payload = {
            "time_unix": round(time.time(), 3),
            "sim_step": sim.step,
            "qpos_xyz_yaw": [float(sim.data.qpos[0]), float(sim.data.qpos[1]), float(sim.data.qpos[2]), yaw_from_quat(sim.data.qpos[3:7])],
            "dashboard_url": args.dashboard_url,
            "dashboard_fetch_error": sim.last_fetch_error,
            "dashboard_target": sim.last_target,
            **info,
        }
        shared.update(jpeg, payload)
        next_t += CONTROL_DT
        time.sleep(max(0.0, next_t - time.time()))


def handler_factory(shared: PreviewState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/?"):
                html = """<!doctype html><html><head><title>G1 MuJoCo Preview</title>
<style>body{margin:0;background:#101318;color:#e5e7eb;font-family:Arial,sans-serif}main{max-width:1100px;margin:0 auto;padding:14px}img{width:100%;background:#222}pre{font-size:13px;white-space:pre-wrap}</style>
</head><body><main><h2>G1 MuJoCo Live Command Preview</h2><img id="view" src="/snapshot.jpg"><pre id="state"></pre>
<script>
async function tick(){
  document.getElementById('view').src='/snapshot.jpg?t='+Date.now();
  try { let r=await fetch('/state?t='+Date.now()); document.getElementById('state').textContent=JSON.stringify(await r.json(), null, 2); } catch(e) {}
}
setInterval(tick, 250); tick();
</script></main></body></html>"""
                data = html.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.startswith("/snapshot.jpg"):
                jpeg, _ = shared.snapshot()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path.startswith("/state"):
                _, state = shared.snapshot()
                data = json.dumps(state, indent=2, sort_keys=True).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--dashboard-url", default=DEFAULT_DASHBOARD)
    parser.add_argument("--dashboard-timeout", type=float, default=0.15)
    parser.add_argument("--max-target-age", type=float, default=3.0)
    parser.add_argument("--command-scale", type=float, default=0.50)
    parser.add_argument("--max-vx", type=float, default=0.15)
    parser.add_argument("--max-vy", type=float, default=0.08)
    parser.add_argument("--max-wz", type=float, default=0.25)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--require-go", action="store_true", help="Only simulate movement while the dashboard browser button is GO")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    for path in (args.model, args.bin, args.lib):
        if not path.exists():
            raise SystemExit(f"missing required file: {path}")

    shared = PreviewState()
    thread = threading.Thread(target=worker, args=(shared, args), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.bind, args.port), handler_factory(shared))
    print(f"serving MuJoCo preview http://{args.bind}:{args.port}", flush=True)
    print("SIM ONLY: no physical robot commands are sent", flush=True)
    try:
        server.serve_forever()
    finally:
        shared.stop = True
        server.server_close()


if __name__ == "__main__":
    main()
