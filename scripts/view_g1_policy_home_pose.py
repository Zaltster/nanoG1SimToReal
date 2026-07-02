#!/usr/bin/env python3
"""Serve a MuJoCo render of the exact G1 policy HOME pose."""
from __future__ import annotations

import argparse
import io
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mujoco
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "perception-sim" / "model-mujoco39" / "g1.mjb"


def render_home(model_path: Path, width: int, height: int, azimuth: float, elevation: float, distance: float) -> tuple[bytes, dict[str, Any]]:
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise RuntimeError("model has no 'home' keyframe")
    data.qpos[:] = model.key_qpos[key_id]
    data.qvel[:] = 0.0
    data.ctrl[:] = model.key_ctrl[key_id] if model.nkey and model.key_ctrl.size else data.qpos[7:7 + model.nu]
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.78]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera=cam)
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    try:
        title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    except Exception:
        title = font = None
    lines = [
        "Policy HOME pose",
        "Base z=0.785m, torso upright, symmetric slight crouch",
        "Legs: hip pitch -0.10, knee +0.30, ankle pitch -0.20 rad",
        "Use this as visual target for the airborne robot before GO.",
    ]
    draw.rectangle((0, 0, width, 120), fill=(255, 255, 255))
    draw.text((18, 14), lines[0], fill=(20, 28, 38), font=title)
    y = 46
    for line in lines[1:]:
        draw.text((18, y), line, fill=(35, 45, 60), font=font)
        y += 22
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    state = {
        "model": str(model_path),
        "keyframe": "home",
        "base_xyz_quat": [float(x) for x in data.qpos[:7]],
        "joint_qpos": [float(x) for x in data.qpos[7:7 + model.nu]],
        "note": "Static MuJoCo render of the target policy HOME pose; no physical robot commands.",
    }
    return buf.getvalue(), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-16.0)
    parser.add_argument("--distance", type=float, default=3.0)
    args = parser.parse_args()
    jpeg, state = render_home(args.model, args.width, args.height, args.azimuth, args.elevation, args.distance)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/?"):
                html = """<!doctype html><html><head><title>G1 Policy HOME</title>
<style>body{margin:0;background:#111;color:#eee;font-family:Arial,sans-serif}main{max-width:1000px;margin:0 auto;padding:14px}img{width:100%;background:#222}pre{font-size:13px;white-space:pre-wrap}</style>
</head><body><main><h2>G1 Policy HOME Pose</h2><img src="/home.jpg"><pre id="state"></pre>
<script>fetch('/state').then(r=>r.json()).then(j=>state.textContent=JSON.stringify(j,null,2))</script>
</main></body></html>"""
                data = html.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.startswith("/home.jpg"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path.startswith("/state"):
                data = json.dumps(state, indent=2).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"serving policy HOME pose http://{args.bind}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
