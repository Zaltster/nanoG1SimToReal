#!/usr/bin/env python3
"""G1 RGB + depth live dashboard.

This intentionally does not mix the front RGB camera with RealSense depth. It
shows RGB for visual context, then reports raw RealSense depth region
percentiles so we can debug real distances without RGB/depth projection error.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RawDepthSource:
    def __init__(self, cmd: list[str], width: int, height: int) -> None:
        self.cmd = cmd
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 2
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self.stop()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.proc is None or self.proc.stdout is None or self.proc.poll() is not None:
            self.start()
            time.sleep(0.25)
        assert self.proc is not None and self.proc.stdout is not None
        data = self.proc.stdout.read(self.frame_bytes)
        if len(data) != self.frame_bytes:
            self.start()
            return False, None
        return True, np.frombuffer(data, dtype=np.uint16).reshape((self.height, self.width)).copy()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=1.5)


class RawRgbSource:
    def __init__(self, cmd: list[str], width: int, height: int) -> None:
        self.cmd = cmd
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self.stop()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.proc is None or self.proc.stdout is None or self.proc.poll() is not None:
            self.start()
            time.sleep(0.25)
        assert self.proc is not None and self.proc.stdout is not None
        data = self.proc.stdout.read(self.frame_bytes)
        if len(data) != self.frame_bytes:
            self.start()
            return False, None
        return True, np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=1.5)


class SnapshotRgbSource:
    def __init__(self, cmd: list[str], width: int, height: int, timeout_s: float) -> None:
        self.cmd = cmd
        self.width = width
        self.height = height
        self.timeout_s = timeout_s

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            proc = subprocess.run(self.cmd, capture_output=True, timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return False, None
        if proc.returncode != 0 or not proc.stdout:
            return False, None
        try:
            img = Image.open(io.BytesIO(proc.stdout)).convert("RGB").resize((self.width, self.height), Image.Resampling.BILINEAR)
        except Exception:
            return False, None
        return True, np.array(img)

    def stop(self) -> None:
        return


class State:
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


def colorize_depth(depth: np.ndarray, near_mm: float, far_mm: float) -> np.ndarray:
    valid = (depth >= near_mm) & (depth <= far_mm)
    norm = np.clip((depth.astype(np.float32) - near_mm) / max(1.0, far_mm - near_mm), 0.0, 1.0)
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    rgb[..., 0] = ((1.0 - norm) * 255.0).astype(np.uint8)
    rgb[..., 2] = (norm * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def region_stats(depth: np.ndarray, rect: tuple[int, int, int, int], args: argparse.Namespace) -> dict[str, Any]:
    x0, y0, x1, y1 = rect
    vals = depth[y0:y1, x0:x1]
    vals = vals[(vals >= args.near_mm) & (vals <= args.far_mm)]
    if vals.size == 0:
        return {"valid_pixels": 0}
    return {
        "valid_pixels": int(vals.size),
        "min_m": float(vals.min() / 1000.0),
        "p05_m": float(np.percentile(vals, 5) / 1000.0),
        "p10_m": float(np.percentile(vals, 10) / 1000.0),
        "p35_m": float(np.percentile(vals, 35) / 1000.0),
        "p50_m": float(np.percentile(vals, 50) / 1000.0),
        "p80_m": float(np.percentile(vals, 80) / 1000.0),
    }


def local_depth_m(depth: np.ndarray, x: int, y: int, radius: int, args: argparse.Namespace) -> float | None:
    y0 = max(0, y - radius)
    y1 = min(depth.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(depth.shape[1], x + radius + 1)
    vals = depth[y0:y1, x0:x1]
    vals = vals[(vals >= args.near_mm) & (vals <= args.far_mm)]
    if vals.size == 0:
        return None
    return float(np.median(vals) / 1000.0)


def regions(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "center": (int(width * 0.34), int(height * 0.10), int(width * 0.66), int(height * 0.70)),
        "upper_center": (int(width * 0.34), int(height * 0.05), int(width * 0.66), int(height * 0.45)),
        "lower_center": (int(width * 0.34), int(height * 0.45), int(width * 0.66), int(height * 0.90)),
        "right_center": (int(width * 0.56), int(height * 0.10), int(width * 0.92), int(height * 0.70)),
        "left_center": (int(width * 0.08), int(height * 0.10), int(width * 0.44), int(height * 0.70)),
    }


def make_panel(rgb: np.ndarray | None, depth: np.ndarray, stats: dict[str, Any], fps: float, args: argparse.Namespace) -> bytes:
    view = Image.fromarray(colorize_depth(depth, args.near_mm, args.far_mm)).resize(
        (636, 360), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (1280, 720), (245, 247, 250))
    if rgb is None:
        rgb_view = Image.new("RGB", (640, 360), (20, 24, 30))
    else:
        rgb_view = Image.fromarray(rgb).resize((640, 360), Image.Resampling.BILINEAR)
    canvas.paste(rgb_view, (0, 58))
    canvas.paste(view, (644, 58))
    draw = ImageDraw.Draw(canvas)
    try:
        title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
        label = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    except Exception:
        title = label = small = None
    draw.text((18, 18), "G1 RGB view + measurement bounds", fill=(20, 28, 38), font=title)
    draw.text((662, 18), "RealSense depth: same bounds", fill=(20, 28, 38), font=title)
    colors = {
        "center": (255, 230, 0),
        "upper_center": (0, 210, 255),
        "lower_center": (255, 128, 0),
        "right_center": (0, 210, 90),
        "left_center": (210, 80, 255),
    }
    rgb_sx = 640 / args.depth_width
    rgb_sy = 360 / args.depth_height
    depth_sx = 636 / args.depth_width
    depth_sy = 360 / args.depth_height
    rgb_origin = (0, 58)
    depth_origin = (644, 58)

    def draw_axes(origin: tuple[int, int], pane_w: int, pane_h: int) -> None:
        ox, oy = origin
        cx = ox + pane_w // 2
        cy = oy + pane_h // 2
        draw.line((cx, oy, cx, oy + pane_h), fill=(255, 255, 255), width=1)
        draw.line((ox, cy, ox + pane_w, cy), fill=(255, 255, 255), width=1)
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=(255, 255, 255), width=2)

    draw_axes(rgb_origin, 640, 360)
    draw_axes(depth_origin, 636, 360)
    center_x = args.depth_width // 2
    center_y = args.depth_height // 2
    center_depth = local_depth_m(depth, center_x, center_y, args.point_radius, args)
    if center_depth is not None:
        text = f"axis {center_depth:.2f}m"
        for ox, oy, sx, sy in ((rgb_origin[0], rgb_origin[1], rgb_sx, rgb_sy), (depth_origin[0], depth_origin[1], depth_sx, depth_sy)):
            px = ox + center_x * sx
            py = oy + center_y * sy
            draw.rectangle((px + 8, py - 25, px + 118, py - 3), fill=(0, 0, 0))
            draw.text((px + 12, py - 24), text, fill=(255, 255, 255), font=small)

    for name, rect in regions(args.depth_width, args.depth_height).items():
        x0, y0, x1, y1 = rect
        color = colors[name]
        stat = stats["regions"].get(name, {})
        region_text = f"{name} near {stat.get('p05_m', 0.0):.2f}m med {stat.get('p50_m', 0.0):.2f}m"
        draw.rectangle((x0 * rgb_sx, 58 + y0 * rgb_sy, x1 * rgb_sx, 58 + y1 * rgb_sy), outline=color, width=3)
        draw.rectangle((4 + x0 * rgb_sx, 62 + y0 * rgb_sy, 255 + x0 * rgb_sx, 84 + y0 * rgb_sy), fill=(0, 0, 0))
        draw.text((8 + x0 * rgb_sx, 62 + y0 * rgb_sy), region_text, fill=color, font=small)
        draw.rectangle(
            (
                depth_origin[0] + x0 * depth_sx,
                depth_origin[1] + y0 * depth_sy,
                depth_origin[0] + x1 * depth_sx,
                depth_origin[1] + y1 * depth_sy,
            ),
            outline=color,
            width=3,
        )
        dx = depth_origin[0] + x0 * depth_sx
        dy = depth_origin[1] + y0 * depth_sy
        draw.rectangle((dx + 4, dy + 4, dx + 255, dy + 26), fill=(0, 0, 0))
        draw.text((dx + 8, dy + 4), region_text, fill=color, font=small)

    legend_x = 24
    legend_y = 436
    legend = [
        "Depth color: red=near, purple=mid, blue=far, black=invalid/no depth",
        f"Axis label is median depth in a {args.point_radius * 2 + 1}px square around the crosshair.",
        "Region labels show near=p05 and med=p50 for that box.",
    ]
    for line in legend:
        draw.text((legend_x, legend_y), line, fill=(35, 45, 60), font=label)
        legend_y += 26

    y = 565
    draw.text((24, y), "DRY RUN ONLY: RGB is visual context; distance numbers are raw RealSense depth regions", fill=(130, 35, 35), font=label)
    y += 30
    draw.text((24, y), f"fps={fps:.2f}  center p35={stats['regions']['center'].get('p35_m', 0.0):.2f}m p50={stats['regions']['center'].get('p50_m', 0.0):.2f}m", fill=(35, 45, 60), font=label)
    y += 30
    draw.text((24, y), f"right p35={stats['regions']['right_center'].get('p35_m', 0.0):.2f}m  lower p35={stats['regions']['lower_center'].get('p35_m', 0.0):.2f}m", fill=(35, 45, 60), font=label)
    import io

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def worker(shared: State, args: argparse.Namespace) -> None:
    depth_device = args.depth_device
    if depth_device == "auto":
        depth_device = discover_depth_device(args.robot_host)
        print(f"depth_device={depth_device}", flush=True)
    depth_source = RawDepthSource(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"unitree@{args.robot_host}",
            f"ffmpeg -hide_banner -loglevel error -f v4l2 -input_format gray16le -video_size {args.depth_width}x{args.depth_height} -framerate {args.depth_fps} -i {depth_device} -f rawvideo -pix_fmt gray16le -",
        ],
        args.depth_width,
        args.depth_height,
    )
    rgb_source: RawRgbSource | SnapshotRgbSource | None = None
    if args.rgb_source == "front-rtsp":
        rgb_source = SnapshotRgbSource(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "tcp",
                "-i",
                f"rtsp://{args.robot_host}:8554/front",
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-",
            ],
            args.rgb_width,
            args.rgb_height,
            args.rgb_snapshot_timeout,
        )
    elif args.rgb_source == "realsense-ir":
        rgb_source = RawRgbSource(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "StrictHostKeyChecking=accept-new",
                f"unitree@{args.robot_host}",
                (
                    "ffmpeg -hide_banner -loglevel error "
                    f"-f v4l2 -input_format uyvy422 -video_size 424x240 -framerate 15 -i {args.rgb_device} "
                    f"-vf scale={args.rgb_width}:{args.rgb_height},fps={args.rgb_fps} "
                    "-f rawvideo -pix_fmt rgb24 -"
                ),
            ],
            args.rgb_width,
            args.rgb_height,
        )
    rgb_lock = threading.Lock()
    latest_rgb: dict[str, np.ndarray | None] = {"frame": None}
    if rgb_source is not None:
        def read_rgb_loop() -> None:
            assert rgb_source is not None
            while not shared.stop:
                rgb_ok, rgb = rgb_source.read()
                if rgb_ok and rgb is not None:
                    with rgb_lock:
                        latest_rgb["frame"] = rgb
                    time.sleep(max(0.1, 1.0 / max(0.1, args.rgb_fps)))
                else:
                    time.sleep(0.1)

        threading.Thread(target=read_rgb_loop, daemon=True).start()
    frames = 0
    start = time.time()
    while not shared.stop:
        ok, depth = depth_source.read()
        if not ok or depth is None:
            time.sleep(0.2)
            continue
        frames += 1
        fps = frames / max(1e-6, time.time() - start)
        reg = {name: region_stats(depth, rect, args) for name, rect in regions(args.depth_width, args.depth_height).items()}
        axis_depth = local_depth_m(depth, args.depth_width // 2, args.depth_height // 2, args.point_radius, args)
        stats = {
            "time_unix": round(time.time(), 3),
            "dry_run_only": True,
            "frame": frames,
            "fps": fps,
            "depth_device": depth_device,
            "rgb_source": args.rgb_source,
            "axis_depth_m": axis_depth,
            "bounds": {name: list(rect) for name, rect in regions(args.depth_width, args.depth_height).items()},
            "bounds_note": "bounds are depth pixels [x0,y0,x1,y1], drawn at the same normalized positions on RGB",
            "regions": reg,
        }
        with rgb_lock:
            rgb_for_panel = None if latest_rgb["frame"] is None else latest_rgb["frame"].copy()
        shared.update(make_panel(rgb_for_panel, depth, stats, fps, args), stats)
    depth_source.stop()
    if rgb_source is not None:
        rgb_source.stop()


def discover_depth_device(robot_host: str) -> str:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"unitree@{robot_host}",
        "for d in /dev/video*; do ffmpeg -hide_banner -f v4l2 -list_formats all -i $d 2>&1 | grep -q gray16le && { echo $d; exit 0; }; done; exit 1",
    ]
    return subprocess.check_output(cmd, text=True).strip().splitlines()[0]


def handler_factory(shared: State, robot_host: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                html = """<!doctype html><html><head><title>G1 RGB + Depth</title>
<style>body{margin:0;background:#111;color:#eee;font-family:Arial,sans-serif}main{max-width:1280px;margin:0 auto;padding:14px}img{width:100%;height:auto;background:#222;border:0}pre{font-size:14px;white-space:pre-wrap}</style>
</head><body><main><h2>G1 RGB + Depth Dashboard</h2><img id="view" src="/snapshot.jpg"><pre id="state"></pre>
<script>
async function tick(){
  try {
    document.getElementById('view').src = '/snapshot.jpg?t=' + Date.now();
    let r = await fetch('/state?t=' + Date.now());
    document.getElementById('state').textContent = JSON.stringify(await r.json(), null, 2);
  } catch(e) {}
}
setInterval(tick, 350); tick();
</script></main></body></html>"""
                data = html.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/state":
                _, state = shared.snapshot()
                data = json.dumps(state, indent=2, sort_keys=True).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/snapshot.jpg":
                jpeg, _ = shared.snapshot()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-host", default="192.168.0.108")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--depth-device", default="auto")
    parser.add_argument("--depth-width", type=int, default=424)
    parser.add_argument("--depth-height", type=int, default=240)
    parser.add_argument("--depth-fps", type=int, default=15)
    parser.add_argument("--rgb-source", choices=("front-rtsp", "realsense-ir", "none"), default="front-rtsp")
    parser.add_argument("--rgb-device", default="/dev/video2")
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=360)
    parser.add_argument("--rgb-fps", type=int, default=4)
    parser.add_argument("--rgb-snapshot-timeout", type=float, default=3.5)
    parser.add_argument("--near-mm", type=float, default=300.0)
    parser.add_argument("--far-mm", type=float, default=4500.0)
    parser.add_argument("--point-radius", type=int, default=4)
    args = parser.parse_args()

    shared = State()
    thread = threading.Thread(target=worker, args=(shared, args), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.bind, args.port), handler_factory(shared, args.robot_host))
    print(f"DRY RUN ONLY: serving RGB + depth dashboard http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        shared.stop = True
        server.server_close()


if __name__ == "__main__":
    main()
