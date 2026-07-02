#!/usr/bin/env python3
"""Dry-run G1 camera/depth -> visual-command policy loop.

This script is intentionally non-moving. It does not import Unitree SDK modules
and does not publish robot commands. It only prints/logs proposed vx/vy/wz/stop
commands so the camera/depth/policy bridge can be validated on the robot.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np


def load_policy(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def render_observation(
    front: np.ndarray,
    lateral: np.ndarray,
    visible: np.ndarray,
    *,
    width: int,
    height: int,
    fov_deg: float,
) -> np.ndarray:
    n = front.shape[0]
    obs = np.zeros((n, 4, height, width), dtype=np.float32)
    obs[:, 0:3, :, :] = 0.92
    obs[:, 3, :, :] = 0.05
    horizon = int(height * 0.58)
    obs[:, 0:3, horizon:, :] = 0.35
    obs[:, 3, horizon:, :] = 0.18

    half_fov = math.radians(fov_deg / 2.0)
    bearing = np.arctan2(lateral, np.maximum(front, 1e-3))
    x_center = ((bearing / half_fov) * 0.5 + 0.5) * (width - 1)
    dist = np.sqrt(front * front + lateral * lateral)
    capsule_h = np.clip(30.0 / np.maximum(dist, 0.25), 5.0, height * 0.85)
    capsule_w = np.clip(capsule_h * 0.22, 2.0, width * 0.14)
    closeness = np.clip(1.0 - (dist - 0.35) / 3.0, 0.0, 1.0)

    yy, xx = np.mgrid[0:height, 0:width]
    for i in range(n):
        if not visible[i]:
            continue
        cx = x_center[i]
        if cx < -capsule_w[i] or cx > width + capsule_w[i]:
            continue
        body_top = height * 0.48 - capsule_h[i] * 0.55
        body_bot = height * 0.48 + capsule_h[i] * 0.42
        body = (np.abs(xx - cx) <= capsule_w[i] * 0.38) & (yy >= body_top) & (yy <= body_bot)
        head = (xx - cx) ** 2 + (yy - body_top) ** 2 <= (capsule_w[i] * 0.72) ** 2
        mask = body | head
        obs[i, 0, mask] = 0.04
        obs[i, 1, mask] = 0.25
        obs[i, 2, mask] = 1.0
        obs[i, 3, mask] = closeness[i]
    return obs


def features_from_observation(obs: np.ndarray) -> np.ndarray:
    rgb = obs[:, :3]
    depth = obs[:, 3]
    blue_mask = (rgb[:, 2] > 0.68) & (rgb[:, 2] > rgb[:, 0] + 0.20) & (rgb[:, 2] > rgb[:, 1] + 0.20)
    n, h, w = blue_mask.shape
    feats = np.zeros((n, 6), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    min_area = max(4, int(0.0015 * h * w))
    for i in range(n):
        mask = blue_mask[i]
        area = int(mask.sum())
        if area < min_area:
            continue
        xs = xx[mask]
        ys = yy[mask]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        feats[i, 0] = 1.0
        feats[i, 1] = ((float(xs.mean()) / max(1.0, w - 1.0)) - 0.5) * 2.0
        feats[i, 2] = (y1 - y0 + 1.0) / h
        feats[i, 3] = area / float(h * w)
        feats[i, 4] = float(depth[i][mask].mean())
        feats[i, 5] = ((float(ys.mean()) / max(1.0, h - 1.0)) - 0.5) * 2.0
    return feats


def forward(params: dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    x = features_from_observation(obs)
    pre = x @ params["w1"] + params["b1"]
    h = np.maximum(pre, 0.0)
    out = h @ params["w2"] + params["b2"]
    out[:, 0] = 0.35 / (1.0 + np.exp(-out[:, 0]))
    out[:, 1] = 0.18 * np.tanh(out[:, 1])
    out[:, 2] = 0.45 * np.tanh(out[:, 2])
    out[:, 3] = 1.0 / (1.0 + np.exp(-out[:, 3]))
    return out


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def normalize_depth_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((0, 0), dtype=np.uint16)
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    if frame.dtype == np.uint16:
        return frame
    if frame.dtype == np.uint8:
        return frame.astype(np.uint16) * 16
    return np.asarray(frame, dtype=np.uint16)


class FFmpegRGBCapture:
    def __init__(self, source: str, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if source.startswith("rtsp://"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += [
            "-i",
            source,
            "-vf",
            f"scale={width}:{height},fps={fps}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def is_opened(self) -> bool:
        return self.proc.poll() is None and self.proc.stdout is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.proc.stdout is None:
            return False, None
        data = self.proc.stdout.read(self.frame_bytes)
        if len(data) != self.frame_bytes:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
        return True, frame

    def release(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)


class FFmpegDepthCapture:
    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 2
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "gray16le",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            device,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def is_opened(self) -> bool:
        return self.proc.poll() is None and self.proc.stdout is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.proc.stdout is None:
            return False, None
        data = self.proc.stdout.read(self.frame_bytes)
        if len(data) != self.frame_bytes:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint16).reshape(self.height, self.width).copy()
        return True, frame

    def release(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)


def estimate_target_from_depth(depth_mm: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    if depth_mm.size == 0:
        return {"visible": False, "reason": "empty_depth"}
    h, w = depth_mm.shape
    x0 = int(round(w * args.roi_x0))
    x1 = int(round(w * args.roi_x1))
    y0 = int(round(h * args.roi_y0))
    y1 = int(round(h * args.roi_y1))
    roi = depth_mm[y0:y1, x0:x1].astype(np.float32)
    valid = (roi >= args.depth_near_mm) & (roi <= args.depth_far_mm)
    valid_count = int(valid.sum())
    if valid_count < args.min_valid_pixels:
        return {"visible": False, "reason": "too_few_valid_depth_pixels", "valid_pixels": valid_count}

    valid_depths = roi[valid]
    near_cut = float(np.percentile(valid_depths, args.near_percentile))
    mask = valid & (roi <= near_cut + args.near_slack_mm)
    target_pixels = int(mask.sum())
    if target_pixels < args.min_target_pixels:
        return {
            "visible": False,
            "reason": "too_few_target_pixels",
            "valid_pixels": valid_count,
            "target_pixels": target_pixels,
        }

    yy, xx = np.nonzero(mask)
    depths = roi[mask]
    weights = np.clip((args.depth_far_mm - depths) / max(1.0, args.depth_far_mm - args.depth_near_mm), 0.05, 1.0)
    cx = float(np.average(xx.astype(np.float32), weights=weights)) + x0
    cy = float(np.average(yy.astype(np.float32), weights=weights)) + y0
    depth_m = float(np.percentile(depths, 35) / 1000.0)
    bearing = ((cx / max(1.0, w - 1.0)) - 0.5) * math.radians(args.fov_deg)
    return {
        "visible": True,
        "reason": "near_depth_target",
        "valid_pixels": valid_count,
        "target_pixels": target_pixels,
        "pixel_x": cx,
        "pixel_y": cy,
        "depth_m": depth_m,
        "bearing_rad": bearing,
        "front_m": max(0.05, depth_m * math.cos(bearing)),
        "lateral_m": depth_m * math.sin(bearing),
    }


def command_from_estimate(params: dict[str, np.ndarray], estimate: dict[str, object], args: argparse.Namespace) -> np.ndarray:
    visible = bool(estimate.get("visible", False))
    front = float(estimate.get("front_m", 0.0))
    lateral = float(estimate.get("lateral_m", 0.0))
    obs = render_observation(
        np.array([front], dtype=np.float32),
        np.array([lateral], dtype=np.float32),
        np.array([visible]),
        width=args.policy_width,
        height=args.policy_height,
        fov_deg=args.fov_deg,
    )
    cmd = forward(params, obs)[0]
    if not visible:
        cmd[:] = [0.0, 0.0, 0.0, 1.0]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("policies/latest_policy.npz"))
    parser.add_argument("--rgb-source", default="rtsp://127.0.0.1:8554/front")
    parser.add_argument("--depth-device", default="/dev/video0")
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=360)
    parser.add_argument("--depth-width", type=int, default=424)
    parser.add_argument("--depth-height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--log", type=Path, default=Path("logs/dry_run_commands.jsonl"))
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--policy-width", type=int, default=64)
    parser.add_argument("--policy-height", type=int, default=48)
    parser.add_argument("--fov-deg", type=float, default=69.0)
    parser.add_argument("--depth-near-mm", type=float, default=300.0)
    parser.add_argument("--depth-far-mm", type=float, default=4500.0)
    parser.add_argument("--roi-x0", type=float, default=0.15)
    parser.add_argument("--roi-x1", type=float, default=0.85)
    parser.add_argument("--roi-y0", type=float, default=0.18)
    parser.add_argument("--roi-y1", type=float, default=0.96)
    parser.add_argument("--near-percentile", type=float, default=12.0)
    parser.add_argument("--near-slack-mm", type=float, default=240.0)
    parser.add_argument("--min-valid-pixels", type=int, default=250)
    parser.add_argument("--min-target-pixels", type=int, default=45)
    parser.add_argument("--enable-motion", action="store_true", help="Refused in this dry-run script.")
    args = parser.parse_args()

    if args.enable_motion:
        raise SystemExit("--enable-motion is intentionally not implemented in this dry-run script")

    params = load_policy(args.policy)
    rgb_cap = FFmpegRGBCapture(args.rgb_source, args.rgb_width, args.rgb_height, args.fps)
    depth_cap = FFmpegDepthCapture(args.depth_device, args.depth_width, args.depth_height, args.fps)
    if not rgb_cap.is_opened():
        raise SystemExit(f"failed to open RGB source {args.rgb_source}")
    if not depth_cap.is_opened():
        raise SystemExit(f"failed to open depth camera {args.depth_device}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    print("DRY RUN ONLY: no Unitree SDK import, no command publisher, no robot motion")
    print(f"policy={args.policy} rgb={args.rgb_source} depth={args.depth_device} log={args.log}")
    start = time.time()
    frame = 0
    try:
        while args.seconds <= 0.0 or time.time() - start < args.seconds:
            rgb_ok, rgb = rgb_cap.read()
            depth_ok, depth_raw = depth_cap.read()
            now = time.time()
            depth_mm = normalize_depth_frame(depth_raw if depth_ok else None)
            estimate = estimate_target_from_depth(depth_mm, args)
            command = command_from_estimate(params, estimate, args)
            record = {
                "time_unix": round(now, 3),
                "frame": frame,
                "rgb_ok": bool(rgb_ok),
                "depth_ok": bool(depth_ok),
                "rgb_shape": list(rgb.shape) if rgb_ok and rgb is not None else None,
                "depth_shape": list(depth_mm.shape),
                "depth_dtype": str(depth_mm.dtype),
                "target_estimate": estimate,
                "command_vx_vy_wz_stop": [round(float(x), 6) for x in command],
            }
            append_jsonl(args.log, record)
            if frame % max(1, args.print_every) == 0:
                print(
                    "frame={frame} visible={visible} front={front:.2f} lateral={lat:+.2f} "
                    "vx={vx:+.3f} vy={vy:+.3f} wz={wz:+.3f} stop={stop:.3f} reason={reason}".format(
                        frame=frame,
                        visible=bool(estimate.get("visible", False)),
                        front=float(estimate.get("front_m", 0.0)),
                        lat=float(estimate.get("lateral_m", 0.0)),
                        vx=float(command[0]),
                        vy=float(command[1]),
                        wz=float(command[2]),
                        stop=float(command[3]),
                        reason=str(estimate.get("reason", "")),
                    ),
                    flush=True,
                )
            frame += 1
    finally:
        rgb_cap.release()
        depth_cap.release()
    elapsed = max(1e-6, time.time() - start)
    print(f"done frames={frame} fps={frame / elapsed:.2f} log={args.log}")


if __name__ == "__main__":
    main()
