#!/usr/bin/env python3
"""Local browser dashboard for G1 follow perception and browser arming.

Runs on the Mac. It reads G1 RGB/depth over SSH or the front RTSP stream, runs
YOLO person detection plus the visual-command policy, and serves a live browser
view. It does not publish Unitree motion commands; a separate robot-side
controller may poll /target.json and honor the browser GO/STOP state.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from ultralytics import FastSAM, YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "artifacts" / "visual-command" / "2026-06-23" / "moving-random-1091043328" / "latest_policy.npz"
DEFAULT_MODEL = Path("/Users/smile/WendyOS/hat/models/yolo11n.pt")
DEFAULT_SEGMENTER_MODEL = Path("/Users/smile/WendyOS/hat/models/FastSAM-s.pt")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
SSH = shutil.which("ssh") or "/usr/bin/ssh"


def load_policy(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def render_observation(front: np.ndarray, lateral: np.ndarray, visible: np.ndarray, *, width: int, height: int, fov_deg: float) -> np.ndarray:
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


class RawFrameSource:
    def __init__(self, cmd: list[str], width: int, height: int, channels: int, dtype: np.dtype[Any], shape: tuple[int, ...]) -> None:
        self.cmd = cmd
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        self.shape = shape
        self.proc: subprocess.Popen[bytes] | None = None
        self.frame_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

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
        return True, np.frombuffer(data, dtype=self.dtype).reshape(self.shape).copy()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=1.5)


class SnapshotFrameSource:
    def __init__(self, cmd: list[str], dtype: np.dtype[Any], shape: tuple[int, ...], timeout_s: float) -> None:
        self.cmd = cmd
        self.dtype = dtype
        self.shape = shape
        self.timeout_s = timeout_s
        self.frame_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            proc = subprocess.run(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return False, None
        if proc.returncode != 0 or len(proc.stdout) != self.frame_bytes:
            return False, None
        return True, np.frombuffer(proc.stdout, dtype=self.dtype).reshape(self.shape).copy()

    def stop(self) -> None:
        return


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm > 0
    near = float(np.percentile(depth_mm[valid], 2)) if valid.any() else 350.0
    far = float(np.percentile(depth_mm[valid], 98)) if valid.any() else 3600.0
    norm = np.clip((depth_mm.astype(np.float32) - near) / max(1.0, far - near), 0.0, 1.0)
    out = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    out[..., 0] = ((1.0 - norm) * 255.0).astype(np.uint8)
    out[..., 2] = (norm * 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def depth_region_summary(depth_mm: np.ndarray, rect: tuple[int, int, int, int], args: argparse.Namespace) -> dict[str, Any]:
    h, w = depth_mm.shape
    x0, y0, x1, y1 = rect
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, y0 + 1, h))
    vals = depth_mm[y0:y1, x0:x1]
    valid = vals[(vals >= args.depth_near_mm) & (vals <= args.depth_far_mm)]
    payload: dict[str, Any] = {
        "rect_xyxy": [x0, y0, x1, y1],
        "total_pixels": int(vals.size),
        "valid_pixels": int(valid.size),
        "units": "millimeters",
    }
    if valid.size == 0:
        return payload
    for pct in (5, 10, 25, 35, 50, 80, 95):
        payload[f"p{pct:02d}_mm"] = float(np.percentile(valid, pct))
    payload["min_mm"] = int(valid.min())
    payload["max_mm"] = int(valid.max())
    payload["mean_mm"] = float(valid.mean())
    return payload


def estimate_closest_scene_object(depth_mm: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    h, w = depth_mm.shape
    x0 = int(np.clip(round(w * args.scene_roi_x0), 0, w - 1))
    x1 = int(np.clip(round(w * args.scene_roi_x1), x0 + 1, w))
    y0 = int(np.clip(round(h * args.scene_roi_y0), 0, h - 1))
    y1 = int(np.clip(round(h * args.scene_roi_y1), y0 + 1, h))
    valid_mask = np.zeros_like(depth_mm, dtype=bool)
    roi = depth_mm[y0:y1, x0:x1]
    valid_roi = (roi >= args.person_depth_near_mm) & (roi <= args.depth_far_mm)
    valid_mask[y0:y1, x0:x1] = valid_roi
    valid = depth_mm[valid_mask]
    if valid.size == 0:
        return {"visible": False, "reason": "no_valid_depth", "scene_roi": [x0, y0, x1, y1]}
    nearest_mm = int(valid.min())
    candidate_mask = valid_mask & (depth_mm <= nearest_mm + args.scene_object_slack_mm)
    component, component_info = largest_component(candidate_mask)
    component_pixels = int(component.sum())
    if component_pixels < args.min_scene_object_pixels:
        return {
            "visible": False,
            "reason": "closest_depth_not_connected_object",
            "nearest_depth_mm": nearest_mm,
            "component_pixels": component_pixels,
            "min_component_pixels": args.min_scene_object_pixels,
            "scene_roi": [x0, y0, x1, y1],
        }
    y, x = np.unravel_index(np.argmin(np.where(component, depth_mm, np.iinfo(depth_mm.dtype).max)), depth_mm.shape)
    depth_mm_value = int(depth_mm[y, x])
    bearing = ((float(x) / max(1.0, depth_mm.shape[1] - 1.0)) - 0.5) * math.radians(args.fov_deg)
    depth_m = depth_mm_value / 1000.0
    return {
        "visible": True,
        "reason": "closest_connected_depth_object",
        "depth_m": depth_m,
        "depth_mm": depth_mm_value,
        "pixel_x": int(x),
        "pixel_y": int(y),
        "front_m": max(0.05, depth_m * math.cos(bearing)),
        "lateral_m": depth_m * math.sin(bearing),
        "valid_pixels": int(valid.size),
        "component_pixels": component_pixels,
        "component_box": component_info.get("component_box"),
        "scene_roi": [x0, y0, x1, y1],
    }


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    visited = np.zeros(mask.shape, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    ys, xs = np.nonzero(mask)
    for seed_y, seed_x in zip(ys.tolist(), xs.tolist()):
        if visited[seed_y, seed_x]:
            continue
        stack = [(seed_y, seed_x)]
        visited[seed_y, seed_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) > len(best_pixels):
            best_pixels = pixels
    component = np.zeros(mask.shape, dtype=bool)
    if not best_pixels:
        return component, {"component_pixels": 0}
    py = np.array([p[0] for p in best_pixels], dtype=np.int32)
    px = np.array([p[1] for p in best_pixels], dtype=np.int32)
    component[py, px] = True
    return component, {
        "component_pixels": int(len(best_pixels)),
        "component_box": [int(px.min()), int(py.min()), int(px.max()) + 1, int(py.max()) + 1],
    }


def estimate_from_depth_target(depth_mm: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if depth_mm.size == 0:
        return {"visible": False, "reason": "empty_depth"}
    h, w = depth_mm.shape
    x0 = int(round(w * args.depth_roi_x0))
    x1 = int(round(w * args.depth_roi_x1))
    y0 = int(round(h * args.depth_roi_y0))
    y1 = int(round(h * args.depth_roi_y1))
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, y0 + 1, h))
    roi = depth_mm[y0:y1, x0:x1].astype(np.float32)
    valid = (roi >= args.depth_near_mm) & (roi <= args.depth_far_mm)
    valid_count = int(valid.sum())
    if valid_count < args.min_valid_depth_pixels:
        return {
            "visible": False,
            "reason": "too_few_valid_depth_pixels",
            "valid_pixels": valid_count,
            "depth_roi": [x0, y0, x1, y1],
        }

    valid_depths = roi[valid]
    near_cut = float(np.percentile(valid_depths, args.depth_target_near_percentile))
    near_mask = valid & (roi <= near_cut + args.depth_target_slack_mm)
    near_pixels = int(near_mask.sum())
    if near_pixels < args.min_depth_target_pixels:
        return {
            "visible": False,
            "reason": "too_few_near_depth_pixels",
            "valid_pixels": valid_count,
            "near_pixels": near_pixels,
            "near_cut_mm": near_cut,
            "depth_roi": [x0, y0, x1, y1],
        }

    blob_mask, blob_info = largest_component(near_mask)
    blob_pixels = int(blob_mask.sum())
    if blob_pixels < args.min_depth_target_pixels:
        return {
            "visible": False,
            "reason": "too_few_connected_depth_pixels",
            "valid_pixels": valid_count,
            "near_pixels": near_pixels,
            "target_pixels": blob_pixels,
            "near_cut_mm": near_cut,
            "depth_roi": [x0, y0, x1, y1],
        }

    yy, xx = np.nonzero(blob_mask)
    depths = roi[blob_mask]
    depth_m = float(np.percentile(depths, args.depth_target_percentile) / 1000.0)
    weights = np.clip((args.depth_far_mm - depths) / max(1.0, args.depth_far_mm - args.depth_near_mm), 0.05, 1.0)
    cx = float(np.average(xx.astype(np.float32), weights=weights)) + x0
    cy = float(np.average(yy.astype(np.float32), weights=weights)) + y0
    bearing = ((cx / max(1.0, w - 1.0)) - 0.5) * math.radians(args.fov_deg)
    local_box = blob_info.get("component_box", [0, 0, 0, 0])
    box = [int(local_box[0]) + x0, int(local_box[1]) + y0, int(local_box[2]) + x0, int(local_box[3]) + y0]
    return {
        "visible": True,
        "reason": "raw_depth_nearest_connected_target",
        "detector": "raw_depth",
        "trusted_depth_m": depth_m,
        "depth_m": depth_m,
        "depth_p10_m": float(np.percentile(depths, 10) / 1000.0),
        "depth_p35_m": float(np.percentile(depths, 35) / 1000.0),
        "depth_p50_m": float(np.percentile(depths, 50) / 1000.0),
        "front_m": max(0.05, depth_m * math.cos(bearing)),
        "lateral_m": depth_m * math.sin(bearing),
        "bearing_rad": bearing,
        "pixel_x": cx,
        "pixel_y": cy,
        "valid_pixels": valid_count,
        "near_pixels": near_pixels,
        "target_pixels": blob_pixels,
        "near_cut_mm": near_cut,
        "depth_roi": [x0, y0, x1, y1],
        "depth_box": box,
    }


def mask_to_depth(mask: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    depth_w = depth_shape[1]
    depth_h = depth_shape[0]
    return np.array(mask_img.resize((depth_w, depth_h), Image.Resampling.NEAREST)) > 0


def select_trace_mask(
    segment_result: Any,
    box: list[float],
    rgb_w: int,
    rgb_h: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if segment_result.masks is None or segment_result.masks.data is None:
        return None, {"trace_reason": "no_segment_masks"}
    masks = segment_result.masks.data.cpu().numpy() > 0.5
    if masks.size == 0:
        return None, {"trace_reason": "empty_segment_masks"}
    x0, y0, x1, y1 = [float(v) for v in box]
    best_mask: np.ndarray | None = None
    best_score = -1.0
    best_info: dict[str, Any] = {"trace_reason": "no_overlap"}
    for idx, mask in enumerate(masks):
        if mask.shape != (rgb_h, rgb_w):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((rgb_w, rgb_h), Image.Resampling.NEAREST)
            ) > 0
        bx0 = int(np.clip(x0, 0, rgb_w - 1))
        bx1 = int(np.clip(x1, bx0 + 1, rgb_w))
        by0 = int(np.clip(y0, 0, rgb_h - 1))
        by1 = int(np.clip(y1, by0 + 1, rgb_h))
        box_mask = np.zeros((rgb_h, rgb_w), dtype=bool)
        box_mask[by0:by1, bx0:bx1] = True
        inter = int((mask & box_mask).sum())
        mask_area = int(mask.sum())
        box_area = int(box_mask.sum())
        if mask_area <= 0 or box_area <= 0:
            continue
        box_coverage = inter / box_area
        mask_inside_box = inter / mask_area
        score = box_coverage * min(1.0, mask_inside_box / max(1e-6, args.trace_min_mask_inside_box))
        if score > best_score:
            best_score = score
            best_mask = mask
            best_info = {
                "trace_reason": "fastsam_overlap",
                "trace_index": int(idx),
                "trace_box_coverage": float(box_coverage),
                "trace_mask_inside_box": float(mask_inside_box),
                "trace_mask_pixels_rgb": mask_area,
            }
    if best_mask is None:
        return None, best_info
    if (
        float(best_info["trace_box_coverage"]) < args.trace_min_box_coverage
        or float(best_info["trace_mask_inside_box"]) < args.trace_min_mask_inside_box
    ):
        best_info["trace_reason"] = "trace_overlap_below_threshold"
        return None, best_info
    return best_mask, best_info


def estimate_from_person(
    depth_mm: np.ndarray,
    box: list[float] | None,
    mask: np.ndarray | None,
    rgb_w: int,
    rgb_h: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if box is None:
        return {"visible": False, "reason": "no_person_detection"}
    h, w = depth_mm.shape
    x0, y0, x1, y1 = box
    cx_rgb = 0.5 * (x0 + x1)
    dx0 = int(np.clip(x0 / rgb_w * w, 0, w - 1))
    dx1 = int(np.clip(x1 / rgb_w * w, dx0 + 1, w))
    dy0 = int(np.clip(y0 / rgb_h * h, 0, h - 1))
    dy1 = int(np.clip(y1 / rgb_h * h, dy0 + 1, h))
    crop = depth_mm[dy0:dy1, dx0:dx1].astype(np.float32)
    reason = "yolo_person_box_closest_depth"
    mask_pixels = int(mask_to_depth(mask, depth_mm.shape).sum()) if mask is not None else 0
    valid_mask = (crop >= args.person_depth_near_mm) & (crop <= args.depth_far_mm)
    valid_depths = crop[valid_mask]
    if valid_depths.size < args.min_person_depth_pixels:
        return {
            "visible": False,
            "reason": "too_few_person_depth_pixels",
            "valid_pixels": int(valid_depths.size),
            "mask_pixels": mask_pixels,
        }
    closest_y, closest_x = np.unravel_index(np.argmin(np.where(valid_mask, crop, np.inf)), crop.shape)
    closest_depth_x = int(dx0 + closest_x)
    closest_depth_y = int(dy0 + closest_y)
    closest_mm = float(crop[closest_y, closest_x])
    depth_m = closest_mm / 1000.0
    cx_depth = cx_rgb / max(1.0, rgb_w - 1.0) * (w - 1.0)
    bearing = ((cx_depth / max(1.0, w - 1.0)) - 0.5) * math.radians(args.fov_deg)
    return {
        "visible": True,
        "reason": reason,
        "valid_pixels": int(valid_depths.size),
        "mask_pixels": mask_pixels,
        "closest_depth_m": depth_m,
        "closest_depth_mm": closest_mm,
        "closest_pixel_x": closest_depth_x,
        "closest_pixel_y": closest_depth_y,
        "depth_m": depth_m,
        "depth_min_m": depth_m,
        "depth_p10_m": float(np.percentile(valid_depths, 10) / 1000.0),
        "depth_p35_m": float(np.percentile(valid_depths, 35) / 1000.0),
        "depth_p50_m": float(np.percentile(valid_depths, 50) / 1000.0),
        "front_m": max(0.05, depth_m * math.cos(bearing)),
        "lateral_m": depth_m * math.sin(bearing),
        "bearing_rad": bearing,
        "pixel_x": cx_depth,
        "pixel_y": 0.5 * (dy0 + dy1),
        "depth_box": [dx0, dy0, dx1, dy1],
    }


def command_from_estimate(params: dict[str, np.ndarray], estimate: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    visible = bool(estimate.get("visible", False))
    obs = render_observation(
        np.array([float(estimate.get("front_m", 0.0))], dtype=np.float32),
        np.array([float(estimate.get("lateral_m", 0.0))], dtype=np.float32),
        np.array([visible]),
        width=args.policy_width,
        height=args.policy_height,
        fov_deg=args.fov_deg,
    )
    cmd = forward(params, obs)[0]
    if not visible:
        cmd[:] = [0.0, 0.0, 0.0, 1.0]
    return cmd


def safety_from_scene_closest(scene_closest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    depth_m = scene_closest.get("depth_m")
    active = bool(scene_closest.get("visible")) and depth_m is not None and float(depth_m) < args.safety_stop_m
    return {
        "active": active,
        "reason": "closest_object_too_near" if active else "clear",
        "stop_distance_m": args.safety_stop_m,
        "closest_depth_m": float(depth_m) if depth_m is not None else None,
        "closest_pixel_x": scene_closest.get("pixel_x"),
        "closest_pixel_y": scene_closest.get("pixel_y"),
    }


def apply_safety_gate(command: np.ndarray, safety: dict[str, Any]) -> np.ndarray:
    if not safety.get("active"):
        return command
    safe = command.copy()
    safe[:] = [0.0, 0.0, 0.0, 1.0]
    return safe


def choose_policy_estimate(
    rgb_estimate: dict[str, Any],
    depth_estimate: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    if args.policy_target_source == "rgb-person":
        return rgb_estimate, "rgb-person"
    if args.policy_target_source == "depth-nearest":
        return depth_estimate, "depth-nearest"
    if rgb_estimate.get("visible"):
        return rgb_estimate, "auto:rgb-person"
    return {"visible": False, "reason": "no_person_detection"}, "auto:no-person"


def draw_cross(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int]) -> None:
    draw.ellipse((x - 13, y - 13, x + 13, y + 13), outline=color, width=4)
    draw.line((x - 19, y, x + 19, y), fill=color, width=2)
    draw.line((x, y - 19, x, y + 19), fill=color, width=2)


class DashboardState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg = b""
        self.state: dict[str, Any] = {"status": "starting"}
        self.depth_mm: np.ndarray | None = None
        self.motion_enabled = False
        self.motion_updated_at = time.time()
        self.stop = False

    def update(self, jpeg: bytes, state: dict[str, Any], depth_mm: np.ndarray | None = None) -> None:
        with self.lock:
            state = dict(state)
            state["motion_control"] = self._motion_payload_locked()
            self.jpeg = jpeg
            self.state = state
            if depth_mm is not None:
                self.depth_mm = depth_mm.copy()

    def snapshot(self) -> tuple[bytes, dict[str, Any]]:
        with self.lock:
            return self.jpeg, dict(self.state)

    def depth_snapshot(self) -> np.ndarray | None:
        with self.lock:
            return None if self.depth_mm is None else self.depth_mm.copy()

    def motion_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._motion_payload_locked()

    def _motion_payload_locked(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.motion_enabled),
            "updated_at_unix": round(self.motion_updated_at, 3),
        }

    def set_motion_enabled(self, enabled: bool) -> dict[str, Any]:
        with self.lock:
            self.motion_enabled = bool(enabled)
            self.motion_updated_at = time.time()
            self.state = dict(self.state)
            self.state["motion_control"] = self._motion_payload_locked()
            return self.state["motion_control"]

    def toggle_motion_enabled(self) -> dict[str, Any]:
        with self.lock:
            self.motion_enabled = not self.motion_enabled
            self.motion_updated_at = time.time()
            self.state = dict(self.state)
            self.state["motion_control"] = self._motion_payload_locked()
            return self.state["motion_control"]


def overlay_mask(image: Image.Image, mask: np.ndarray | None, color: tuple[int, int, int], alpha: int) -> Image.Image:
    if mask is None:
        return image
    mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", image.size, (*color, alpha))
    out = image.convert("RGBA")
    out.paste(overlay, (0, 0), mask_img)
    edges = mask_img.filter(ImageFilter.FIND_EDGES)
    outline = Image.new("RGBA", image.size, (*color, 255))
    out.paste(outline, (0, 0), edges)
    return out.convert("RGB")


def make_panel(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    box: list[float] | None,
    mask: np.ndarray | None,
    rgb_estimate: dict[str, Any],
    depth_estimate: dict[str, Any],
    scene_closest: dict[str, Any],
    policy_estimate: dict[str, Any],
    policy_source: str,
    command: np.ndarray,
    raw_command: np.ndarray,
    safety: dict[str, Any],
    fps: float,
    args: argparse.Namespace,
) -> bytes:
    canvas = Image.new("RGB", (1280, 720), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    try:
        title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 27)
        label = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except Exception:
        title = label = small = None
    rgb_img = Image.fromarray(rgb).resize((640, 360), Image.Resampling.BILINEAR)
    rgb_img = overlay_mask(rgb_img, mask, (255, 230, 0), 70)
    depth_img = Image.fromarray(colorize_depth(depth_mm)).resize((636, 360), Image.Resampling.NEAREST)
    if mask is not None:
        depth_mask = mask_to_depth(mask, depth_mm.shape)
        depth_img = overlay_mask(depth_img, depth_mask.astype(np.float32), (255, 230, 0), 55)
    canvas.paste(rgb_img, (0, 58))
    canvas.paste(depth_img, (644, 58))
    draw.text((18, 18), f"G1 {args.rgb_source_label} RGB + YOLO", fill=(20, 28, 38), font=title)
    draw.text((662, 18), "G1 depth: red closer, blue farther", fill=(20, 28, 38), font=title)

    if box is not None:
        sx = 640 / args.rgb_width
        sy = 360 / args.rgb_height
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((int(x0 * sx), 58 + int(y0 * sy), int(x1 * sx), 58 + int(y1 * sy)), outline=(255, 230, 0), width=4)
    if policy_source == "depth-nearest" and depth_estimate.get("visible") and "depth_box" in depth_estimate:
        dx0, dy0, dx1, dy1 = [float(v) for v in depth_estimate["depth_box"]]
        draw.rectangle(
            (
                644 + int(dx0 / args.depth_width * 636),
                58 + int(dy0 / args.depth_height * 360),
                644 + int(dx1 / args.depth_width * 636),
                58 + int(dy1 / args.depth_height * 360),
            ),
            outline=(0, 210, 255),
            width=3,
        )
    if rgb_estimate.get("visible"):
        dx = float(rgb_estimate["pixel_x"]) / args.depth_width
        dy = float(rgb_estimate["pixel_y"]) / args.depth_height
        draw_cross(draw, int(644 + dx * 636), int(58 + dy * 360), (255, 230, 0))
        draw_cross(draw, int(dx * 640), int(58 + dy * 360), (255, 230, 0))
        if "closest_pixel_x" in rgb_estimate and "closest_pixel_y" in rgb_estimate:
            cx = float(rgb_estimate["closest_pixel_x"]) / args.depth_width
            cy = float(rgb_estimate["closest_pixel_y"]) / args.depth_height
            px = int(644 + cx * 636)
            py = int(58 + cy * 360)
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(255, 255, 255), outline=(20, 28, 38), width=2)
    if scene_closest.get("visible"):
        cx = float(scene_closest["pixel_x"]) / args.depth_width
        cy = float(scene_closest["pixel_y"]) / args.depth_height
        px = int(644 + cx * 636)
        py = int(58 + cy * 360)
        draw.rectangle((px - 6, py - 6, px + 6, py + 6), fill=(0, 255, 120), outline=(20, 28, 38), width=2)
    lines = [
        "BROWSER MOTION CONTROL: " + (
            ("GO enabled - robot bridge may move" if args.browser_motion_control_enabled else "STOP enabled")
            if args.allow_browser_motion_control
            else "disabled - no robot bridge should move"
        ),
        f"policy source={policy_source} visible={bool(policy_estimate.get('visible', False))} reason={policy_estimate.get('reason', '')} fps={fps:.2f}",
        f"closest depth inside human box={float(rgb_estimate.get('closest_depth_m', 0.0)):.2f}m pixels={int(rgb_estimate.get('valid_pixels', 0))}",
        f"closest object anywhere={float(scene_closest.get('depth_m', 0.0)):.2f}m at ({int(scene_closest.get('pixel_x', 0))},{int(scene_closest.get('pixel_y', 0))})",
        f"safety stop={bool(safety.get('active', False))} threshold={float(safety.get('stop_distance_m', 0.0)):.2f}m reason={safety.get('reason', '')}",
        f"policy target front={float(policy_estimate.get('front_m', 0.0)):.2f}m lateral={float(policy_estimate.get('lateral_m', 0.0)):+.2f}m depth={float(policy_estimate.get('depth_m', 0.0)):.2f}m",
        f"rgb person visible={bool(rgb_estimate.get('visible', False))} conf={float(rgb_estimate.get('person_confidence', 0.0)):.2f} reason={rgb_estimate.get('reason', '')}",
        f"wanted input: vx={command[0]:+.3f}  vy={command[1]:+.3f}  wz={command[2]:+.3f}  stop={command[3]:.3f}",
    ]
    y = 454
    for i, line in enumerate(lines):
        fill = (130, 35, 35) if i == 0 and args.browser_motion_control_enabled else (35, 45, 60)
        draw.text((26, y), line, fill=fill, font=label)
        y += 36
    import io

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=86)
    return buf.getvalue()


def worker(shared: DashboardState, args: argparse.Namespace) -> None:
    params = load_policy(args.policy)
    detector = YOLO(str(args.model))
    segmenter = FastSAM(str(args.segmenter_model)) if args.trace_mode == "fastsam" else None
    depth_device = args.depth_device
    if depth_device == "auto":
        depth_device = discover_depth_device(args.robot_host)
        print(f"depth_device={depth_device}", flush=True)
    if args.rgb_source == "realsense":
        rgb_device = args.rgb_device
        if rgb_device == "auto":
            rgb_device = discover_rgb_device(args.robot_host)
            print(f"rgb_device={rgb_device}", flush=True)
        args.rgb_source_label = "RealSense"
        rgb_cmd = [
            SSH,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"unitree@{args.robot_host}",
            (
                "ffmpeg -hide_banner -loglevel error "
                f"-f v4l2 -input_format {args.rgb_input_format} "
                f"-video_size {args.rgb_capture_width}x{args.rgb_capture_height} "
                f"-framerate {args.rgb_fps} -i {rgb_device} "
                + ("-frames:v 1 " if args.rgb_snapshot else "")
                + f"-vf scale={args.rgb_width}:{args.rgb_height}"
                + ("" if args.rgb_snapshot else f",fps={args.fps}")
                + " -f rawvideo -pix_fmt rgb24 -"
            ),
        ]
        if args.rgb_snapshot:
            rgb_source = SnapshotFrameSource(
                rgb_cmd,
                np.uint8,
                (args.rgb_height, args.rgb_width, 3),
                args.rgb_snapshot_timeout,
            )
        else:
            rgb_source = RawFrameSource(
                rgb_cmd,
                args.rgb_width,
                args.rgb_height,
                3,
                np.uint8,
                (args.rgb_height, args.rgb_width, 3),
            )
    else:
        args.rgb_source_label = "front"
        rgb_cmd = [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            f"rtsp://{args.robot_host}:8554/front",
            "-vf",
            f"scale={args.rgb_width}:{args.rgb_height},fps={args.fps}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        rgb_source = RawFrameSource(
            rgb_cmd,
            args.rgb_width,
            args.rgb_height,
            3,
            np.uint8,
            (args.rgb_height, args.rgb_width, 3),
        )
    depth_source = SnapshotFrameSource(
        [
            SSH,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"unitree@{args.robot_host}",
            f"ffmpeg -hide_banner -loglevel error -f v4l2 -input_format gray16le -video_size {args.depth_width}x{args.depth_height} -framerate {args.depth_fps} -i {depth_device} -frames:v 1 -f rawvideo -pix_fmt gray16le -",
        ],
        np.uint16,
        (args.depth_height, args.depth_width),
        args.depth_snapshot_timeout,
    )
    frames = 0
    start = time.time()
    while not shared.stop:
        rgb_ok, rgb = rgb_source.read()
        depth_ok, depth = depth_source.read()
        if not rgb_ok or rgb is None or not depth_ok or depth is None:
            time.sleep(0.2)
            continue
        frames += 1
        elapsed = max(1e-6, time.time() - start)
        best_box: list[float] | None = None
        best_mask: np.ndarray | None = None
        trace_info: dict[str, Any] = {"trace_reason": "trace_disabled" if segmenter is None else "no_person_box"}
        best_conf = 0.0
        results = detector.predict(rgb, conf=args.conf, imgsz=args.imgsz, classes=[0], verbose=False)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = []
            for idx, box in enumerate(results[0].boxes):
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                boxes.append((conf, area, idx, xyxy))
            boxes.sort(reverse=True)
            best_conf, _, best_idx, best_box = boxes[0]
            if segmenter is not None and frames % args.segment_every == 0:
                segment_results = segmenter.predict(
                    rgb,
                    conf=args.segment_conf,
                    imgsz=args.segment_imgsz,
                    retina_masks=True,
                    verbose=False,
                )
                if segment_results:
                    best_mask, trace_info = select_trace_mask(
                        segment_results[0], best_box, args.rgb_width, args.rgb_height, args
                    )
                else:
                    trace_info = {"trace_reason": "no_segment_result"}
        rgb_estimate = estimate_from_person(depth, best_box, best_mask, args.rgb_width, args.rgb_height, args)
        rgb_estimate.update(trace_info)
        rgb_estimate["person_confidence"] = best_conf
        depth_estimate = estimate_from_depth_target(depth, args)
        scene_closest = estimate_closest_scene_object(depth, args)
        policy_estimate, policy_source = choose_policy_estimate(rgb_estimate, depth_estimate, args)
        raw_command = command_from_estimate(params, policy_estimate, args)
        safety = safety_from_scene_closest(scene_closest, args)
        command = apply_safety_gate(raw_command, safety)
        args.browser_motion_control_enabled = bool(shared.motion_snapshot()["enabled"])
        fps = frames / elapsed
        jpeg = make_panel(
            rgb,
            depth,
            best_box,
            best_mask,
            rgb_estimate,
            depth_estimate,
            scene_closest,
            policy_estimate,
            policy_source,
            command,
            raw_command,
            safety,
            fps,
            args,
        )
        shared.update(
            jpeg,
            {
                "time_unix": round(time.time(), 3),
                "dry_run_only": True,
                "frame": frames,
                "fps": fps,
                "person_confidence": best_conf,
                "segmentation_mask_used": best_mask is not None,
                "trace_info": trace_info,
                "policy_target_source": policy_source,
                "target_estimate": policy_estimate,
                "rgb_person_estimate": rgb_estimate,
                "depth_target_estimate": depth_estimate,
                "closest_scene_object": scene_closest,
                "safety_stop": safety,
                "raw_policy_vx_vy_wz_stop": [float(x) for x in raw_command],
                "wanted_input_vx_vy_wz_stop": [float(x) for x in command],
            },
            depth,
        )
    rgb_source.stop()
    depth_source.stop()


def discover_depth_device(robot_host: str) -> str:
    cmd = [
        SSH,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"unitree@{robot_host}",
        "for d in /dev/video*; do ffmpeg -hide_banner -f v4l2 -list_formats all -i $d 2>&1 | grep -q gray16le && { echo $d; exit 0; }; done; exit 1",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if not out:
        raise RuntimeError("could not discover G1 depth video device")
    return out.splitlines()[0].strip()


def discover_rgb_device(robot_host: str) -> str:
    cmd = [
        SSH,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"unitree@{robot_host}",
        "for d in /dev/video*; do ffmpeg -hide_banner -f v4l2 -list_formats all -i $d 2>&1 | grep -q yuyv422 && { echo $d; exit 0; }; done; exit 1",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if not out:
        raise RuntimeError("could not discover G1 RGB video device")
    return out.splitlines()[0].strip()


def handler_factory(shared: DashboardState, args: argparse.Namespace) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, indent=2, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/motion/go", "/motion/stop", "/motion/toggle"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not args.allow_browser_motion_control:
                self._write_json(
                    {
                        "ok": False,
                        "error": "browser motion control was not enabled at server startup",
                        "hint": "restart with --allow-browser-motion-control",
                        "motion_control": shared.motion_snapshot(),
                    },
                    HTTPStatus.FORBIDDEN,
                )
                return
            if parsed.path == "/motion/go":
                motion = shared.set_motion_enabled(True)
            elif parsed.path == "/motion/stop":
                motion = shared.set_motion_enabled(False)
            else:
                motion = shared.toggle_motion_enabled()
            self._write_json({"ok": True, "motion_control": motion})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                html = f"""<!doctype html><html><head><title>G1 Follow Dashboard</title>
<style>
body{{margin:0;background:#111;color:#eee;font-family:Arial,sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:14px}}
.bar{{display:flex;align-items:center;gap:12px;margin:0 0 12px}}
button{{border:0;border-radius:6px;padding:12px 20px;font-weight:700;font-size:16px;cursor:pointer}}
button.go{{background:#16a34a;color:white}}
button.stop{{background:#dc2626;color:white}}
button:disabled{{background:#3f3f46;color:#aaa;cursor:not-allowed}}
.status{{font-size:15px;color:#d4d4d8}}
img{{width:100%;height:auto;background:#222}}
pre{{font-size:14px;white-space:pre-wrap}}
</style>
</head><body><main><h2>G1 Follow Dashboard</h2>
<div class="bar"><button id="toggle" disabled>Loading...</button><span class="status" id="motion"></span></div>
<img id="view" src="/snapshot.jpg"><pre id="state"></pre>
<script>
const browserMotionAllowed = {str(bool(args.allow_browser_motion_control)).lower()};
async function setMotion(path){{
  try {{ await fetch(path, {{method:'POST'}}); }} catch(e) {{}}
  await tick();
}}
async function tick(){{
  try {{
    document.getElementById('view').src = '/snapshot.jpg?t=' + Date.now();
    let r = await fetch('/state?t=' + Date.now());
    let state = await r.json();
    let ctl = state.motion_control || {{enabled:false}};
    let btn = document.getElementById('toggle');
    btn.disabled = !browserMotionAllowed;
    btn.textContent = ctl.enabled ? 'STOP' : 'GO';
    btn.className = ctl.enabled ? 'stop' : 'go';
    btn.onclick = () => setMotion(ctl.enabled ? '/motion/stop' : '/motion/go');
    document.getElementById('motion').textContent = browserMotionAllowed
      ? (ctl.enabled ? 'motion state is GO; a running robot bridge may move' : 'motion state is STOP')
      : 'browser motion control disabled; restart server with --allow-browser-motion-control';
    document.getElementById('state').textContent = JSON.stringify(state, null, 2);
  }} catch(e) {{}}
}}
setInterval(tick, 350); tick();
</script>
</main></body></html>"""
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
            if path == "/target.json":
                _, state = shared.snapshot()
                payload = {
                    "dashboard_sends_motion_commands": False,
                    "browser_motion_control_allowed": bool(args.allow_browser_motion_control),
                    "time_unix": state.get("time_unix"),
                    "motion_control": state.get("motion_control", shared.motion_snapshot()),
                    "policy_target_source": state.get("policy_target_source"),
                    "target_estimate": state.get("target_estimate"),
                    "rgb_person_estimate": state.get("rgb_person_estimate"),
                    "depth_target_estimate": state.get("depth_target_estimate"),
                    "closest_scene_object": state.get("closest_scene_object"),
                    "safety_stop": state.get("safety_stop"),
                    "raw_policy_vx_vy_wz_stop": state.get("raw_policy_vx_vy_wz_stop"),
                    "wanted_input_vx_vy_wz_stop": state.get("wanted_input_vx_vy_wz_stop"),
                    "note": "Depth target is measured in raw RealSense depth space. RGB person projection needs calibration before it is trusted.",
                }
                data = json.dumps(payload, indent=2, sort_keys=True).encode()
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
            if path == "/depth.raw":
                depth = shared.depth_snapshot()
                if depth is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no depth frame yet")
                    return
                data = depth.astype("<u2", copy=False).tobytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Depth-Width", str(depth.shape[1]))
                self.send_header("X-Depth-Height", str(depth.shape[0]))
                self.send_header("X-Depth-Dtype", "uint16_le_mm")
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/depth.json":
                depth = shared.depth_snapshot()
                if depth is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no depth frame yet")
                    return
                query = parse_qs(parsed.query)
                valid = depth[(depth > 0) & (depth < 10000)]
                payload: dict[str, Any] = {
                    "dtype": "uint16",
                    "units": "millimeters",
                    "shape_hw": [int(depth.shape[0]), int(depth.shape[1])],
                    "valid_pixels": int(valid.size),
                    "min_mm": int(valid.min()) if valid.size else None,
                    "max_mm": int(valid.max()) if valid.size else None,
                    "p05_mm": float(np.percentile(valid, 5)) if valid.size else None,
                    "p10_mm": float(np.percentile(valid, 10)) if valid.size else None,
                    "p35_mm": float(np.percentile(valid, 35)) if valid.size else None,
                    "p50_mm": float(np.percentile(valid, 50)) if valid.size else None,
                    "p80_mm": float(np.percentile(valid, 80)) if valid.size else None,
                    "center_pixel_mm": int(depth[depth.shape[0] // 2, depth.shape[1] // 2]),
                }
                if {"x0", "y0", "x1", "y1"}.issubset(query):
                    try:
                        rect = (
                            int(float(query["x0"][0])),
                            int(float(query["y0"][0])),
                            int(float(query["x1"][0])),
                            int(float(query["y1"][0])),
                        )
                    except (TypeError, ValueError):
                        self.send_error(HTTPStatus.BAD_REQUEST, "x0,y0,x1,y1 must be numeric")
                        return
                    payload["region"] = depth_region_summary(depth, rect, args)
                if query.get("full", ["0"])[0] in {"1", "true", "yes"}:
                    payload["data_mm"] = depth.tolist()
                data = json.dumps(payload, separators=(",", ":")).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/depth.png":
                depth = shared.depth_snapshot()
                if depth is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no depth frame yet")
                    return
                import io

                buf = io.BytesIO()
                Image.fromarray(colorize_depth(depth)).save(buf, format="PNG")
                data = buf.getvalue()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    jpeg, _ = shared.snapshot()
                    if jpeg:
                        try:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                            self.wfile.write(jpeg + b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    time.sleep(0.1)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-host", default="192.168.0.108")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--segmenter-model", type=Path, default=DEFAULT_SEGMENTER_MODEL)
    parser.add_argument("--trace-mode", choices=("fastsam", "off"), default="fastsam")
    parser.add_argument("--segment-conf", type=float, default=0.2)
    parser.add_argument("--segment-imgsz", type=int, default=640)
    parser.add_argument("--segment-every", type=int, default=1)
    parser.add_argument("--trace-min-box-coverage", type=float, default=0.18)
    parser.add_argument("--trace-min-mask-inside-box", type=float, default=0.35)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--rgb-source", choices=("realsense", "front-rtsp"), default="realsense")
    parser.add_argument("--rgb-device", default="auto")
    parser.add_argument("--rgb-input-format", default="yuyv422")
    parser.add_argument("--rgb-capture-width", type=int, default=424)
    parser.add_argument("--rgb-capture-height", type=int, default=240)
    parser.add_argument("--rgb-fps", type=int, default=15)
    parser.add_argument("--rgb-width", type=int, default=424)
    parser.add_argument("--rgb-height", type=int, default=240)
    parser.add_argument("--rgb-snapshot", action="store_true", help="Capture RGB one frame at a time instead of using a long-lived ffmpeg stream")
    parser.add_argument("--rgb-snapshot-timeout", type=float, default=3.5)
    parser.add_argument("--depth-device", default="auto")
    parser.add_argument("--depth-width", type=int, default=424)
    parser.add_argument("--depth-height", type=int, default=240)
    parser.add_argument("--depth-fps", type=int, default=15)
    parser.add_argument("--depth-snapshot-timeout", type=float, default=2.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--policy-width", type=int, default=64)
    parser.add_argument("--policy-height", type=int, default=36)
    parser.add_argument("--depth-near-mm", type=float, default=350.0)
    parser.add_argument("--depth-far-mm", type=float, default=3600.0)
    parser.add_argument("--policy-target-source", choices=("auto", "rgb-person", "depth-nearest"), default="auto")
    parser.add_argument("--depth-roi-x0", type=float, default=0.15)
    parser.add_argument("--depth-roi-x1", type=float, default=0.85)
    parser.add_argument("--depth-roi-y0", type=float, default=0.12)
    parser.add_argument("--depth-roi-y1", type=float, default=0.92)
    parser.add_argument("--depth-target-near-percentile", type=float, default=12.0)
    parser.add_argument("--depth-target-slack-mm", type=float, default=240.0)
    parser.add_argument("--depth-target-percentile", type=float, default=35.0)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=250)
    parser.add_argument("--min-depth-target-pixels", type=int, default=45)
    parser.add_argument("--person-depth-percentile", type=float, default=35.0)
    parser.add_argument("--person-depth-near-mm", type=float, default=100.0)
    parser.add_argument("--person-crop-x0", type=float, default=0.18)
    parser.add_argument("--person-crop-x1", type=float, default=0.82)
    parser.add_argument("--person-crop-y0", type=float, default=0.12)
    parser.add_argument("--person-crop-y1", type=float, default=0.78)
    parser.add_argument("--min-person-depth-pixels", type=int, default=80)
    parser.add_argument("--scene-roi-x0", type=float, default=0.04)
    parser.add_argument("--scene-roi-x1", type=float, default=0.96)
    parser.add_argument("--scene-roi-y0", type=float, default=0.02)
    parser.add_argument("--scene-roi-y1", type=float, default=0.88)
    parser.add_argument("--scene-object-slack-mm", type=float, default=180.0)
    parser.add_argument("--min-scene-object-pixels", type=int, default=80)
    parser.add_argument("--safety-stop-m", type=float, default=0.40)
    parser.add_argument(
        "--allow-browser-motion-control",
        action="store_true",
        help="Enable the browser GO/STOP state for a separate robot bridge to poll. The dashboard still does not publish Unitree commands.",
    )
    args = parser.parse_args()
    args.browser_motion_control_enabled = False

    shared = DashboardState()
    thread = threading.Thread(target=worker, args=(shared, args), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.bind, args.port), handler_factory(shared, args))
    print(f"DRY RUN ONLY: serving http://{args.bind}:{args.port}", flush=True)
    print("No Unitree SDK import and no motion command publisher.", flush=True)
    try:
        server.serve_forever()
    finally:
        shared.stop = True
        server.server_close()


if __name__ == "__main__":
    main()
