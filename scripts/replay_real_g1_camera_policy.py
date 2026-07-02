#!/usr/bin/env python3
"""Replay real G1 RGB/depth capture through the visual command policy.

The current visual command policy was trained on a canonical synthetic target
view. This replay keeps the real sensor evidence visible, estimates a target
from real depth, converts that estimate into the canonical policy observation,
and logs the resulting vx/vy/wz/stop commands.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_visual_command_policy import forward, load_policy, render_observation  # noqa: E402


DEFAULT_CAPTURE = ROOT / "artifacts" / "g1-camera-2026-06-22" / "side_by_side_20260622_153043"
DEFAULT_POLICY = ROOT / "artifacts" / "visual-command" / "2026-06-23" / "moving-random-1091043328" / "latest_policy.npz"
DEFAULT_OUT = ROOT / "artifacts" / "g1-camera-2026-06-22" / "policy_replay_20260622_153043"
DEFAULT_DETECTIONS = DEFAULT_OUT / "person_detections_yolo.jsonl"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def ffprobe_video(path: Path) -> dict[str, object]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        cwd=ROOT,
        text=True,
    )
    return json.loads(out)


def parse_rate(rate: str) -> float:
    if "/" not in rate:
        return float(rate)
    num, den = rate.split("/", 1)
    return float(num) / float(den)


def load_front_frames(path: Path, limit: int | None = None) -> tuple[list[np.ndarray], float]:
    info = ffprobe_video(path)
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = parse_rate(str(stream["r_frame_rate"]))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=ROOT)
    assert proc.stdout is not None
    frame_bytes = width * height * 3
    frames: list[np.ndarray] = []
    while limit is None or len(frames) < limit:
        data = proc.stdout.read(frame_bytes)
        if len(data) < frame_bytes:
            break
        frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()
        frames.append(frame)
    proc.stdout.close()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed reading {path} with exit code {code}")
    return frames, fps


def load_depth_frames(path: Path, width: int, height: int, limit: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    frame_size = width * height
    count = raw.size // frame_size
    if limit is not None:
        count = min(count, limit)
    if count == 0:
        raise ValueError(f"no depth frames in {path}")
    return raw[: count * frame_size].reshape(count, height, width)


def load_detections(path: Path | None, limit: int | None = None) -> dict[int, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    detections: dict[int, dict[str, object]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            frame = int(payload["frame"])
            detections[frame] = payload
            if limit is not None and len(detections) >= limit:
                break
    return detections


def resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def resize_float(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(frame.astype(np.float32), mode="F")
    return np.asarray(img.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def colorize_depth_mm(depth_mm: np.ndarray, near_mm: float, far_mm: float) -> np.ndarray:
    valid = depth_mm > 0
    norm = np.clip((depth_mm.astype(np.float32) - near_mm) / max(1.0, far_mm - near_mm), 0.0, 1.0)
    out = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    out[..., 0] = ((1.0 - norm) * 255.0).astype(np.uint8)
    out[..., 2] = (norm * 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def direct_policy_obs(rgb: np.ndarray, depth_mm: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    rgb_small = resize_rgb(rgb, (args.policy_width, args.policy_height))
    depth_small_mm = resize_float(depth_mm.astype(np.float32), (args.policy_width, args.policy_height))
    valid = depth_small_mm > 0.0
    closeness = 1.0 - (depth_small_mm - args.depth_near_mm) / max(1.0, args.depth_far_mm - args.depth_near_mm)
    closeness = np.clip(closeness, 0.0, 1.0)
    closeness[~valid] = 0.05
    obs = np.zeros((4, args.policy_height, args.policy_width), dtype=np.float32)
    obs[:3] = np.moveaxis(rgb_small, -1, 0)
    obs[3] = closeness
    return obs


def estimate_target_from_depth(depth_mm: np.ndarray, args: argparse.Namespace) -> dict[str, float | bool | int]:
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
    mask_count = int(mask.sum())
    if mask_count < args.min_target_pixels:
        return {"visible": False, "reason": "too_few_near_pixels", "valid_pixels": valid_count, "target_pixels": mask_count}

    yy, xx = np.nonzero(mask)
    depths = roi[mask]
    weights = np.clip((args.depth_far_mm - depths) / max(1.0, args.depth_far_mm - args.depth_near_mm), 0.05, 1.0)
    cx_roi = float(np.average(xx.astype(np.float32), weights=weights))
    cy_roi = float(np.average(yy.astype(np.float32), weights=weights))
    depth_m = float(np.percentile(depths, 35) / 1000.0)
    cx = cx_roi + x0
    cy = cy_roi + y0
    bearing = ((cx / max(1.0, w - 1.0)) - 0.5) * math.radians(args.fov_deg)
    front = max(0.05, depth_m * math.cos(bearing))
    lateral = depth_m * math.sin(bearing)
    return {
        "visible": True,
        "reason": "near_depth_target",
        "valid_pixels": valid_count,
        "target_pixels": mask_count,
        "pixel_x": cx,
        "pixel_y": cy,
        "depth_m": depth_m,
        "bearing_rad": bearing,
        "front_m": front,
        "lateral_m": lateral,
    }


def estimate_target_from_person_box(
    depth_mm: np.ndarray,
    detection: dict[str, object] | None,
    rgb_width: int,
    rgb_height: int,
    args: argparse.Namespace,
) -> dict[str, float | bool | int | str]:
    if not detection or not detection.get("best_person"):
        out = estimate_target_from_depth(depth_mm, args)
        out["detector"] = "depth_fallback"
        return out

    best = detection["best_person"]
    if not isinstance(best, dict):
        out = estimate_target_from_depth(depth_mm, args)
        out["detector"] = "depth_fallback_bad_detection"
        return out
    xyxy = best.get("xyxy")
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        out = estimate_target_from_depth(depth_mm, args)
        out["detector"] = "depth_fallback_bad_box"
        return out

    h, w = depth_mm.shape
    x0_rgb, y0_rgb, x1_rgb, y1_rgb = [float(x) for x in xyxy]
    cx_rgb = 0.5 * (x0_rgb + x1_rgb)
    # Use the middle/lower part of the person box for depth so a face or hand
    # does not dominate, but keep horizontal control tied to person center.
    dx0 = int(np.clip(x0_rgb / rgb_width * w, 0, w - 1))
    dx1 = int(np.clip(x1_rgb / rgb_width * w, dx0 + 1, w))
    dy0 = int(np.clip((y0_rgb + 0.25 * (y1_rgb - y0_rgb)) / rgb_height * h, 0, h - 1))
    dy1 = int(np.clip((y0_rgb + 0.92 * (y1_rgb - y0_rgb)) / rgb_height * h, dy0 + 1, h))
    crop = depth_mm[dy0:dy1, dx0:dx1].astype(np.float32)
    valid = (crop >= args.depth_near_mm) & (crop <= args.depth_far_mm)
    valid_count = int(valid.sum())
    if valid_count < args.min_target_pixels:
        out = estimate_target_from_depth(depth_mm, args)
        out["detector"] = "depth_fallback_no_person_depth"
        out["person_confidence"] = float(best.get("confidence", 0.0))
        return out

    depths = crop[valid]
    depth_m = float(np.percentile(depths, args.person_depth_percentile) / 1000.0)
    cx_depth = cx_rgb / max(1.0, rgb_width - 1.0) * (w - 1.0)
    cy_depth = 0.5 * (dy0 + dy1)
    bearing = ((cx_depth / max(1.0, w - 1.0)) - 0.5) * math.radians(args.fov_deg)
    front = max(0.05, depth_m * math.cos(bearing))
    lateral = depth_m * math.sin(bearing)
    return {
        "visible": True,
        "reason": "yolo_person_depth",
        "detector": "yolo_person",
        "person_confidence": float(best.get("confidence", 0.0)),
        "valid_pixels": valid_count,
        "target_pixels": valid_count,
        "pixel_x": cx_depth,
        "pixel_y": cy_depth,
        "depth_m": depth_m,
        "bearing_rad": bearing,
        "front_m": front,
        "lateral_m": lateral,
    }


def canonical_policy_obs(estimate: dict[str, float | bool | int], args: argparse.Namespace) -> np.ndarray:
    visible = bool(estimate.get("visible", False))
    front = float(estimate.get("front_m", 0.0))
    lateral = float(estimate.get("lateral_m", 0.0))
    return render_observation(
        np.array([front], dtype=np.float32),
        np.array([lateral], dtype=np.float32),
        np.array([visible]),
        width=args.policy_width,
        height=args.policy_height,
        fov_deg=args.fov_deg,
    )[0]


def obs_to_panel(obs: np.ndarray, scale: int) -> tuple[Image.Image, Image.Image]:
    rgb = np.moveaxis(obs[:3], 0, -1)
    depth = obs[3]
    depth_rgb = np.zeros((*depth.shape, 3), dtype=np.float32)
    depth_rgb[..., 0] = depth
    depth_rgb[..., 2] = 1.0 - depth
    rgb_img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
        (obs.shape[2] * scale, obs.shape[1] * scale), Image.Resampling.NEAREST
    )
    depth_img = Image.fromarray((np.clip(depth_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
        (obs.shape[2] * scale, obs.shape[1] * scale), Image.Resampling.NEAREST
    )
    return rgb_img, depth_img


def draw_frame(
    *,
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    direct_obs: np.ndarray,
    adapted_obs: np.ndarray,
    direct_command: np.ndarray,
    adapted_command: np.ndarray,
    estimate: dict[str, float | bool | int],
    frame_idx: int,
    fps: float,
    args: argparse.Namespace,
) -> Image.Image:
    out_w, out_h = 1280, 720
    img = Image.new("RGB", (out_w, out_h), (245, 247, 250))
    draw = ImageDraw.Draw(img)
    title_font = font(27, True)
    label_font = font(18, True)
    body_font = font(16)
    small_font = font(13)

    draw.text((30, 22), "Real G1 camera replay through visual command policy", fill=(21, 29, 40), font=title_font)
    draw.text(
        (32, 58),
        "Real RGB/depth is shown on the left. Right side shows the policy inputs and emitted commands.",
        fill=(74, 84, 98),
        font=body_font,
    )

    rgb_img = Image.fromarray(rgb).resize((610, 343), Image.Resampling.BILINEAR)
    depth_rgb = colorize_depth_mm(depth_mm, args.depth_near_mm, args.depth_far_mm)
    depth_img = Image.fromarray(depth_rgb).resize((610, 343), Image.Resampling.NEAREST)

    draw.rectangle((30, 96, 640, 463), fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((44, 108), "Real RGB", fill=(30, 42, 58), font=label_font)
    img.paste(rgb_img, (30, 132))

    if bool(estimate.get("visible", False)):
        sx = 30 + int(float(estimate["pixel_x"]) / args.depth_width * 610)
        sy = 132 + int(float(estimate["pixel_y"]) / args.depth_height * 343)
        draw.ellipse((sx - 13, sy - 13, sx + 13, sy + 13), outline=(255, 214, 0), width=4)
        draw.line((sx - 18, sy, sx + 18, sy), fill=(255, 214, 0), width=2)
        draw.line((sx, sy - 18, sx, sy + 18), fill=(255, 214, 0), width=2)

    draw.rectangle((30, 492, 640, 690), fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((44, 504), "Real depth Z16", fill=(30, 42, 58), font=label_font)
    img.paste(depth_img.resize((300, 170), Image.Resampling.NEAREST), (44, 530))
    est_lines = [
        f"frame {frame_idx}  time {frame_idx / max(1.0, fps):.2f}s",
        f"target visible {str(bool(estimate.get('visible', False))).lower()}",
        f"depth {float(estimate.get('depth_m', 0.0)):.2f}m  lateral {float(estimate.get('lateral_m', 0.0)):+.2f}m",
        f"front {float(estimate.get('front_m', 0.0)):.2f}m  pixels {int(estimate.get('target_pixels', 0))}",
    ]
    for i, line in enumerate(est_lines):
        draw.text((370, 532 + i * 28), line, fill=(45, 55, 70), font=body_font)

    direct_rgb, direct_depth = obs_to_panel(direct_obs, scale=5)
    adapted_rgb, adapted_depth = obs_to_panel(adapted_obs, scale=5)

    x = 675
    draw.rectangle((x, 96, 1250, 296), fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((x + 14, 108), "Direct raw policy input", fill=(30, 42, 58), font=label_font)
    img.paste(direct_rgb, (x + 14, 136))
    img.paste(direct_depth, (x + 348, 136))
    draw.text((x + 14, 324), "Direct raw command", fill=(30, 42, 58), font=label_font)
    draw.text(
        (x + 14, 352),
        f"vx {direct_command[0]:+.3f}  vy {direct_command[1]:+.3f}  wz {direct_command[2]:+.3f}  stop {direct_command[3]:.3f}",
        fill=(45, 55, 70),
        font=body_font,
    )
    draw.text((x + 14, 378), "Expected to stop unless real image matches training target style.", fill=(100, 110, 124), font=small_font)

    draw.rectangle((x, 422, 1250, 622), fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((x + 14, 434), "Adapted real-depth target input", fill=(30, 42, 58), font=label_font)
    img.paste(adapted_rgb, (x + 14, 462))
    img.paste(adapted_depth, (x + 348, 462))
    draw.text((x + 14, 650), "Adapted command for walker", fill=(30, 42, 58), font=label_font)
    draw.text(
        (x + 14, 678),
        f"vx {adapted_command[0]:+.3f}  vy {adapted_command[1]:+.3f}  wz {adapted_command[2]:+.3f}  stop {adapted_command[3]:.3f}",
        fill=(22, 90, 54),
        font=body_font,
    )
    return img


def run_replay(args: argparse.Namespace) -> tuple[Path, Path]:
    args.out.mkdir(parents=True, exist_ok=True)
    video_path = args.out / "real_g1_camera_policy_replay.mp4"
    preview_path = args.out / "preview.jpg"
    commands_path = args.out / "commands.jsonl"
    summary_path = args.out / "summary.json"
    if commands_path.exists():
        commands_path.unlink()

    rgb_frames, fps = load_front_frames(args.front_video, limit=args.max_frames)
    depth_frames = load_depth_frames(args.depth_raw, args.depth_width, args.depth_height, limit=args.max_frames)
    frame_count = min(len(rgb_frames), depth_frames.shape[0])
    if frame_count == 0:
        raise SystemExit("no aligned RGB/depth frames found")
    rgb_frames = rgb_frames[:frame_count]
    depth_frames = depth_frames[:frame_count]
    detections = load_detections(args.person_detections, limit=frame_count)

    params = load_policy(args.policy)
    out_w, out_h = 1280, 720
    ffmpeg = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    proc = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE, cwd=ROOT)
    assert proc.stdin is not None

    visible_count = 0
    direct_stops: list[float] = []
    adapted_stops: list[float] = []
    adapted_speed: list[float] = []
    try:
        for frame_idx, (rgb, depth_mm) in enumerate(zip(rgb_frames, depth_frames)):
            direct_obs = direct_policy_obs(rgb, depth_mm, args)
            if args.detector == "person-yolo":
                estimate = estimate_target_from_person_box(
                    depth_mm,
                    detections.get(frame_idx),
                    rgb.shape[1],
                    rgb.shape[0],
                    args,
                )
            else:
                estimate = estimate_target_from_depth(depth_mm, args)
            adapted_obs = canonical_policy_obs(estimate, args)
            direct_command, _, _, _ = forward(params, direct_obs[None, :, :, :], args.encoder)
            adapted_command, _, _, _ = forward(params, adapted_obs[None, :, :, :], args.encoder)
            direct = direct_command[0]
            adapted = adapted_command[0]
            visible_count += int(bool(estimate.get("visible", False)))
            direct_stops.append(float(direct[3]))
            adapted_stops.append(float(adapted[3]))
            adapted_speed.append(float(abs(adapted[0]) + abs(adapted[1]) + abs(adapted[2])))

            record = {
                "frame": frame_idx,
                "time_s": round(frame_idx / max(1.0, fps), 4),
                "target_estimate": {
                    key: (bool(value) if isinstance(value, np.bool_) else float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value)
                    for key, value in estimate.items()
                },
                "direct_command_vx_vy_wz_stop": [round(float(x), 6) for x in direct],
                "adapted_command_vx_vy_wz_stop": [round(float(x), 6) for x in adapted],
            }
            append_jsonl(commands_path, record)

            frame = draw_frame(
                rgb=rgb,
                depth_mm=depth_mm,
                direct_obs=direct_obs,
                adapted_obs=adapted_obs,
                direct_command=direct,
                adapted_command=adapted,
                estimate=estimate,
                frame_idx=frame_idx,
                fps=fps,
                args=args,
            )
            if frame_idx == min(frame_count - 1, frame_count // 2):
                frame.save(preview_path, quality=92)
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {code}")

    summary = {
        "created_unix": round(time.time(), 3),
        "front_video": str(args.front_video),
        "depth_raw": str(args.depth_raw),
        "policy": str(args.policy),
        "video": str(video_path),
        "preview": str(preview_path),
        "commands": str(commands_path),
        "frames": frame_count,
        "fps": fps,
        "depth_shape_hw": [args.depth_height, args.depth_width],
        "policy_shape_chw": [4, args.policy_height, args.policy_width],
        "adapter": "real depth target estimate -> canonical trained policy observation",
        "detector": args.detector,
        "person_detections": str(args.person_detections) if args.person_detections else None,
        "direct_path_note": "Direct raw RGB/depth is logged and shown, but current policy was trained on a synthetic blue target.",
        "visible_fraction": float(visible_count / max(1, frame_count)),
        "direct_mean_stop_probability": float(np.mean(direct_stops)),
        "adapted_mean_stop_probability": float(np.mean(adapted_stops)),
        "adapted_mean_abs_command": float(np.mean(adapted_speed)),
        "depth_near_mm": args.depth_near_mm,
        "depth_far_mm": args.depth_far_mm,
        "fov_deg_assumption": args.fov_deg,
    }
    write_json(summary_path, summary)
    return video_path, preview_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-video", type=Path, default=DEFAULT_CAPTURE / "front.mp4")
    parser.add_argument("--depth-raw", type=Path, default=DEFAULT_CAPTURE / "depth_z16_424x240_15fps.raw")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--person-detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--detector", choices=("person-yolo", "depth-nearest"), default="person-yolo")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--encoder", choices=("features", "raw"), default="features")
    parser.add_argument("--policy-width", type=int, default=64)
    parser.add_argument("--policy-height", type=int, default=36)
    parser.add_argument("--depth-width", type=int, default=424)
    parser.add_argument("--depth-height", type=int, default=240)
    parser.add_argument("--depth-near-mm", type=float, default=350.0)
    parser.add_argument("--depth-far-mm", type=float, default=3600.0)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--roi-x0", type=float, default=0.08)
    parser.add_argument("--roi-x1", type=float, default=0.92)
    parser.add_argument("--roi-y0", type=float, default=0.08)
    parser.add_argument("--roi-y1", type=float, default=0.95)
    parser.add_argument("--near-percentile", type=float, default=18.0)
    parser.add_argument("--near-slack-mm", type=float, default=180.0)
    parser.add_argument("--min-valid-pixels", type=int, default=1200)
    parser.add_argument("--min-target-pixels", type=int, default=160)
    parser.add_argument("--person-depth-percentile", type=float, default=35.0)
    args = parser.parse_args()

    for path in (args.front_video, args.depth_raw, args.policy):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    video_path, preview_path = run_replay(args)
    print(json.dumps({"preview": str(preview_path), "video": str(video_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
