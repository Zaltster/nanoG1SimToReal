#!/usr/bin/env python3
"""Render a closed-loop video of the trained visual-command policy.

The policy input is the same synthetic RGB/depth raster used during native
visual-command training. The rollout integrates the emitted vx/vy/wz command
through a simple 2D command interface that stands in for the frozen 1.09B walker.
This is evidence for the camera/depth command layer, not full-body MuJoCo
physics.
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


DEFAULT_RUN = ROOT / "artifacts" / "visual-command" / "2026-06-23" / "moving-random-1091043328"
DEFAULT_POLICY = DEFAULT_RUN / "latest_policy.npz"
DEFAULT_OUT = DEFAULT_RUN / "video"
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def world_to_panel(point: np.ndarray, center: np.ndarray, scale: float, origin: tuple[int, int]) -> tuple[int, int]:
    x = origin[0] + int(round((point[0] - center[0]) * scale))
    y = origin[1] - int(round((point[1] - center[1]) * scale))
    return x, y


def obs_to_rgb_depth(obs: np.ndarray) -> tuple[Image.Image, Image.Image]:
    rgb = np.moveaxis(obs[:3], 0, -1)
    depth = obs[3]
    depth_rgb = np.zeros((*depth.shape, 3), dtype=np.float32)
    depth_rgb[..., 0] = depth
    depth_rgb[..., 2] = 1.0 - depth
    rgb_img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    depth_img = Image.fromarray((np.clip(depth_rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    return rgb_img, depth_img


def draw_trail(
    draw: ImageDraw.ImageDraw,
    points: list[np.ndarray],
    center: np.ndarray,
    scale: float,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    if len(points) < 2:
        return
    screen = [world_to_panel(p, center, scale, origin) for p in points[-120:]]
    for i in range(1, len(screen)):
        alpha = i / max(1, len(screen) - 1)
        faded = tuple(int(c * (0.35 + 0.65 * alpha)) for c in color)
        draw.line([screen[i - 1], screen[i]], fill=faded, width=3)


def draw_frame(
    *,
    width: int,
    height: int,
    t: float,
    robot_pos: np.ndarray,
    robot_yaw: float,
    target_pos: np.ndarray,
    robot_trail: list[np.ndarray],
    target_trail: list[np.ndarray],
    obs: np.ndarray,
    command: np.ndarray,
    visible: bool,
    distance: float,
    front: float,
    lateral: float,
    fov_deg: float,
    seed: int,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(img)
    title_font = font(28, True)
    label_font = font(19, True)
    body_font = font(17)
    small_font = font(14)

    draw.rectangle((0, 0, width, height), fill=(245, 247, 250))
    draw.text((34, 24), "Visual command policy following a moving target", fill=(22, 28, 36), font=title_font)
    draw.text(
        (36, 60),
        "Input: RGB + depth frame only. Output: vx, vy, wz, stop -> frozen 1.09B walker command interface.",
        fill=(72, 82, 96),
        font=body_font,
    )

    panel = (36, 98, 904, 678)
    draw.rectangle(panel, fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((panel[0] + 18, panel[1] + 14), "Closed-loop top-down rollout", fill=(30, 42, 58), font=label_font)

    map_rect = (74, 148, 866, 636)
    draw.rectangle(map_rect, fill=(236, 240, 245), outline=(206, 214, 224), width=1)
    map_center = robot_pos.copy()
    scale = 86.0
    origin = ((map_rect[0] + map_rect[2]) // 2, (map_rect[1] + map_rect[3]) // 2)

    for gx in np.arange(-8.0, 8.01, 1.0):
        a = world_to_panel(map_center + np.array([gx, -5.0]), map_center, scale, origin)
        b = world_to_panel(map_center + np.array([gx, 5.0]), map_center, scale, origin)
        if map_rect[0] <= a[0] <= map_rect[2]:
            draw.line([(a[0], map_rect[1]), (a[0], map_rect[3])], fill=(220, 226, 234), width=1)
    for gy in np.arange(-5.0, 5.01, 1.0):
        a = world_to_panel(map_center + np.array([-8.0, gy]), map_center, scale, origin)
        if map_rect[1] <= a[1] <= map_rect[3]:
            draw.line([(map_rect[0], a[1]), (map_rect[2], a[1])], fill=(220, 226, 234), width=1)

    draw_trail(draw, target_trail, map_center, scale, origin, (24, 92, 210))
    draw_trail(draw, robot_trail, map_center, scale, origin, (225, 84, 45))

    robot_screen = world_to_panel(robot_pos, map_center, scale, origin)
    target_screen = world_to_panel(target_pos, map_center, scale, origin)
    fwd = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
    left = np.array([-math.sin(robot_yaw), math.cos(robot_yaw)])
    half = math.radians(fov_deg / 2.0)
    far = 3.8
    left_ray = robot_pos + far * (math.cos(half) * fwd + math.sin(half) * left)
    right_ray = robot_pos + far * (math.cos(-half) * fwd + math.sin(-half) * left)
    wedge = [
        robot_screen,
        world_to_panel(left_ray, map_center, scale, origin),
        world_to_panel(right_ray, map_center, scale, origin),
    ]
    draw.polygon(wedge, fill=(255, 214, 153), outline=(236, 151, 54))

    draw.line([robot_screen, world_to_panel(robot_pos + 0.7 * fwd, map_center, scale, origin)], fill=(20, 26, 38), width=5)
    draw.ellipse((robot_screen[0] - 14, robot_screen[1] - 14, robot_screen[0] + 14, robot_screen[1] + 14), fill=(225, 84, 45), outline=(130, 45, 24), width=3)
    draw.ellipse((target_screen[0] - 15, target_screen[1] - 15, target_screen[0] + 15, target_screen[1] + 15), fill=(24, 92, 210), outline=(8, 43, 120), width=3)
    draw.text((robot_screen[0] + 18, robot_screen[1] - 12), "G1", fill=(90, 42, 30), font=small_font)
    draw.text((target_screen[0] + 18, target_screen[1] - 12), "person target", fill=(14, 55, 130), font=small_font)

    status_y = map_rect[3] - 72
    status = [
        f"time {t:04.1f}s",
        f"distance {distance:.2f}m",
        f"front {front:.2f}m",
        f"lateral {lateral:.2f}m",
        f"visible {str(visible).lower()}",
    ]
    for i, text in enumerate(status):
        x = map_rect[0] + 16 + i * 148
        draw.rounded_rectangle((x - 6, status_y - 6, x + 130, status_y + 24), radius=4, fill=(255, 255, 255), outline=(212, 220, 230))
        draw.text((x, status_y), text, fill=(44, 52, 64), font=small_font)

    cam_w, cam_h = 320, 180
    rgb_img, depth_img = obs_to_rgb_depth(obs)
    rgb_big = rgb_img.resize((cam_w, cam_h), Image.Resampling.NEAREST)
    depth_big = depth_img.resize((cam_w, cam_h), Image.Resampling.NEAREST)

    left_x, top_y = 936, 98
    draw.rectangle((left_x, top_y, left_x + cam_w, top_y + cam_h + 34), fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((left_x + 12, top_y + 9), "Depth camera", fill=(30, 42, 58), font=label_font)
    img.paste(depth_big, (left_x, top_y + 34))
    draw.text((left_x + 12, top_y + cam_h + 42), "red close, blue far", fill=(72, 82, 96), font=small_font)

    top_y2 = 342
    draw.rectangle((left_x, top_y2, left_x + cam_w, top_y2 + cam_h + 34), fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((left_x + 12, top_y2 + 9), "RGB camera", fill=(30, 42, 58), font=label_font)
    img.paste(rgb_big, (left_x, top_y2 + 34))

    cmd_y = 594
    draw.rectangle((936, cmd_y, 1256, 678), fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((952, cmd_y + 12), "Policy command", fill=(30, 42, 58), font=label_font)
    cmd_text = f"vx {command[0]:+.3f}  vy {command[1]:+.3f}  wz {command[2]:+.3f}"
    stop_text = f"stop probability {command[3]:.3f}"
    draw.text((952, cmd_y + 42), cmd_text, fill=(44, 52, 64), font=body_font)
    draw.text((952, cmd_y + 64), stop_text, fill=(44, 52, 64), font=body_font)
    draw.text((1030, 696), f"seed {seed}", fill=(126, 137, 151), font=small_font)
    return img


def run_rollout(args: argparse.Namespace) -> tuple[Path, Path]:
    params = load_policy(args.policy)
    args.out.mkdir(parents=True, exist_ok=True)
    video_path = args.out / "visual_policy_moving_follow.mp4"
    preview_path = args.out / "preview.jpg"
    rollout_path = args.out / "rollout.jsonl"
    summary_path = args.out / "summary.json"
    if rollout_path.exists():
        rollout_path.unlink()

    width, height = 1280, 720
    fps = args.fps
    dt = 1.0 / fps
    total_frames = int(round(args.seconds * fps))
    rng = np.random.default_rng(args.seed)

    robot_pos = np.array([0.0, 0.0], dtype=np.float64)
    robot_yaw = 0.0
    target_pos = np.array([2.45, 0.85], dtype=np.float64)
    target_goal_rel = np.array([2.45, 0.85], dtype=np.float64)
    robot_trail: list[np.ndarray] = []
    target_trail: list[np.ndarray] = []
    distances: list[float] = []
    stop_probs: list[float] = []
    visible_count = 0

    ffmpeg = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
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

    try:
        for frame_idx in range(total_frames):
            t = frame_idx * dt
            fwd_now = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
            left_now = np.array([-math.sin(robot_yaw), math.cos(robot_yaw)])
            if frame_idx % int(max(1, fps * 2.7)) == 0:
                target_goal_rel = np.array(
                    [
                        rng.uniform(1.35, 3.15),
                        rng.uniform(-1.05, 1.05),
                    ],
                    dtype=np.float64,
                )

            target_goal = robot_pos + fwd_now * target_goal_rel[0] + left_now * target_goal_rel[1]
            to_goal = target_goal - target_pos
            goal_dist = float(np.linalg.norm(to_goal))
            if goal_dist > 1e-6:
                target_vel = to_goal / goal_dist * min(args.target_speed, goal_dist * 0.9)
            else:
                target_vel = np.zeros(2, dtype=np.float64)
            target_vel += 0.035 * np.array([math.sin(0.9 * t), math.cos(0.7 * t)], dtype=np.float64)
            speed = float(np.linalg.norm(target_vel))
            if speed > args.target_speed:
                target_vel *= args.target_speed / speed
            target_pos += target_vel * dt

            delta = target_pos - robot_pos
            fwd = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
            left = np.array([-math.sin(robot_yaw), math.cos(robot_yaw)])
            front = float(delta @ fwd)
            lateral = float(delta @ left)
            distance = float(np.linalg.norm(delta))
            bearing_deg = math.degrees(math.atan2(lateral, max(front, 1e-4)))
            visible = bool(front > 0.0 and abs(bearing_deg) < args.fov_deg / 2.0 and distance < 5.0)
            visible_count += int(visible)

            obs = render_observation(
                np.array([front], dtype=np.float32),
                np.array([lateral], dtype=np.float32),
                np.array([visible]),
                width=args.camera_width,
                height=args.camera_height,
                fov_deg=args.fov_deg,
            )[0]
            noise = rng.normal(0.0, 0.006, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)
            pred, _, _, _ = forward(params, obs[None, :, :, :], args.encoder)
            command = pred[0].astype(np.float64)
            if command[3] > args.stop_threshold:
                command[:3] = 0.0

            robot_yaw += float(command[2]) * dt
            robot_yaw = math.atan2(math.sin(robot_yaw), math.cos(robot_yaw))
            fwd = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
            left = np.array([-math.sin(robot_yaw), math.cos(robot_yaw)])
            robot_pos += (fwd * float(command[0]) + left * float(command[1])) * dt

            robot_trail.append(robot_pos.copy())
            target_trail.append(target_pos.copy())
            distances.append(distance)
            stop_probs.append(float(pred[0, 3]))

            frame = draw_frame(
                width=width,
                height=height,
                t=t,
                robot_pos=robot_pos,
                robot_yaw=robot_yaw,
                target_pos=target_pos,
                robot_trail=robot_trail,
                target_trail=target_trail,
                obs=obs,
                command=pred[0],
                visible=visible,
                distance=distance,
                front=front,
                lateral=lateral,
                fov_deg=args.fov_deg,
                seed=args.seed,
            )
            if frame_idx == total_frames // 2:
                frame.save(preview_path, quality=92)
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

            if frame_idx % fps == 0:
                append_jsonl(
                    rollout_path,
                    {
                        "frame": frame_idx,
                        "time_s": round(t, 3),
                        "robot_xy_yaw": [round(float(robot_pos[0]), 4), round(float(robot_pos[1]), 4), round(float(robot_yaw), 4)],
                        "target_xy": [round(float(target_pos[0]), 4), round(float(target_pos[1]), 4)],
                        "visible": visible,
                        "distance_m": round(distance, 4),
                        "front_m": round(front, 4),
                        "lateral_m": round(lateral, 4),
                        "command_vx_vy_wz_stop": [round(float(x), 5) for x in pred[0]],
                    },
                )
    finally:
        proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    if not preview_path.exists():
        Image.open(video_path)  # pragma: no cover

    summary = {
        "created_unix": round(time.time(), 3),
        "policy": str(args.policy),
        "video": str(video_path),
        "preview": str(preview_path),
        "rollout": str(rollout_path),
        "seconds": args.seconds,
        "fps": fps,
        "frame_size": [width, height],
        "camera_shape_chw": [4, args.camera_height, args.camera_width],
        "target_motion": "moving target with random visible waypoints relative to the robot",
        "policy_inputs": ["rgb_lowres", "depth_closeness_lowres"],
        "policy_outputs": ["vx", "vy", "wz", "stop_probability"],
        "execution_stack": "visual command policy -> frozen 1.09B walker command interface",
        "limitation": "2D command-interface rollout, not full-body MuJoCo physics",
        "mean_distance_m": float(np.mean(distances)),
        "min_distance_m": float(np.min(distances)),
        "max_distance_m": float(np.max(distances)),
        "visible_fraction": float(visible_count / max(1, total_frames)),
        "mean_stop_probability": float(np.mean(stop_probs)),
    }
    write_json(summary_path, summary)
    return video_path, preview_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1091043328)
    parser.add_argument("--encoder", choices=("features", "raw"), default="features")
    parser.add_argument("--camera-width", type=int, default=64)
    parser.add_argument("--camera-height", type=int, default=36)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--target-speed", type=float, default=0.28)
    args = parser.parse_args()

    if not args.policy.exists():
        raise SystemExit(f"policy not found: {args.policy}")
    video_path, preview_path = run_rollout(args)
    print(json.dumps({"video": str(video_path), "preview": str(preview_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
