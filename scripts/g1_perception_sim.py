#!/usr/bin/env python3
"""Render a G1 head-camera perception sandbox with RGB + depth.

This is intentionally separate from the fast RL trainer. It uses MuJoCo only as a
perception sandbox: a virtual head-mounted camera sees a moving person stand-in,
depth is rendered from the same camera pose, and a small prompt-compatible
planner emits the walking command that would be sent to the existing RL command
wrapper.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import urllib.error
import urllib.request
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "perception-sim" / "model-mujoco39" / "g1.mjb"
DEFAULT_OUT = ROOT / "artifacts" / "perception-sim" / "latest"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
STOP_COMMAND = {"intent": "stop", "vx": 0.0, "vy": 0.0, "wz": 0.0}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    raise SystemExit(
        f"model not found: {model_path}\n"
        "Create it with:\n"
        f"  G1_MODEL_DIR={model_path.parent} .venv/bin/python tools/extract_g1_model.py\n"
        "A noncanonical MuJoCo byte fingerprint is acceptable for this perception sandbox."
    )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def camera_angles_from_pose(pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Return MuJoCo free-camera lookat/distance/azimuth/elevation."""
    vec = np.asarray(pos, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    distance = float(np.linalg.norm(vec))
    if distance < 1e-6:
        distance = 1e-6
        vec = np.array([-distance, 0.0, 0.0], dtype=np.float64)

    horiz = math.hypot(float(vec[0]), float(vec[1]))
    elevation = math.degrees(math.atan2(float(vec[2]), horiz))
    # MuJoCo's free camera stores the camera position around lookat as:
    # x = -d*cos(elev)*cos(az), y = -d*cos(elev)*sin(az), z = d*sin(elev).
    azimuth = math.degrees(math.atan2(float(-vec[1]), float(-vec[0])))
    return np.asarray(target, dtype=np.float64), distance, azimuth, elevation


def make_head_camera(
    torso_pos: np.ndarray,
    torso_mat: np.ndarray,
    offset: np.ndarray,
    lookahead_m: float,
    pitch_down_deg: float,
) -> mujoco.MjvCamera:
    cam_pos = torso_pos + torso_mat @ offset
    forward = torso_mat @ np.array([1.0, 0.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    target = cam_pos + lookahead_m * forward + math.tan(math.radians(pitch_down_deg)) * lookahead_m * down
    lookat, distance, azimuth, elevation = camera_angles_from_pose(cam_pos, target)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def add_person(scene: mujoco.MjvScene, pos: np.ndarray, rgba: tuple[float, float, float, float]) -> None:
    mat = np.eye(3, dtype=np.float64).ravel()
    geoms = [
        (mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([0.11, 0.55, 0.0]), pos + [0.0, 0.0, 0.75], rgba),
        (mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.16, 0.0, 0.0]), pos + [0.0, 0.0, 1.42], rgba),
    ]
    for geom_type, size, gpos, color in geoms:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            geom_type,
            size.astype(np.float64),
            np.asarray(gpos, dtype=np.float64),
            mat,
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1


def colorize_depth(depth: np.ndarray, near: float | None = None, far: float | None = None) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    if near is None:
        near = float(np.percentile(depth[valid], 2)) if np.any(valid) else 0.1
    if far is None:
        far = float(np.percentile(depth[valid], 98)) if np.any(valid) else 4.0
    if far <= near:
        far = near + 1e-3

    norm = np.clip((depth - near) / (far - near), 0.0, 1.0)
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    rgb[..., 0] = ((1.0 - norm) * 255.0).astype(np.uint8)
    rgb[..., 2] = (norm * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def estimate_person(robot_pos: np.ndarray, robot_yaw: float, person_pos: np.ndarray, fov_deg: float) -> dict[str, float | bool]:
    dx = float(person_pos[0] - robot_pos[0])
    dy = float(person_pos[1] - robot_pos[1])
    forward = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
    left = np.array([-math.sin(robot_yaw), math.cos(robot_yaw)])
    front = dx * forward[0] + dy * forward[1]
    lateral = dx * left[0] + dy * left[1]
    dist = math.hypot(dx, dy)
    bearing = math.atan2(lateral, front)
    visible = front > 0.0 and abs(math.degrees(bearing)) <= fov_deg / 2.0 and dist < 5.0
    return {
        "visible": visible,
        "distance_m": dist,
        "bearing_rad": bearing,
        "front_m": front,
        "lateral_m": lateral,
    }


def mock_llm_command(estimate: dict[str, float | bool], stop_distance: float) -> dict[str, float | str]:
    """Prompt-engineering stand-in for the future VLM/LLM call."""
    if not estimate["visible"]:
        return {"intent": "lost_person_stop", "vx": 0.0, "vy": 0.0, "wz": 0.0}
    if float(estimate["distance_m"]) <= stop_distance:
        return {"intent": "arrived_stop", "vx": 0.0, "vy": 0.0, "wz": 0.0}

    front = float(estimate["front_m"])
    lateral = float(estimate["lateral_m"])
    bearing = float(estimate["bearing_rad"])
    return {
        "intent": "follow_person",
        "vx": float(np.clip(0.55 * (front - stop_distance), 0.0, 0.35)),
        "vy": float(np.clip(0.45 * lateral, -0.18, 0.18)),
        "wz": float(np.clip(0.85 * bearing, -0.45, 0.45)),
    }


def normalize_command(raw: dict[str, object]) -> dict[str, float | str | bool]:
    intent = str(raw.get("intent", "follow_person"))
    command: dict[str, float | str | bool] = {
        "intent": intent,
        "vx": float(np.clip(float(raw.get("vx", 0.0)), -0.15, 0.45)),
        "vy": float(np.clip(float(raw.get("vy", 0.0)), -0.25, 0.25)),
        "wz": float(np.clip(float(raw.get("wz", 0.0)), -0.65, 0.65)),
    }
    for key in ("target_visible", "distance_m", "bearing_rad", "reason"):
        if key in raw:
            command[key] = raw[key]  # type: ignore[assignment]
    return command


def image_only_command(rgb: np.ndarray, depth_rgb: np.ndarray, fov_deg: float, stop_distance: float) -> tuple[dict[str, float | str | bool], dict[str, float | bool]]:
    """Camera-only local planner used to test the VLM boundary without an API call."""
    del stop_distance
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    blue_target = (b > 120.0) & (b > 1.45 * r) & (b > 1.2 * g)

    h, w = blue_target.shape
    min_area = max(80, int(0.0015 * h * w))
    ys, xs = np.nonzero(blue_target)
    if xs.size < min_area:
        estimate = {"visible": False, "distance_m": 99.0, "bearing_rad": 0.0, "front_m": 0.0, "lateral_m": 0.0}
        return normalize_command({"intent": "lost_person_stop", "vx": 0.0, "vy": 0.0, "wz": 0.0, "target_visible": False, "reason": "no blue target in RGB frame"}), estimate

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = float(xs.mean())
    bbox_h = max(1.0, float(y1 - y0 + 1))
    apparent_height = bbox_h / float(h)
    x_offset = (cx - (w / 2.0)) / (w / 2.0)
    bearing = -x_offset * math.radians(fov_deg / 2.0)

    depth_target = depth_rgb[blue_target].astype(np.float32)
    red = float(depth_target[:, 0].mean())
    blue = float(depth_target[:, 2].mean())
    closeness = red / max(1.0, red + blue)

    # This is not world geometry. It is a coarse image-space proxy for "too far"
    # based on target size plus the red/blue depth visualization.
    desired_height = 0.62
    depth_far_bonus = float(np.clip(0.65 - closeness, 0.0, 0.3))
    forward_error = desired_height - apparent_height + depth_far_bonus
    vx = float(np.clip(0.78 * forward_error, 0.0, 0.35))
    vy = float(np.clip(0.10 * bearing, -0.10, 0.10))
    wz = float(np.clip(1.05 * bearing, -0.45, 0.45))

    estimate = {
        "visible": True,
        "distance_m": float(np.clip(1.45 - apparent_height + (1.0 - closeness), 0.25, 5.0)),
        "bearing_rad": bearing,
        "front_m": float(np.clip(forward_error + 0.7, 0.0, 5.0)),
        "lateral_m": float(x_offset),
    }
    return normalize_command({
        "intent": "follow_person" if vx > 0.01 or abs(wz) > 0.03 else "hold_person_centered",
        "vx": vx,
        "vy": vy,
        "wz": wz,
        "target_visible": True,
        "distance_m": estimate["distance_m"],
        "bearing_rad": estimate["bearing_rad"],
        "reason": "blue target located from RGB; distance estimated from apparent size and depth colors",
    }), estimate


def image_data_uri(frame: np.ndarray, *, quality: int = 80) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def response_text(payload: dict[str, object]) -> str:
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"])
    for item in payload.get("output", []):  # type: ignore[union-attr]
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                return str(content.get("text", ""))
    raise RuntimeError(f"OpenAI response did not contain output_text: {payload}")


def openai_vlm_command(
    rgb: np.ndarray,
    depth_rgb: np.ndarray,
    *,
    model: str,
    api_key_env: str,
    base_url: str,
    timeout_s: float,
) -> dict[str, float | str | bool]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set; use --planner vision-heuristic for local camera-only testing")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string"},
            "vx": {"type": "number", "description": "forward velocity command in m/s, stop=0"},
            "vy": {"type": "number", "description": "lateral velocity command in m/s"},
            "wz": {"type": "number", "description": "yaw rate command in rad/s"},
            "target_visible": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["intent", "vx", "vy", "wz", "target_visible", "reason"],
    }
    prompt = (
        "You are the high-level planner for a Unitree G1 walking policy. "
        "You receive only two camera views: image 1 is RGB, image 2 is depth where red means close and blue means far. "
        "Goal: follow the blue person-like target, keep roughly 0.6m clearance, and stop if the target is not clear or the view looks unsafe. "
        "Return a cautious command for the RL walking wrapper. Limits: vx in [0,0.35], vy in [-0.18,0.18], wz in [-0.45,0.45]. "
        "Do not infer or use world coordinates."
    )
    body = {
        "model": model,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_data_uri(rgb), "detail": "low"},
                {"type": "input_image", "image_url": image_data_uri(depth_rgb), "detail": "low"},
            ],
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "g1_walk_command",
                "strict": True,
                "schema": schema,
            }
        },
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI VLM request failed: HTTP {exc.code}: {detail}") from exc

    return normalize_command(json.loads(response_text(result)))


def annotate(frame: np.ndarray, command: dict[str, float | str | bool], estimate: dict[str, float | bool], planner: str) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, 0, img.width, 92), fill=(8, 10, 14, 185))
    d.text((20, 14), f"G1 simulated head cameras: RGB + depth  |  planner={planner}", font=font(28, True), fill=(255, 255, 255, 255))
    d.text(
        (20, 52),
        (
            f"person_visible={int(bool(estimate['visible']))}  "
            f"dist={float(estimate['distance_m']):.2f}m  "
            f"bearing={float(estimate['bearing_rad']):+.2f}rad  "
            f"cmd=({float(command['vx']):+.2f},{float(command['vy']):+.2f},{float(command['wz']):+.2f})"
        ),
        font=font(21),
        fill=(218, 226, 236, 255),
    )
    d.rectangle((0, img.height - 54, img.width, img.height), fill=(8, 10, 14, 185))
    d.text((20, img.height - 39), "left: RGB camera  |  right: depth red=close blue=far  |  command goes to RL walking wrapper", font=font(22), fill=(255, 255, 255, 255))
    return np.asarray(img)


def write_video(frames: list[np.ndarray], out: Path, fps: int) -> None:
    if not frames:
        raise RuntimeError("no frames to write")
    h, w = frames[0].shape[:2]
    raw = out.with_suffix(".rgb")
    try:
        raw.write_bytes(b"".join(np.ascontiguousarray(f).astype(np.uint8).tobytes() for f in frames))
        run([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{w}x{h}", "-framerate", str(fps),
            "-i", str(raw),
            "-vf", "format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            str(out),
        ])
    finally:
        raw.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--person-x", type=float, default=2.2)
    parser.add_argument("--person-y", type=float, default=0.55)
    parser.add_argument("--person-vx", type=float, default=-0.04)
    parser.add_argument("--person-vy", type=float, default=-0.08)
    parser.add_argument("--robot-yaw", type=float, default=0.0)
    parser.add_argument("--stop-distance", type=float, default=0.6)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--camera-offset", type=float, nargs=3, default=(0.16, 0.0, 0.46), metavar=("X", "Y", "Z"))
    parser.add_argument("--lookahead-m", type=float, default=2.4)
    parser.add_argument("--pitch-down-deg", type=float, default=7.0)
    parser.add_argument(
        "--planner",
        choices=("oracle", "vision-heuristic", "openai-vlm"),
        default="oracle",
        help="oracle uses hidden sim state; vision-heuristic and openai-vlm receive only RGB/depth frames",
    )
    parser.add_argument("--planner-period", type=int, default=15, help="frames between openai-vlm planner calls")
    parser.add_argument("--vlm-model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--openai-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    ensure_model(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    if model.nkey and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home") >= 0:
        data.qpos[:] = model.key_qpos[model.key("home").id]
    else:
        data.qpos[:] = model.qpos0
    data.qpos[3:7] = yaw_to_quat(args.robot_yaw)
    mujoco.mj_forward(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id < 0:
        torso_id = 1

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    log: list[dict[str, object]] = []
    person_rgba = (0.08, 0.36, 1.0, 1.0)
    last_command: dict[str, float | str | bool] = dict(STOP_COMMAND)
    last_estimate: dict[str, float | bool] = {"visible": False, "distance_m": 99.0, "bearing_rad": 0.0, "front_m": 0.0, "lateral_m": 0.0}
    planner_input = ["sim_robot_pose", "sim_person_xyz"] if args.planner == "oracle" else ["rgb", "depth_rgb"]

    steps = max(1, int(round(args.seconds * args.fps)))
    for i in range(steps):
        t = i / args.fps
        person_pos = np.array([args.person_x + args.person_vx * t, args.person_y + args.person_vy * t, 0.0], dtype=np.float64)
        torso_pos = np.array(data.xpos[torso_id], dtype=np.float64)
        torso_mat = np.array(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
        cam = make_head_camera(torso_pos, torso_mat, np.asarray(args.camera_offset, dtype=np.float64), args.lookahead_m, args.pitch_down_deg)

        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        add_person(renderer.scene, person_pos, person_rgba)
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        add_person(renderer.scene, person_pos, person_rgba)
        depth = renderer.render()
        renderer.disable_depth_rendering()

        depth_rgb = colorize_depth(depth)

        if args.planner == "oracle":
            estimate = estimate_person(np.array([data.qpos[0], data.qpos[1], 0.0]), args.robot_yaw, person_pos, args.fov_deg)
            command = normalize_command(mock_llm_command(estimate, args.stop_distance))
        elif args.planner == "vision-heuristic":
            command, estimate = image_only_command(rgb, depth_rgb, args.fov_deg, args.stop_distance)
        else:
            estimate = last_estimate
            if i == 0 or (args.planner_period > 0 and i % args.planner_period == 0):
                command = openai_vlm_command(
                    rgb,
                    depth_rgb,
                    model=args.vlm_model,
                    api_key_env=args.openai_api_key_env,
                    base_url=args.openai_base_url,
                    timeout_s=args.openai_timeout_s,
                )
                estimate = {
                    "visible": bool(command.get("target_visible", False)),
                    "distance_m": float(command.get("distance_m", 0.0)) if "distance_m" in command else 0.0,
                    "bearing_rad": float(command.get("bearing_rad", 0.0)) if "bearing_rad" in command else 0.0,
                    "front_m": 0.0,
                    "lateral_m": 0.0,
                }
                last_command = command
                last_estimate = estimate
            else:
                command = last_command

        rgb_frames.append(rgb)
        depth_frames.append(depth_rgb)
        combined = np.concatenate([rgb, depth_rgb], axis=1)
        frames.append(annotate(combined, command, estimate, args.planner))
        if i == 0:
            Image.fromarray(rgb).save(args.out / "rgb_000.png")
            Image.fromarray(depth_rgb).save(args.out / "depth_000.png")
            Image.fromarray(frames[-1]).save(args.out / "rgb_depth_000.png")

        log.append({
            "frame": i,
            "t": t,
            "ground_truth_person_xyz": [float(x) for x in person_pos],
            "planner": args.planner,
            "planner_inputs": planner_input,
            "camera_offset_torso_xyz": [float(x) for x in args.camera_offset],
            "estimate": estimate,
            "command": command,
        })

    write_video(frames, args.out / "g1_head_rgb_depth_follow.mp4", args.fps)
    write_video(rgb_frames, args.out / "head_rgb.mp4", args.fps)
    write_video(depth_frames, args.out / "head_depth_red_close_blue_far.mp4", args.fps)
    (args.out / "commands.jsonl").write_text("\n".join(json.dumps(row) for row in log) + "\n")
    (args.out / "summary.json").write_text(json.dumps({
        "model": str(args.model),
        "video": str(args.out / "g1_head_rgb_depth_follow.mp4"),
        "rgb_video": str(args.out / "head_rgb.mp4"),
        "depth_video": str(args.out / "head_depth_red_close_blue_far.mp4"),
        "frames": len(frames),
        "fps": args.fps,
        "seconds": args.seconds,
        "camera_mount": "torso_link + offset, standing in for G1 head camera",
        "planner": args.planner,
        "planner_inputs": planner_input,
        "vlm_model": args.vlm_model if args.planner == "openai-vlm" else None,
    }, indent=2) + "\n")
    print(f"wrote {args.out / 'g1_head_rgb_depth_follow.mp4'}")
    print(f"wrote {args.out / 'commands.jsonl'}")


if __name__ == "__main__":
    main()
