#!/usr/bin/env python3
"""Train a small native RGB/depth -> walking-command policy.

This does not retrain the G1 locomotion network. The intended deployment stack is:

    RGB/depth visual command policy -> vx/vy/wz/stop -> frozen nanoG1 walker

The trainer is dependency-light on purpose so it can run in the current repo
environment. It uses a synthetic low-resolution camera raster and an oracle
teacher as the first bootstrap stage; the frozen walker checkpoint is recorded as
the base controller that will execute the learned commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_WALKER = ROOT / "artifacts" / "protected" / "2026-06-21" / "1091043328" / "checkpoint.bin"
DEFAULT_OUT = ROOT / "artifacts" / "visual-command" / time.strftime("%Y-%m-%d") / time.strftime("%H%M%S")


@dataclass
class Batch:
    obs: np.ndarray
    target: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def save_policy(params: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{time.time_ns()}.tmp.npz")
    np.savez(tmp, **params)
    tmp.replace(path)


def load_policy(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def policy_score(metrics: dict[str, float]) -> float:
    return float(metrics["stop_accuracy"] - metrics["velocity_mae"])


def write_heartbeat(out: Path, payload: dict[str, object]) -> None:
    event = {"time_unix": round(time.time(), 3), **payload}
    write_json_atomic(out / "visual_training_heartbeat.json", event)
    append_jsonl(out / "visual_training_heartbeat.jsonl", event)


def render_observation(
    front: np.ndarray,
    lateral: np.ndarray,
    visible: np.ndarray,
    *,
    width: int,
    height: int,
    fov_deg: float,
) -> np.ndarray:
    """Render low-res RGB + depth-like camera observations.

    Channels are RGB plus depth closeness. The blue capsule is the person stand-in.
    Depth is represented as closeness in [0, 1], where 1 means near.
    """
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


def oracle_command(front: np.ndarray, lateral: np.ndarray, visible: np.ndarray, stop_distance: float) -> np.ndarray:
    dist = np.sqrt(front * front + lateral * lateral)
    bearing = np.arctan2(lateral, np.maximum(front, 1e-3))
    should_stop = (~visible) | (dist <= stop_distance)
    vx = np.clip(0.55 * (front - stop_distance), 0.0, 0.35)
    vy = np.clip(0.45 * lateral, -0.18, 0.18)
    wz = np.clip(0.85 * bearing, -0.45, 0.45)
    vx[should_stop] = 0.0
    vy[should_stop] = 0.0
    wz[should_stop] = 0.0
    return np.stack([vx, vy, wz, should_stop.astype(np.float32)], axis=1).astype(np.float32)


def sample_static_batch(rng: np.random.Generator, batch_size: int, width: int, height: int, fov_deg: float, stop_distance: float) -> Batch:
    front = rng.uniform(0.25, 3.8, size=batch_size).astype(np.float32)
    lateral = rng.uniform(-1.8, 1.8, size=batch_size).astype(np.float32)
    bearing = np.degrees(np.arctan2(lateral, np.maximum(front, 1e-3)))
    visible = (front > 0.0) & (np.abs(bearing) < fov_deg / 2.0) & (rng.random(batch_size) > 0.05)
    lost = rng.random(batch_size) < 0.08
    visible[lost] = False
    obs = render_observation(front, lateral, visible, width=width, height=height, fov_deg=fov_deg)
    target = oracle_command(front, lateral, visible, stop_distance)
    noise = rng.normal(0.0, 0.015, size=obs.shape).astype(np.float32)
    return Batch(np.clip(obs + noise, 0.0, 1.0), target)


def sample_moving_batch(
    rng: np.random.Generator,
    batch_size: int,
    width: int,
    height: int,
    fov_deg: float,
    stop_distance: float,
    trajectory_len: int,
) -> Batch:
    """Sample frames from moving random target trajectories.

    The policy still receives a single camera/depth frame at a time. Motion is
    represented in the data distribution: random starts, velocities, turns,
    bounces, and target-loss events produce varied current views.
    """
    trajectory_len = max(2, trajectory_len)
    n_traj = int(math.ceil(batch_size / trajectory_len))
    front = rng.uniform(0.45, 3.8, size=n_traj).astype(np.float32)
    lateral = rng.uniform(-1.5, 1.5, size=n_traj).astype(np.float32)
    vf = rng.uniform(-0.045, 0.035, size=n_traj).astype(np.float32)
    vl = rng.uniform(-0.055, 0.055, size=n_traj).astype(np.float32)
    frames_front: list[np.ndarray] = []
    frames_lateral: list[np.ndarray] = []
    frames_visible: list[np.ndarray] = []

    for t in range(trajectory_len):
        if t > 0:
            turn = rng.random(n_traj) < 0.10
            vf[turn] += rng.uniform(-0.035, 0.035, size=int(turn.sum())).astype(np.float32)
            vl[turn] += rng.uniform(-0.045, 0.045, size=int(turn.sum())).astype(np.float32)
            vf[:] = np.clip(vf, -0.08, 0.06)
            vl[:] = np.clip(vl, -0.10, 0.10)
            front += vf + rng.normal(0.0, 0.008, size=n_traj).astype(np.float32)
            lateral += vl + rng.normal(0.0, 0.012, size=n_traj).astype(np.float32)

            too_close = front < 0.25
            too_far = front > 4.0
            too_left = lateral < -1.95
            too_right = lateral > 1.95
            vf[too_close | too_far] *= -0.7
            vl[too_left | too_right] *= -0.7
            front[:] = np.clip(front, 0.25, 4.0)
            lateral[:] = np.clip(lateral, -1.95, 1.95)

        bearing = np.degrees(np.arctan2(lateral, np.maximum(front, 1e-3)))
        visible = (front > 0.0) & (np.abs(bearing) < fov_deg / 2.0)
        visible &= rng.random(n_traj) > 0.04
        # Occlusion / target leaves view for short intervals.
        visible &= ~((rng.random(n_traj) < 0.06) & (t % 3 != 0))
        frames_front.append(front.copy())
        frames_lateral.append(lateral.copy())
        frames_visible.append(visible.copy())

    flat_front = np.concatenate(frames_front)[:batch_size].astype(np.float32)
    flat_lateral = np.concatenate(frames_lateral)[:batch_size].astype(np.float32)
    flat_visible = np.concatenate(frames_visible)[:batch_size]
    order = rng.permutation(flat_front.shape[0])
    flat_front = flat_front[order]
    flat_lateral = flat_lateral[order]
    flat_visible = flat_visible[order]
    obs = render_observation(flat_front, flat_lateral, flat_visible, width=width, height=height, fov_deg=fov_deg)
    target = oracle_command(flat_front, flat_lateral, flat_visible, stop_distance)
    noise = rng.normal(0.0, 0.015, size=obs.shape).astype(np.float32)
    return Batch(np.clip(obs + noise, 0.0, 1.0), target)


def sample_batch(rng: np.random.Generator, batch_size: int, args: argparse.Namespace) -> Batch:
    if args.target_mode == "static":
        return sample_static_batch(rng, batch_size, args.width, args.height, args.fov_deg, args.stop_distance)
    if args.target_mode == "moving":
        return sample_moving_batch(rng, batch_size, args.width, args.height, args.fov_deg, args.stop_distance, args.trajectory_len)
    if args.target_mode == "mixed":
        moving_count = int(round(batch_size * args.moving_fraction))
        static_count = batch_size - moving_count
        batches = []
        if moving_count:
            batches.append(sample_moving_batch(rng, moving_count, args.width, args.height, args.fov_deg, args.stop_distance, args.trajectory_len))
        if static_count:
            batches.append(sample_static_batch(rng, static_count, args.width, args.height, args.fov_deg, args.stop_distance))
        obs = np.concatenate([b.obs for b in batches], axis=0)
        target = np.concatenate([b.target for b in batches], axis=0)
        order = rng.permutation(obs.shape[0])
        return Batch(obs[order], target[order])
    raise ValueError(f"unknown target mode: {args.target_mode}")


def features_from_observation(obs: np.ndarray) -> np.ndarray:
    """Extract native camera/depth features without simulator coordinates."""
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


def encode_obs(obs: np.ndarray, encoder: str) -> np.ndarray:
    if encoder == "features":
        return features_from_observation(obs)
    if encoder == "raw":
        return obs.reshape(obs.shape[0], -1)
    raise ValueError(f"unknown encoder: {encoder}")


def init_policy(rng: np.random.Generator, input_dim: int, hidden_dim: int) -> dict[str, np.ndarray]:
    return {
        "w1": (rng.normal(0.0, 1.0 / math.sqrt(input_dim), size=(input_dim, hidden_dim))).astype(np.float32),
        "b1": np.zeros((hidden_dim,), dtype=np.float32),
        "w2": (rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, 4))).astype(np.float32),
        "b2": np.zeros((4,), dtype=np.float32),
    }


def forward(params: dict[str, np.ndarray], obs: np.ndarray, encoder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = encode_obs(obs, encoder)
    pre = x @ params["w1"] + params["b1"]
    h = np.maximum(pre, 0.0)
    out = h @ params["w2"] + params["b2"]
    out[:, 0] = 0.35 / (1.0 + np.exp(-out[:, 0]))
    out[:, 1] = 0.18 * np.tanh(out[:, 1])
    out[:, 2] = 0.45 * np.tanh(out[:, 2])
    out[:, 3] = 1.0 / (1.0 + np.exp(-out[:, 3]))
    return out, h, pre, x


def train_step(params: dict[str, np.ndarray], batch: Batch, lr: float, encoder: str) -> dict[str, float]:
    pred, h, pre, x = forward(params, batch.obs, encoder)
    err = pred - batch.target
    loss_v = float(np.mean(err[:, :3] ** 2))
    loss_stop = float(np.mean(err[:, 3] ** 2))
    loss = loss_v + 0.4 * loss_stop

    grad = np.zeros_like(pred)
    grad[:, :3] = (2.0 / pred.shape[0]) * err[:, :3]
    grad[:, 3] = (0.8 / pred.shape[0]) * err[:, 3]
    grad[:, 0] *= pred[:, 0] * (1.0 - pred[:, 0] / 0.35)
    grad[:, 1] *= 0.18 * (1.0 - (pred[:, 1] / 0.18) ** 2)
    grad[:, 2] *= 0.45 * (1.0 - (pred[:, 2] / 0.45) ** 2)
    grad[:, 3] *= pred[:, 3] * (1.0 - pred[:, 3])

    dw2 = h.T @ grad
    db2 = grad.sum(axis=0)
    dh = grad @ params["w2"].T
    dpre = dh * (pre > 0.0)
    dw1 = x.T @ dpre
    db1 = dpre.sum(axis=0)

    for key, update in (("w1", dw1), ("b1", db1), ("w2", dw2), ("b2", db2)):
        params[key] -= lr * np.clip(update, -1.0, 1.0).astype(np.float32)
    return {"loss": loss, "loss_velocity": loss_v, "loss_stop": loss_stop}


def evaluate(params: dict[str, np.ndarray], rng: np.random.Generator, args: argparse.Namespace) -> dict[str, float]:
    batch = sample_batch(rng, args.eval_batch_size, args)
    pred, _, _, _ = forward(params, batch.obs, args.encoder)
    velocity_mae = float(np.mean(np.abs(pred[:, :3] - batch.target[:, :3])))
    pred_stop = pred[:, 3] > 0.5
    true_stop = batch.target[:, 3] > 0.5
    stop_acc = float(np.mean(pred_stop == true_stop))
    return {"velocity_mae": velocity_mae, "stop_accuracy": stop_acc}


def corrupt_observation(obs: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "normal":
        return obs.copy()
    if mode == "blank":
        out = obs.copy()
        out[:, 0:3, :, :] = 0.5
        out[:, 3, :, :] = 0.05
        return out
    if mode == "target_removed":
        out = obs.copy()
        rgb = out[:, :3]
        mask = (rgb[:, 2] > 0.68) & (rgb[:, 2] > rgb[:, 0] + 0.20) & (rgb[:, 2] > rgb[:, 1] + 0.20)
        out[:, 0, :, :][mask] = 0.92
        out[:, 1, :, :][mask] = 0.92
        out[:, 2, :, :][mask] = 0.92
        out[:, 3, :, :][mask] = 0.05
        return out
    if mode == "shuffle_batch":
        return obs[rng.permutation(obs.shape[0])].copy()
    if mode == "shuffle_pixels":
        out = obs.copy()
        n, c, h, w = out.shape
        for i in range(n):
            order = rng.permutation(h * w)
            out[i] = out[i].reshape(c, h * w)[:, order].reshape(c, h, w)
        return out
    raise ValueError(f"unknown corruption mode: {mode}")


def evaluate_sensitivity(params: dict[str, np.ndarray], rng: np.random.Generator, args: argparse.Namespace) -> dict[str, object]:
    batch = sample_batch(rng, args.eval_batch_size, args)
    normal_pred, _, _, _ = forward(params, batch.obs, args.encoder)
    result: dict[str, object] = {}
    for mode in ["normal", "blank", "target_removed", "shuffle_batch", "shuffle_pixels"]:
        obs = corrupt_observation(batch.obs, mode, rng)
        pred, _, _, _ = forward(params, obs, args.encoder)
        velocity_mae = float(np.mean(np.abs(pred[:, :3] - batch.target[:, :3])))
        stop_acc = float(np.mean((pred[:, 3] > 0.5) == (batch.target[:, 3] > 0.5)))
        command_delta = float(np.mean(np.abs(pred - normal_pred)))
        mean_command = [float(x) for x in pred.mean(axis=0)]
        result[mode] = {
            "velocity_mae": velocity_mae,
            "stop_accuracy": stop_acc,
            "mean_abs_command_delta_vs_normal": command_delta,
            "mean_command_vx_vy_wz_stop": mean_command,
        }
    return result


def save_preview(out: Path, args: argparse.Namespace) -> None:
    front = np.array([2.2], dtype=np.float32)
    lateral = np.array([0.55], dtype=np.float32)
    visible = np.array([True])
    obs = render_observation(front, lateral, visible, width=args.width, height=args.height, fov_deg=args.fov_deg)[0]
    rgb = np.moveaxis(obs[:3], 0, -1)
    depth = np.zeros_like(rgb)
    depth[..., 0] = obs[3]
    depth[..., 2] = 1.0 - obs[3]
    frame = np.concatenate([rgb, depth], axis=1)
    Image.fromarray((frame * 255).astype(np.uint8)).resize((args.width * 6, args.height * 3)).save(out / "preview_rgb_depth.png")


def save_moving_preview(out: Path, args: argparse.Namespace) -> None:
    steps = 24
    t = np.arange(steps, dtype=np.float32)
    front = np.clip(3.4 - 0.08 * t + 0.18 * np.sin(t * 0.45), 0.35, 3.8)
    lateral = np.clip(-1.25 + 0.11 * t + 0.25 * np.sin(t * 0.65), -1.75, 1.75)
    bearing = np.degrees(np.arctan2(lateral, np.maximum(front, 1e-3)))
    visible = (front > 0.0) & (np.abs(bearing) < args.fov_deg / 2.0)
    obs_seq = render_observation(front, lateral, visible, width=args.width, height=args.height, fov_deg=args.fov_deg)
    noise = np.random.default_rng(args.seed + 999).normal(0.0, 0.01, size=obs_seq.shape).astype(np.float32)
    obs_seq = np.clip(obs_seq + noise, 0.0, 1.0)
    frames = []
    for obs in obs_seq:
        rgb = np.moveaxis(obs[:3], 0, -1)
        depth = np.zeros_like(rgb)
        depth[..., 0] = obs[3]
        depth[..., 2] = 1.0 - obs[3]
        frames.append(np.concatenate([rgb, depth], axis=1))
    rows = []
    for i in range(0, len(frames), 6):
        rows.append(np.concatenate(frames[i:i + 6], axis=1))
    strip = np.concatenate(rows, axis=0)
    Image.fromarray((strip * 255).astype(np.uint8)).resize((args.width * 12, args.height * 8)).save(out / "preview_moving_targets.png")


def training_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_walker": str(args.base_walker),
        "base_walker_sha256": sha256(args.base_walker),
        "base_walker_steps": 1091043328,
        "batch_size": args.batch_size,
        "encoder": args.encoder,
        "eval_batch_size": args.eval_batch_size,
        "fov_deg": args.fov_deg,
        "height": args.height,
        "hidden_dim": args.hidden_dim,
        "image_shape_chw": [4, args.height, args.width],
        "learning_rate": args.learning_rate,
        "moving_fraction": args.moving_fraction,
        "planner_inputs": ["rgb_lowres", "depth_closeness_lowres"],
        "policy_outputs": ["vx", "vy", "wz", "stop_probability"],
        "seed": args.seed,
        "stop_distance": args.stop_distance,
        "target_mode": args.target_mode,
        "trajectory_len": args.trajectory_len,
        "width": args.width,
    }


def save_training_checkpoint(
    out: Path,
    params: dict[str, np.ndarray],
    step: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    *,
    reason: str,
    best: bool,
) -> None:
    checkpoint_dir = out / "checkpoints"
    checkpoint = checkpoint_dir / f"step_{step:09d}.npz"
    save_policy(params, checkpoint)
    latest = out / "latest_policy.npz"
    shutil.copy2(checkpoint, latest)
    if best:
        shutil.copy2(checkpoint, out / "best_policy.npz")

    payload: dict[str, object] = {
        "step": step,
        "checkpoint": str(checkpoint),
        "latest_policy": str(latest),
        "best_policy": str(out / "best_policy.npz") if (best or (out / "best_policy.npz").exists()) else None,
        "reason": reason,
        "score": policy_score(metrics),
        "metrics": metrics,
        "base_walker": str(args.base_walker),
        "base_walker_sha256": sha256(args.base_walker),
        "time_unix": round(time.time(), 3),
    }
    write_json_atomic(out / "latest_checkpoint.json", payload)
    append_jsonl(out / "checkpoints.jsonl", payload)


def write_summary(out: Path, args: argparse.Namespace, *, final_step: int, best_metrics: dict[str, float] | None, stop_reason: str) -> None:
    summary = {
        "created_unix": round(time.time(), 3),
        "base_walker": str(args.base_walker),
        "base_walker_sha256": sha256(args.base_walker),
        "base_walker_steps": 1091043328,
        "policy": str(out / "latest_policy.npz"),
        "best_policy": str(out / "best_policy.npz"),
        "training_type": "native visual command bootstrap",
        "target_mode": args.target_mode,
        "trajectory_len": args.trajectory_len,
        "moving_fraction": args.moving_fraction,
        "planner_inputs": ["rgb_lowres", "depth_closeness_lowres"],
        "encoder": args.encoder,
        "encoder_inputs": ["rgb_lowres", "depth_closeness_lowres"],
        "encoder_outputs": [
            "target_visible",
            "target_x_offset",
            "target_height_frac",
            "target_area_frac",
            "target_depth_closeness",
            "target_y_offset",
        ],
        "policy_outputs": ["vx", "vy", "wz", "stop_probability"],
        "execution_stack": "visual command policy -> frozen 1.09B nanoG1 walker -> joints",
        "final_step": final_step,
        "best_metrics": best_metrics,
        "batch_size": args.batch_size,
        "image_shape_chw": [4, args.height, args.width],
        "stop_reason": stop_reason,
        "note": "This trainer learns camera/depth-to-command. It does not modify the frozen locomotion checkpoint.",
    }
    write_json_atomic(out / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-walker", type=Path, default=DEFAULT_BASE_WALKER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder", choices=("features", "raw"), default="features")
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-distance", type=float, default=0.65)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--target-mode", choices=("static", "moving", "mixed"), default="moving")
    parser.add_argument("--trajectory-len", type=int, default=16)
    parser.add_argument("--moving-fraction", type=float, default=0.75)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--heartbeat-interval", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--target-stop-accuracy", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="Resume from OUT/latest_policy.npz and latest_checkpoint.json")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from a specific .npz policy")
    parser.add_argument("--eval-only", type=Path, default=None, help="Evaluate a saved .npz policy and exit")
    parser.add_argument("--sensitivity-eval", type=Path, default=None, help="Evaluate a policy with blank/shuffled/target-removed camera inputs")
    args = parser.parse_args()

    if not args.base_walker.exists():
        raise SystemExit(f"base walker checkpoint not found: {args.base_walker}")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    input_dim = 6 if args.encoder == "features" else 4 * args.width * args.height

    if args.eval_only:
        params = load_policy(args.eval_only)
        metrics = evaluate(params, rng, args)
        payload = {"policy": str(args.eval_only), **metrics}
        print(json.dumps(payload, sort_keys=True), flush=True)
        write_json_atomic(args.out / "eval_only.json", payload)
        return

    if args.sensitivity_eval:
        params = load_policy(args.sensitivity_eval)
        payload = {
            "policy": str(args.sensitivity_eval),
            "eval_batch_size": args.eval_batch_size,
            "encoder": args.encoder,
            "results": evaluate_sensitivity(params, rng, args),
        }
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        write_json_atomic(args.out / "sensitivity_eval.json", payload)
        return

    start_step = 0
    resume_path = args.resume_from
    if args.resume and resume_path is None:
        resume_path = args.out / "latest_policy.npz"
    if resume_path:
        if not resume_path.exists():
            raise SystemExit(f"resume policy not found: {resume_path}")
        params = load_policy(resume_path)
        latest = args.out / "latest_checkpoint.json"
        if latest.exists():
            start_step = int(json.loads(latest.read_text()).get("step", 0) or 0)
    else:
        params = init_policy(rng, input_dim, args.hidden_dim)

    metrics_path = args.out / "metrics.jsonl"
    write_json_atomic(args.out / "config.json", training_config(args))
    write_heartbeat(args.out, {"event": "visual_training_start", "step": start_step, "target_steps": args.steps})
    best_score = -1e9
    best_metrics: dict[str, float] | None = None
    final_step = start_step
    stop_reason = "completed"
    started = time.time()

    for step in range(start_step + 1, args.steps + 1):
        batch = sample_batch(rng, args.batch_size, args)
        train_metrics = train_step(params, batch, args.learning_rate, args.encoder)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            eval_metrics = evaluate(params, rng, args)
            row = {"step": step, **train_metrics, **eval_metrics}
            append_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            score = policy_score(eval_metrics)
            if score > best_score:
                best_score = score
                best_metrics = eval_metrics
                save_training_checkpoint(args.out, params, step, eval_metrics, args, reason="best", best=True)
            if args.target_stop_accuracy and eval_metrics["stop_accuracy"] >= args.target_stop_accuracy:
                stop_reason = f"target_stop_accuracy_reached:{args.target_stop_accuracy}"
                final_step = step
                break
        if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            eval_metrics = evaluate(params, rng, args)
            save_training_checkpoint(args.out, params, step, eval_metrics, args, reason="periodic", best=False)
        if args.heartbeat_interval > 0 and step % args.heartbeat_interval == 0:
            write_heartbeat(args.out, {"event": "visual_training_progress", "step": step})
        if args.max_seconds > 0 and time.time() - started >= args.max_seconds:
            stop_reason = f"max_seconds_reached:{args.max_seconds}"
            final_step = step
            break
        final_step = step

    final_metrics = evaluate(params, rng, args)
    save_training_checkpoint(args.out, params, final_step, final_metrics, args, reason="final", best=False)
    save_policy(params, args.out / "visual_command_policy.npz")
    save_preview(args.out, args)
    save_moving_preview(args.out, args)
    write_summary(args.out, args, final_step=final_step, best_metrics=best_metrics, stop_reason=stop_reason)
    write_heartbeat(args.out, {"event": "visual_training_done", "step": final_step, "stop_reason": stop_reason})
    print(f"wrote {args.out / 'visual_command_policy.npz'}")
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
