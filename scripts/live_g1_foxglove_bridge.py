#!/usr/bin/env python3
"""Foxglove live bridge for the G1 follow perception dry-run.

Runs on the Mac. It publishes the front RGB image, colorized RealSense depth,
image annotations, and follow/safety state over a local Foxglove WebSocket
server. It does not publish Unitree motion commands.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import foxglove
import numpy as np
from foxglove import Channel
from foxglove.channels import CompressedImageChannel, ImageAnnotationsChannel
from foxglove.websocket import Capability
from foxglove.messages import (
    CircleAnnotation,
    Color,
    CompressedImage,
    ImageAnnotations,
    Point2,
    PointsAnnotation,
    PointsAnnotationType,
    TextAnnotation,
    Timestamp,
)
from PIL import Image
from ultralytics import FastSAM, YOLO

from live_g1_follow_dashboard import (
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_SEGMENTER_MODEL,
    FFMPEG,
    SSH,
    RawFrameSource,
    SnapshotFrameSource,
    apply_safety_gate,
    choose_policy_estimate,
    colorize_depth,
    command_from_estimate,
    discover_depth_device,
    discover_rgb_device,
    estimate_closest_scene_object,
    estimate_from_depth_target,
    estimate_from_person,
    load_policy,
    select_trace_mask,
)


def timestamp_from_ns(now_ns: int) -> Timestamp:
    return Timestamp(now_ns // 1_000_000_000, now_ns % 1_000_000_000)


def encode_image(image: np.ndarray, fmt: str, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    pil = Image.fromarray(image)
    if fmt == "jpeg":
        pil.save(buf, format="JPEG", quality=quality)
    elif fmt == "png":
        pil.save(buf, format="PNG")
    else:
        raise ValueError(f"unsupported image format {fmt}")
    return buf.getvalue()


def line_loop(points: list[tuple[float, float]], color: Color, thickness: float = 3.0) -> PointsAnnotation:
    return PointsAnnotation(
        type=PointsAnnotationType.LineLoop,
        points=[Point2(x=x, y=y) for x, y in points],
        outline_color=color,
        thickness=thickness,
    )


def circle(x: float, y: float, color: Color, diameter: float = 12.0) -> CircleAnnotation:
    return CircleAnnotation(
        position=Point2(x=x, y=y),
        diameter=diameter,
        thickness=2.0,
        fill_color=color,
        outline_color=Color(r=0.05, g=0.07, b=0.10, a=1.0),
    )


def text(x: float, y: float, value: str, color: Color) -> TextAnnotation:
    return TextAnnotation(
        position=Point2(x=x, y=y),
        text=value,
        font_size=14.0,
        text_color=color,
        background_color=Color(r=0.0, g=0.0, b=0.0, a=0.70),
    )


def build_sources(args: argparse.Namespace) -> tuple[RawFrameSource | SnapshotFrameSource, SnapshotFrameSource, str]:
    depth_device = args.depth_device
    if depth_device == "auto":
        depth_device = discover_depth_device(args.robot_host)
        print(f"depth_device={depth_device}", flush=True)

    if args.rgb_source == "realsense":
        rgb_device = args.rgb_device
        if rgb_device == "auto":
            rgb_device = discover_rgb_device(args.robot_host)
            print(f"rgb_device={rgb_device}", flush=True)
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
                f"-vf scale={args.rgb_width}:{args.rgb_height},fps={args.fps} "
                "-f rawvideo -pix_fmt rgb24 -"
            ),
        ]
        rgb_source: RawFrameSource | SnapshotFrameSource = RawFrameSource(
            rgb_cmd,
            args.rgb_width,
            args.rgb_height,
            3,
            np.uint8,
            (args.rgb_height, args.rgb_width, 3),
        )
        rgb_label = "realsense_rgb"
    else:
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
        rgb_label = "front_rgb"

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
            (
                "ffmpeg -hide_banner -loglevel error "
                f"-f v4l2 -input_format gray16le -video_size {args.depth_width}x{args.depth_height} "
                f"-framerate {args.depth_fps} -i {depth_device} "
                "-frames:v 1 -f rawvideo -pix_fmt gray16le -"
            ),
        ],
        np.uint16,
        (args.depth_height, args.depth_width),
        args.depth_snapshot_timeout,
    )
    return rgb_source, depth_source, rgb_label


def make_annotations(
    now: Timestamp,
    best_box: list[float] | None,
    rgb_estimate: dict[str, Any],
    scene_closest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[ImageAnnotations, ImageAnnotations]:
    yellow = Color(r=1.0, g=0.90, b=0.0, a=1.0)
    white = Color(r=1.0, g=1.0, b=1.0, a=1.0)
    green = Color(r=0.0, g=1.0, b=0.45, a=1.0)

    rgb_points: list[PointsAnnotation] = []
    rgb_circles: list[CircleAnnotation] = []
    rgb_texts: list[TextAnnotation] = []
    depth_points: list[PointsAnnotation] = []
    depth_circles: list[CircleAnnotation] = []
    depth_texts: list[TextAnnotation] = []

    if best_box is not None:
        x0, y0, x1, y1 = [float(v) for v in best_box]
        rgb_points.append(line_loop([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], yellow))
        rgb_texts.append(text(x0, max(16.0, y0 - 4.0), "human", yellow))

    if rgb_estimate.get("visible"):
        px = float(rgb_estimate.get("closest_pixel_x", 0.0))
        py = float(rgb_estimate.get("closest_pixel_y", 0.0))
        depth_circles.append(circle(px, py, white, diameter=14.0))
        depth_texts.append(text(px + 8.0, py - 8.0, f"human {float(rgb_estimate.get('closest_depth_m', 0.0)):.2f}m", white))

    if scene_closest.get("visible"):
        px = float(scene_closest.get("pixel_x", 0.0))
        py = float(scene_closest.get("pixel_y", 0.0))
        depth_circles.append(circle(px, py, green, diameter=12.0))
        depth_texts.append(text(px + 8.0, py + 20.0, f"closest {float(scene_closest.get('depth_m', 0.0)):.2f}m", green))
        box = scene_closest.get("component_box")
        if isinstance(box, list) and len(box) == 4:
            x0, y0, x1, y1 = [float(v) for v in box]
            depth_points.append(line_loop([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], green, thickness=2.0))

    return (
        ImageAnnotations(timestamp=now, points=rgb_points, circles=rgb_circles, texts=rgb_texts),
        ImageAnnotations(timestamp=now, points=depth_points, circles=depth_circles, texts=depth_texts),
    )


def run(args: argparse.Namespace) -> None:
    params = load_policy(args.policy)
    detector = YOLO(str(args.model))
    segmenter = FastSAM(str(args.segmenter_model)) if args.trace_mode == "fastsam" else None
    rgb_source, depth_source, rgb_label = build_sources(args)

    server = foxglove.start_server(
        name="G1 Follow Dry Run",
        host=args.bind,
        port=args.foxglove_port,
        capabilities=[Capability.Time],
    )
    print(f"Foxglove WebSocket: ws://{args.bind}:{server.port}", flush=True)
    app_url = server.app_url()
    if app_url:
        print(f"Foxglove app URL: {app_url}", flush=True)
    print("DRY RUN ONLY: no Unitree motion commands are sent", flush=True)

    rgb_ch = CompressedImageChannel("/g1/front/rgb")
    depth_ch = CompressedImageChannel("/g1/realsense/depth_color")
    rgb_ann_ch = ImageAnnotationsChannel("/g1/front/annotations")
    depth_ann_ch = ImageAnnotationsChannel("/g1/realsense/depth_annotations")
    state_ch = Channel("/g1/follow/state")

    frames = 0
    start = time.time()
    try:
        while True:
            rgb_ok, rgb = rgb_source.read()
            depth_ok, depth = depth_source.read()
            if not rgb_ok or rgb is None or not depth_ok or depth is None:
                time.sleep(0.05)
                continue

            frames += 1
            now_ns = time.time_ns()
            now = timestamp_from_ns(now_ns)
            elapsed = max(1e-6, time.time() - start)

            best_box: list[float] | None = None
            best_mask = None
            best_conf = 0.0
            trace_info: dict[str, Any] = {"trace_reason": "trace_disabled" if segmenter is None else "no_person_box"}
            results = detector.predict(rgb, conf=args.conf, imgsz=args.imgsz, classes=[0], verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = []
                for idx, box in enumerate(results[0].boxes):
                    xyxy = [float(x) for x in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                    boxes.append((conf, area, idx, xyxy))
                boxes.sort(reverse=True)
                best_conf, _, _, best_box = boxes[0]
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
            safety = {
                "active": bool(scene_closest.get("visible"))
                and scene_closest.get("depth_m") is not None
                and float(scene_closest["depth_m"]) < args.safety_stop_m,
                "stop_distance_m": args.safety_stop_m,
                "closest_depth_m": scene_closest.get("depth_m"),
                "closest_pixel_x": scene_closest.get("pixel_x"),
                "closest_pixel_y": scene_closest.get("pixel_y"),
            }
            safety["reason"] = "closest_object_too_near" if safety["active"] else "clear"
            command = apply_safety_gate(raw_command, safety)

            depth_rgb = colorize_depth(depth)
            rgb_ann, depth_ann = make_annotations(now, best_box, rgb_estimate, scene_closest, args)

            rgb_ch.log(
                CompressedImage(timestamp=now, frame_id=rgb_label, data=encode_image(rgb, "jpeg", args.jpeg_quality), format="jpeg"),
                log_time=now_ns,
            )
            depth_ch.log(
                CompressedImage(timestamp=now, frame_id="realsense_depth", data=encode_image(depth_rgb, "png"), format="png"),
                log_time=now_ns,
            )
            rgb_ann_ch.log(rgb_ann, log_time=now_ns)
            depth_ann_ch.log(depth_ann, log_time=now_ns)

            state_ch.log(
                {
                    "time_unix": round(time.time(), 3),
                    "dry_run_only": True,
                    "frame": frames,
                    "fps": frames / elapsed,
                    "policy_target_source": policy_source,
                    "target_estimate": policy_estimate,
                    "rgb_person_estimate": rgb_estimate,
                    "depth_target_estimate": depth_estimate,
                    "closest_scene_object": scene_closest,
                    "safety_stop": safety,
                    "raw_policy_vx_vy_wz_stop": [float(x) for x in raw_command],
                    "wanted_input_vx_vy_wz_stop": [float(x) for x in command],
                },
                log_time=now_ns,
            )
            server.broadcast_time(now_ns)
            time.sleep(max(0.0, (1.0 / max(0.1, args.fps)) - (time.time() - now_ns / 1e9)))
    finally:
        rgb_source.stop()
        depth_source.stop()
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-host", default="192.168.0.108")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--foxglove-port", type=int, default=8765)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--segmenter-model", type=Path, default=DEFAULT_SEGMENTER_MODEL)
    parser.add_argument("--trace-mode", choices=("fastsam", "off"), default="fastsam")
    parser.add_argument("--segment-conf", type=float, default=0.2)
    parser.add_argument("--segment-imgsz", type=int, default=640)
    parser.add_argument("--segment-every", type=int, default=1)
    parser.add_argument("--trace-min-box-coverage", type=float, default=0.18)
    parser.add_argument("--trace-min-mask-inside-box", type=float, default=0.35)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--rgb-source", choices=("realsense", "front-rtsp"), default="front-rtsp")
    parser.add_argument("--rgb-device", default="auto")
    parser.add_argument("--rgb-input-format", default="yuyv422")
    parser.add_argument("--rgb-capture-width", type=int, default=424)
    parser.add_argument("--rgb-capture-height", type=int, default=240)
    parser.add_argument("--rgb-fps", type=int, default=15)
    parser.add_argument("--rgb-width", type=int, default=424)
    parser.add_argument("--rgb-height", type=int, default=240)
    parser.add_argument("--depth-device", default="auto")
    parser.add_argument("--depth-width", type=int, default=424)
    parser.add_argument("--depth-height", type=int, default=240)
    parser.add_argument("--depth-fps", type=int, default=15)
    parser.add_argument("--depth-snapshot-timeout", type=float, default=2.5)
    parser.add_argument("--conf", type=float, default=0.05)
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
    parser.add_argument("--jpeg-quality", type=int, default=82)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
