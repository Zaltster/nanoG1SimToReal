#!/usr/bin/env python3
"""Export YOLO person detections for a real G1 camera video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRONT = ROOT / "artifacts" / "g1-camera-2026-06-22" / "side_by_side_20260622_153043" / "front.mp4"
DEFAULT_MODEL = Path("/Users/smile/WendyOS/hat/models/yolo11n.pt")
DEFAULT_OUT = ROOT / "artifacts" / "g1-camera-2026-06-22" / "policy_replay_20260622_153043" / "person_detections_yolo.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-video", type=Path, default=DEFAULT_FRONT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    for path in (args.front_video, args.model):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    with args.out.open("w") as f:
        for frame_idx, result in enumerate(
            model.predict(
                source=str(args.front_video),
                stream=True,
                conf=args.conf,
                imgsz=args.imgsz,
                classes=[0],
                verbose=False,
            )
        ):
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = [float(x) for x in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                    boxes.append({"xyxy": xyxy, "confidence": conf, "area": area})
            boxes.sort(key=lambda item: (item["confidence"], item["area"]), reverse=True)
            payload = {
                "frame": frame_idx,
                "person_count": len(boxes),
                "best_person": boxes[0] if boxes else None,
                "persons": boxes,
            }
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps({"detections": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
