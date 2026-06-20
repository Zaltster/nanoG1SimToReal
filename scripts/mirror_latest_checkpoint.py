#!/usr/bin/env python3
"""Mirror the newest Puffer checkpoint into the persistent output volume."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import time
from pathlib import Path


def checkpoint_points(ckpt_dir: Path) -> list[tuple[int, float, Path]]:
    points: list[tuple[int, float, Path]] = []
    if not ckpt_dir.exists():
        return points
    for path in ckpt_dir.rglob("*.bin"):
        match = re.match(r"^(\d{16})\.bin$", path.name)
        if not match:
            continue
        stat = path.stat()
        points.append((int(match.group(1)), stat.st_mtime, path))
    points.sort()
    return points


def archive_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.tmp")
    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(src, arcname=src.name)
    tmp.replace(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()

    points = checkpoint_points(args.checkpoint_dir)
    if not points:
        return

    counter, mtime, src = points[-1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latest = args.output_dir / "latest.bin"
    tmp = args.output_dir / f".latest.bin.{os.getpid()}.tmp"
    shutil.copy2(src, tmp)
    tmp.replace(latest)

    info = {
        "counter": counter,
        "source": str(src),
        "mtime_unix": round(mtime, 3),
        "mirrored_at_unix": round(time.time(), 3),
        "checkpoint_count": len(points),
    }
    info_tmp = args.output_dir / f".latest_checkpoint.json.{os.getpid()}.tmp"
    info_tmp.write_text(json.dumps(info, indent=2) + "\n")
    info_tmp.replace(args.output_dir / "latest_checkpoint.json")
    if args.archive:
        archive_tree(args.checkpoint_dir, args.output_dir / "checkpoints.tar.gz")
    print(json.dumps(info, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
