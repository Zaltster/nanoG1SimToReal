#!/usr/bin/env python3
"""Copy checkpoint/video artifacts into date-first, step-named folders."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize nanoG1 artifacts as artifacts/YYYY-MM-DD/<steps>/..."
    )
    parser.add_argument("--date", required=True, help="Run/artifact date, e.g. 2026-06-22")
    parser.add_argument("--steps", required=True, type=int, help="Checkpoint step counter")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts"),
        help="Artifact root directory",
    )
    parser.add_argument(
        "--copy",
        nargs=2,
        action="append",
        metavar=("SRC", "NAME"),
        default=[],
        help="Copy SRC into the run folder as NAME",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.root / args.date / str(args.steps)
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str | int]] = []
    for src_text, name in args.copy:
        src = Path(src_text)
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = run_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(
            {
                "source": str(src),
                "file": str(dst.relative_to(run_dir)),
                "bytes": dst.stat().st_size,
            }
        )

    manifest = {
        "date": args.date,
        "steps": args.steps,
        "layout": "artifacts/YYYY-MM-DD/<steps>/",
        "files": copied,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
