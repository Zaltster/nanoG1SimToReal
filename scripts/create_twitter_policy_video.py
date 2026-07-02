#!/usr/bin/env python3
"""Render a short social video from nanoG1 policy, command, follow, and camera clips."""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "twitter-post-2026-06-22"
TMP = OUT / "clips"
W, H = 1280, 720
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def text_fit(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > 18:
        f = font(size, bold)
        box = draw.textbbox((0, 0), text, font=f)
        if box[2] - box[0] <= max_width:
            return f
        size -= 2
    return font(size, bold)


def draw_centered_text(draw: ImageDraw.ImageDraw, lines: list[str], y: int, max_width: int, size: int, fill: tuple[int, int, int], bold: bool = False) -> None:
    line_fonts = [text_fit(draw, line, max_width, size, bold) for line in lines]
    line_boxes = [draw.textbbox((0, 0), line, font=line_font) for line, line_font in zip(lines, line_fonts)]
    line_heights = [box[3] - box[1] for box in line_boxes]
    total_height = sum(line_heights) + 16 * (len(lines) - 1)
    current_y = y - total_height // 2
    for line, line_font, box, line_height in zip(lines, line_fonts, line_boxes, line_heights):
        width = box[2] - box[0]
        draw.text(((W - width) // 2, current_y), line, font=line_font, fill=fill)
        current_y += line_height + 16


def make_overlay(path: Path, title: str, subtitle: str = "", top_label: str = "") -> None:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if top_label:
        d.rounded_rectangle((26, 24, 430, 72), radius=14, fill=(15, 18, 24, 205))
        d.text((46, 34), top_label, font=font(25, True), fill=(245, 248, 252, 255))
    d.rectangle((0, H - 168, W, H), fill=(8, 10, 14, 205))
    tf = text_fit(d, title, W - 90, 48, True)
    d.text((42, H - 145), title, font=tf, fill=(255, 255, 255, 255))
    if subtitle:
        sf = text_fit(d, subtitle, W - 90, 30, False)
        d.text((44, H - 82), subtitle, font=sf, fill=(206, 217, 230, 255))
    img.save(path)


def make_title_card(path: Path, title: str, subtitle: str | list[str] = "") -> None:
    img = Image.new("RGB", (W, H), (12, 14, 20))
    d = ImageDraw.Draw(img)
    for y in range(H):
        r = int(12 + 16 * y / H)
        g = int(14 + 24 * y / H)
        b = int(20 + 30 * y / H)
        d.line((0, y, W, y), fill=(r, g, b))
    d.rectangle((0, 0, W, H), outline=(44, 52, 66), width=4)
    d.text((60, 66), "nanoG1", font=font(38, True), fill=(120, 190, 255))
    draw_centered_text(d, [title], 282, W - 120, 64, (255, 255, 255), True)
    if subtitle:
        subtitle_lines = subtitle if isinstance(subtitle, list) else [subtitle]
        draw_centered_text(d, subtitle_lines, 400, W - 140, 33, (210, 220, 232), False)
    img.save(path)


def card_to_video(card: Path, out: Path, duration: float) -> None:
    run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(card), "-t", str(duration),
        "-vf", "fps=30,format=yuv420p", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", str(out),
    ])


def video_with_overlay(src: Path, overlay: Path, out: Path, duration: float, start: float = 0.0, loop: bool = False) -> None:
    cmd = ["ffmpeg", "-y"]
    if loop:
        cmd += ["-stream_loop", "-1"]
    cmd += [
        "-ss", str(start), "-i", str(src), "-loop", "1", "-i", str(overlay),
        "-t", str(duration), "-filter_complex",
        "[0:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=30,format=rgba[v];"
        "[v][1:v]overlay=0:0:shortest=1,format=yuv420p[out]",
        "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out),
    ]
    run(cmd)


def command_grid(out: Path, overlay: Path, duration: float) -> None:
    vids = [
        ROOT / "artifacts/2026-06-21/1091043328/command_videos/forward.mp4",
        ROOT / "artifacts/2026-06-21/1091043328/command_videos/left_strafe.mp4",
        ROOT / "artifacts/2026-06-21/1091043328/command_videos/turn_left.mp4",
        ROOT / "artifacts/2026-06-21/1091043328/command_videos/turn_right.mp4",
    ]
    cmd = ["ffmpeg", "-y"]
    for v in vids:
        cmd += ["-i", str(v)]
    cmd += ["-loop", "1", "-i", str(overlay), "-t", str(duration), "-filter_complex",
            "[0:v]scale=640:360:force_original_aspect_ratio=increase,crop=640:360,fps=30[v0];"
            "[1:v]scale=640:360:force_original_aspect_ratio=increase,crop=640:360,fps=30[v1];"
            "[2:v]scale=640:360:force_original_aspect_ratio=increase,crop=640:360,fps=30[v2];"
            "[3:v]scale=640:360:force_original_aspect_ratio=increase,crop=640:360,fps=30[v3];"
            "[v0][v1]hstack[top];[v2][v3]hstack[bot];[top][bot]vstack[grid];"
            "[grid][4:v]overlay=0:0:shortest=1,format=yuv420p[out]",
            "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out)]
    run(cmd)


def side_by_side(out: Path, overlay: Path, duration: float) -> None:
    synced = ROOT / "artifacts/g1-camera-2026-06-22/side_by_side_20260622_153043/front_depth_side_by_side.mp4"
    run([
        "ffmpeg", "-y", "-i", str(synced),
        "-loop", "1", "-i", str(overlay), "-t", str(duration), "-filter_complex",
        "[0:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=30,format=rgba[base];"
        "[base][1:v]overlay=0:0:shortest=1,format=yuv420p[out]",
        "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out),
    ])


def concat_videos(parts: list[Path], out: Path) -> None:
    concat_file = OUT / "concat.txt"
    concat_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(out),
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    cards = {
        "intro": TMP / "intro.png",
        "training": TMP / "training.png",
        "wrapper": TMP / "wrapper.png",
        "idea": TMP / "idea.png",
        "next": TMP / "next.png",
        "close": TMP / "close.png",
        "cta": TMP / "cta.png",
    }
    make_title_card(cards["intro"], "RL locomotion -> command following", "Final policy, command wrapper, then perception")
    make_title_card(cards["training"], "First: train the walking policy", [
        "Used @julianSaks env on a DGX Spark at about 300k steps/sec",
        "Added domain randomization and randomized floor friction values",
    ])
    make_title_card(cards["wrapper"], "Then: wrap it with commands", [
        "Built a command layer an LLM can call",
        "forward / strafe / turn / stop become bounded walking commands",
    ])
    make_title_card(cards["idea"], "Next idea: close the loop", [
        "In sim, the moving target stands in for a person",
        "Later: replace simulated target coords with vision + LLM/VLM intent",
    ])
    make_title_card(cards["next"], "Next: real perception", "Use camera + depth + LLM/VLM intent instead of simulated target coordinates")
    make_title_card(cards["close"], "Camera/depth is live on the G1", "Next step: RGB + depth -> person estimate -> walking command")
    make_title_card(cards["cta"], "Want to work on this?", "If you want to come work with me or test things out, DM me.")

    overlays = {}
    overlay_specs = {
        "walk": ("Final policy walking", "1.09B checkpoint: stable walking policy selected from evals", "01 / policy"),
        "cmd": ("Wrapped with walking commands", "forward, strafe, turn left, turn right -> bounded velocity commands", "02 / command wrapper"),
        "slow": ("Following a simulated person target", "The moving marker is the person stand-in; planner sends walking commands", "03 / follow"),
        "fast": ("Faster travel toward the target", "Same simulated person target, with higher command caps", "04 / faster"),
        "camera": ("Real G1 camera + depth", "synchronized capture: left RGB  |  right depth, red=close blue=far", "05 / next"),
    }
    for key, spec in overlay_specs.items():
        overlays[key] = TMP / f"{key}_overlay.png"
        make_overlay(overlays[key], *spec)

    parts = [
        TMP / "00_intro.mp4",
        TMP / "01_training.mp4",
        TMP / "02_walk.mp4",
        TMP / "03_wrapper.mp4",
        TMP / "04_commands.mp4",
        TMP / "05_idea.mp4",
        TMP / "06_slow_follow.mp4",
        TMP / "07_fast_follow.mp4",
        TMP / "08_next.mp4",
        TMP / "09_camera_depth.mp4",
        TMP / "10_close.mp4",
        TMP / "11_cta.mp4",
    ]
    card_to_video(cards["intro"], parts[0], 3.6)
    card_to_video(cards["training"], parts[1], 4.2)
    video_with_overlay(ROOT / "artifacts/2026-06-21/1091043328/walk.mp4", overlays["walk"], parts[2], 6.0, loop=True)
    card_to_video(cards["wrapper"], parts[3], 4.2)
    command_grid(parts[4], overlays["cmd"], 8.0)
    card_to_video(cards["idea"], parts[5], 4.4)
    video_with_overlay(ROOT / "artifacts/2026-06-21/1091043328/vision_follow_videos/moving_target_follow_strafe.mp4", overlays["slow"], parts[6], 8.0)
    video_with_overlay(ROOT / "artifacts/2026-06-21/1091043328/vision_follow_fast_videos/moving_target_follow_fast_aggressive.mp4", overlays["fast"], parts[7], 5.0)
    card_to_video(cards["next"], parts[8], 3.8)
    side_by_side(parts[9], overlays["camera"], 6.5)
    card_to_video(cards["close"], parts[10], 3.8)
    card_to_video(cards["cta"], parts[11], 3.8)

    concat_videos(parts, OUT / "nanog1_policy_to_vision_post.mp4")


if __name__ == "__main__":
    main()
