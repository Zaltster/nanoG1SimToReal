# G1 Vision Follow Dry Run

This folder is disposable robot-side deployment material for the G1 follow stack.

Current mode is **dry run only**:

- reads RGB from the existing go2rtc stream `rtsp://127.0.0.1:8554/front`
- reads depth from `/dev/video0`
- estimates a visible target from depth
- converts that estimate into the trained visual-command policy input
- prints/logs proposed `vx`, `vy`, `wz`, `stop`
- does not import the Unitree SDK
- does not publish movement commands

Run on the G1:

```bash
cd /home/unitree/g1-vision-follow
python3 live_g1_follow_dry_run.py --seconds 30
```

Logs are written to:

```text
/home/unitree/g1-vision-follow/logs/dry_run_commands.jsonl
```

The `--enable-motion` flag is intentionally refused in this script. Motion
should be added in a separate publisher after the camera/depth/policy command
stream is verified.

Files:

- `policies/latest_policy.npz`: visual command policy trained in sim.
- `models/yolo11n.pt`: optional detector weight for later/offboard use. The
  current G1 Python environment does not have PyTorch/Ultralytics installed.
- `live_g1_follow_dry_run.py`: robot-side dry-run command logger.

Note: Unitree's `videohub_pc4` service owns `/dev/video4`, so this dry-run uses
the already-running go2rtc `front` stream instead of opening `/dev/video4`
directly.
