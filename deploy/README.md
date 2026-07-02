# deploy/ — run nanoG1 on a real Unitree G1

Take the policy you trained (or the shipped `assets/nanoG1.bin`) and run it on a
physical **Unitree G1 (29-DoF)** over the low-level DDS interface.

> ⚠️ **This commands a real humanoid.** The policy is trained in simulation; the
> first hardware runs will be rough. **Hang the robot from a gantry, keep the
> remote E-stop in hand, and start with a zero command.** You are responsible for
> safe operation. Begin suspended until the gait is stable.

## What it does

`deploy_g1.py` runs a 50 Hz control loop: read the G1's IMU + joint encoders →
build the exact 98-dim observation the policy was trained on → run the policy →
send joint **position targets** with PD gains. It controls the **12 leg joints**;
the waist and arms are held at the home pose (this is what the v3 policy expects).
Inference uses the *same* PufferNet forward as the browser demo and `eval.py`
(via a small C shim), so on-robot behavior matches what you validated.

## Prerequisites

1. **The engine fork** (for `puffernet.h`): `bash setup.sh` (once, from repo root).
2. **Build the inference shim**: `bash deploy/build_policy.sh` → `deploy/libnanog1policy.{so,dylib}`.
3. **Unitree SDK (Python)** — install from Unitree:
   ```bash
   git clone https://github.com/unitreerobotics/unitree_sdk2_python
   cd unitree_sdk2_python && pip install -e .          # also needs cyclonedds
   ```
   (`pip install -e ".[deploy]"` here installs numpy; `unitree_sdk2py` is not on PyPI.)
4. A machine on the **robot's network** (wired is recommended). Note the network
   interface name (e.g. `eth0`).

## Run

```bash
python deploy/deploy_g1.py --net eth0            # walk in place (zero command)
python deploy/deploy_g1.py --net eth0 --teleop   # WASD: w/s = forward/back, a/d = turn, space = stop
python deploy/deploy_g1.py --net eth0 --go-to 1.0 1.0
python deploy/deploy_g1.py --net en0 --dashboard-command-url http://127.0.0.1:8094/target.json  # controller on the Mac
python deploy/deploy_g1.py --net eth0 --dashboard-command-url http://192.168.0.135:8094/target.json # controller on the robot
```

`--go-to X Y` is a high-level waypoint command in meters, relative to the
robot's starting pose: `X` is forward and `Y` is left. The deploy script
converts it into the policy's normal `[vx, vy, yaw_rate]` command. The current
implementation estimates relative position by integrating commanded velocity
and IMU yaw, so it is suitable only for cautious first tests in a clear space.
For reliable real-world navigation, replace that pose estimate with motion
capture, VIO, SLAM, or robot odometry.

`--dashboard-command-url` makes the controller poll the browser follow dashboard.
The dashboard must be started with `--allow-browser-motion-control`; its GO/STOP
button only changes the `/target.json` motion state. The dashboard process never
publishes Unitree commands. The low-level deploy process still owns the 50 Hz
DDS loop, holds `HOME` while the browser is STOP, and only feeds visual
`vx/vy/wz` commands to the walking policy while the browser state is GO, the
target is fresh, and the dashboard safety stop is clear.

Sequence the script enforces:
1. **Zero-torque** (~1 s) — robot goes limp; support it.
2. **Move to home** (~3 s) — interpolates to the default crouch.
3. **Wait for ENTER** — confirm the robot is hanging/ready.
4. **Policy** at 50 Hz. **Ctrl-C** → soft damping stop.

## Verify before you run on hardware

These are transcribed from the training reference (`web/g1_demo.c`,
`web/g1_model_const.h`); confirm they're right for *your* robot/firmware:

- **Joint order** — the script assumes the standard 29-DoF order (left leg 0–5,
  right leg 6–11, waist 12–14, left arm 15–21, right arm 22–28) and that this is
  the order the policy was trained in. If your motor indices differ, remap.
- **Ankle mode** — `mode_pr = 0` (PR / serial ankle). Change if your robot uses AB.
- **PD gains** — legs use the trained gains (`KP=[100,100,100,150,40,40]`,
  `KD=[2,2,2,4,2,2]` per leg); waist/arms hold with the model's gains. Tune if needed.
- **Home pose** must match `HOME` in `deploy_g1.py`.

## Tuning notes

- Start with `--net` only (zero command). Once it stands and steps cleanly while
  suspended, lower it to the ground, then introduce small forward commands.
- Treat `--go-to` as the next layer above teleop, not as proof of autonomous
  navigation. Test tiny targets first, for example `--go-to 0.2 0.0`, then stop
  and inspect drift before trying diagonal targets.
- If it's twitchy, lower leg `KP` slightly or add a low-pass on the action.
- The observation/scales are fixed by training — **do not change them** or the
  policy sees out-of-distribution input.
