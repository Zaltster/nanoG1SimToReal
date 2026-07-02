"""Run the nanoG1 walking policy on a REAL Unitree G1 (29-DoF), via the
unitree_sdk2py low-level (HG) interface over DDS.

    python deploy/deploy_g1.py --net eth0                 # walk in place
    python deploy/deploy_g1.py --net eth0 --teleop        # WASD drive
    python deploy/deploy_g1.py --net eth0 --go-to 1.0 1.0 # relative target

╔════════════════════════════════════════════════════════════════════════════╗
║  SAFETY — READ deploy/README.md FIRST.  Hang the robot from a gantry / have  ║
║  the remote E-stop in hand. The policy was trained in sim; first hardware    ║
║  runs WILL be rough. Start suspended, low command, ready to kill power.      ║
╚════════════════════════════════════════════════════════════════════════════╝

The observation, action mask, gains, home pose and 50 Hz / 0.8 s gait phase here
are transcribed verbatim from the validated reference (web/g1_demo.c +
web/g1_model_const.h). Inference uses the exact PufferNet forward via
libnanog1policy (build with deploy/build_policy.sh). Joint order is the standard
29-DoF G1 order (legs 0-11, waist 12-14, arms 15-28) — the same order the policy
was trained in; VERIFY it matches your robot's motor indices before running.
"""
from __future__ import annotations

import argparse, ctypes, json, math, os, sys, threading, time
import select
import urllib.request
import numpy as np

# ── policy interface (must match web/g1_demo.c + web/g1_model_const.h) ──────────
NU            = 29          # actuated joints
LEG_DOF       = 12          # v3 policy controls legs only; waist+arms held at home
CONTROL_DT    = 0.02        # 50 Hz  (G1_DT 0.004 × decimation 5)
PHASE_PERIOD  = 40          # gait-clock period in control steps  → 0.8 s
ACTION_SCALE  = 0.25
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05

# home / default joint angles (hc_key_qpos[7:], radians)
HOME = np.array([
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,      # left leg
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,      # right leg
     0.0,  0.0, 0.0,                         # waist  (yaw, roll, pitch)
     0.20, 0.20, 0.0, 1.28, 0.0, 0.0, 0.0,   # left arm
     0.20,-0.20, 0.0, 1.28, 0.0, 0.0, 0.0,   # right arm
], dtype=np.float64)

# PD gains. Legs: the v3 "Unitree" gains (g1_staged_kernels.cuh / g1_host.c:292).
# Waist+arms: hold at home with the model's actuator gains (hc_act_gain0[12:]).
KP = np.array([
    100,100,100,150,40,40,  100,100,100,150,40,40,   # legs
    75,75,75,                                          # waist
    75,75,75,75,2,2,2,  75,75,75,75,2,2,2,             # arms
], dtype=np.float64)
KD = np.array([
    2,2,2,4,2,2,  2,2,2,4,2,2,                         # legs
    2,2,2,                                              # waist
    2,2,2,2,0.2,0.2,0.2,  2,2,2,2,0.2,0.2,0.2,         # arms
], dtype=np.float64)

# position-target limits (hc_act_ctrlrange), shape (29, 2)
CTRL_RANGE = np.array([
    (-2.5307,2.8798),(-0.5236,2.9671),(-2.7576,2.7576),(-0.087267,2.8798),(-0.87267,0.5236),(-0.2618,0.2618),
    (-2.5307,2.8798),(-0.5236,2.9671),(-2.7576,2.7576),(-0.087267,2.8798),(-0.87267,0.5236),(-0.2618,0.2618),
    (-2.618,2.618),(-0.52,0.52),(-0.52,0.52),
    (-3.0892,2.6704),(-1.5882,2.2515),(-2.618,2.618),(-1.0472,2.0944),(-1.97222,1.97222),(-1.61443,1.61443),(-1.61443,1.61443),
    (-3.0892,2.6704),(-2.2515,1.5882),(-2.618,2.618),(-1.0472,2.0944),(-1.97222,1.97222),(-1.61443,1.61443),(-1.61443,1.61443),
], dtype=np.float64)

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# command teleop step sizes (vx forward, vy lateral, wyaw turn) — kept conservative
CMD_STEP = np.array([0.1, 0.1, 0.2])
CMD_MAX  = np.array([0.8, 0.4, 1.0])

# Conservative waypoint limits for first hardware tests. The policy was trained
# over a wider command range, but navigation should start well below that.
WAYPOINT_CMD_MAX = np.array([0.35, 0.20, 0.60])


def projected_gravity(quat_wxyz):
    """world gravity [0,0,-1] expressed in the base frame (matches world_to_base)."""
    w, x, y, z = quat_wxyz
    return np.array([-2*(x*z + w*y), -2*(y*z - w*x), -(1 - 2*(x*x + y*y))])


def wrap_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_policy(lib_path, bin_path):
    lib = ctypes.CDLL(lib_path)
    lib.nn_init.restype = ctypes.c_int
    lib.nn_obs.restype = ctypes.c_int
    lib.nn_nu.restype = ctypes.c_int
    if lib.nn_init(bin_path.encode()) != 0:
        sys.exit(f"policy load failed: {bin_path}")
    obs_n, nu = lib.nn_obs(), lib.nn_nu()
    assert obs_n == 98 and nu == NU, f"policy shape mismatch obs={obs_n} nu={nu}"
    obs_buf = (ctypes.c_float * obs_n)()
    act_buf = (ctypes.c_float * nu)()

    def infer(obs):
        obs_buf[:] = obs.astype(np.float32)
        lib.nn_infer(obs_buf, act_buf)
        return np.frombuffer(act_buf, dtype=np.float32).copy()
    return infer


def release_motion_mode() -> None:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    try:
        code, data = msc.CheckMode()
        print(f"motion mode before release: code={code} data={data}")
    except Exception as exc:
        print(f"motion mode check failed: {type(exc).__name__}: {exc}")
    code, _ = msc.ReleaseMode()
    print(f"motion mode release: code={code}")


def prepare_with_loco(mode: str, wait_s: float) -> None:
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    if mode == "off":
        return
    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print(f"high-level prepare: {mode}")
    if mode == "lowstand":
        loco.LowStand()
    elif mode == "highstand":
        loco.HighStand()
    elif mode == "squat2stand":
        loco.Damp()
        time.sleep(0.5)
        loco.Squat2StandUp()
    elif mode == "damp":
        loco.Damp()
    else:
        raise ValueError(f"unknown loco prepare mode: {mode}")
    if wait_s > 0:
        print(f"waiting {wait_s:.1f}s for high-level prepare to settle")
        time.sleep(wait_s)


class G1Deploy:
    def __init__(
        self,
        infer,
        teleop,
        waypoint=None,
        command_source=None,
        home_tolerance=0.12,
        home_settle=1.0,
        home_only=False,
        home_tune=False,
        home_offsets=None,
        home_offsets_out=None,
    ):
        self.infer = infer
        self.teleop = teleop
        self.waypoint = waypoint
        self.command_source = command_source
        self.home_tolerance = float(home_tolerance)
        self.home_settle = float(home_settle)
        self.home_only = bool(home_only)
        self.home_tune = bool(home_tune)
        self.home_offsets = np.array(home_offsets if home_offsets is not None else np.zeros(NU), dtype=np.float64)
        self.home_offsets_out = home_offsets_out
        self.prev_action = np.zeros(NU)
        self.cmd = np.zeros(3)
        self.step = 0
        self.pose = RelativePose()
        # SDK objects (imported lazily so the file at least parses without the SDK)
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        self.CRC = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.mode_machine = 0
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_); self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        self.low_state = msg
        self.mode_machine = msg.mode_machine

    def wait_for_state(self, timeout=5.0):
        t0 = time.time()
        while self.low_state is None:
            if time.time() - t0 > timeout:
                sys.exit("no LowState — check --net interface and that the robot is up")
            time.sleep(0.02)

    def _send(self, q, kp, kd, kd_only=False):
        self.low_cmd.mode_pr = 0          # PR (serial ankle) mode
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(NU):
            m = self.low_cmd.motor_cmd[i]
            m.mode = 1                    # enable
            m.q   = 0.0 if kd_only else float(q[i])
            m.dq  = 0.0
            m.tau = 0.0
            m.kp  = 0.0 if kd_only else float(kp[i])
            m.kd  = float(kd[i])
        self.low_cmd.crc = self.CRC.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def measured_q(self):
        return np.array([self.low_state.motor_state[i].q for i in range(NU)])

    def zero_torque(self, secs=1.0):
        print("[1/3] zero-torque (robot is limp — support it)…")
        t_end = time.time() + secs
        while time.time() < t_end:
            self._send(np.zeros(NU), np.zeros(NU), np.zeros(NU))
            time.sleep(CONTROL_DT)

    def move_to_home(self, secs=3.0):
        print(f"[2/3] moving to home pose over {secs:.0f}s…")
        q0 = self.measured_q()
        target_home = self.home_target()
        n = int(secs / CONTROL_DT)
        for k in range(n + 1):
            a = k / n
            q = (1 - a) * q0 + a * target_home
            self._send(q, KP, KD)
            time.sleep(CONTROL_DT)
        if self.home_settle > 0:
            print(f"[2/3] holding HOME for {self.home_settle:.1f}s to settle...")
            for _ in range(int(self.home_settle / CONTROL_DT)):
                self._send(target_home, KP, KD)
                time.sleep(CONTROL_DT)

    def home_target(self) -> np.ndarray:
        return np.clip(HOME + self.home_offsets, CTRL_RANGE[:, 0], CTRL_RANGE[:, 1])

    def home_alignment_report(self) -> tuple[float, str]:
        err = self.measured_q() - HOME
        order = np.argsort(np.abs(err))[::-1]
        max_err = float(np.max(np.abs(err)))
        worst = []
        for idx in order[:6]:
            worst.append(f"{JOINT_NAMES[int(idx)]}={err[int(idx)]:+.3f}rad")
        return max_err, ", ".join(worst)

    def _joint_index(self, token: str) -> int:
        try:
            idx = int(token)
        except ValueError:
            matches = [i for i, name in enumerate(JOINT_NAMES) if token.lower() in name.lower()]
            if len(matches) != 1:
                raise ValueError(f"joint '{token}' matched {matches}; use an index or a more specific name")
            idx = matches[0]
        if idx < 0 or idx >= NU:
            raise ValueError(f"joint index out of range: {idx}")
        return idx

    def _print_home_tune_status(self) -> None:
        q = self.measured_q()
        err = q - HOME
        order = np.argsort(np.abs(err))[::-1]
        print(f"[tune] max measured-vs-policy-HOME error={float(np.max(np.abs(err))):.3f}rad")
        print("[tune] worst measured errors:")
        for idx in order[:8]:
            i = int(idx)
            print(
                f"  {i:02d} {JOINT_NAMES[i]:24s} "
                f"meas={q[i]:+.3f} home={HOME[i]:+.3f} "
                f"err={err[i]:+.3f} target={self.home_target()[i]:+.3f} "
                f"offset={self.home_offsets[i]:+.3f}"
            )

    def _save_home_offsets(self, path: str | None = None) -> None:
        out = path or self.home_offsets_out
        if not out:
            print("[tune] no output path set; use: save /home/unitree/home_offsets.json")
            return
        payload = {
            "created_unix": time.time(),
            "units": "radians",
            "joint_names": JOINT_NAMES,
            "offsets": [float(x) for x in self.home_offsets],
            "note": "Command offset applied as target = policy_HOME + offset so measured joints can match policy HOME.",
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[tune] saved offsets to {out}")

    def home_tune_loop(self) -> None:
        print("[tune] holding target = policy HOME + editable offsets")
        print("[tune] commands: status | list | nudge <idx/name> <deg> | auto <scale> | zero | save [path] | quit")
        print("[tune] example: nudge 4 2.0     # left ankle pitch +2 degrees")
        next_report = 0.0
        while True:
            self._send(self.home_target(), KP, KD)
            now = time.time()
            if now >= next_report:
                max_err, worst = self.home_alignment_report()
                print(f"[tune] max_err={max_err:.3f}rad worst={worst}")
                next_report = now + 1.0
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    raise KeyboardInterrupt
                parts = line.strip().split()
                if not parts:
                    time.sleep(CONTROL_DT)
                    continue
                cmd = parts[0].lower()
                try:
                    if cmd in {"quit", "q", "exit"}:
                        raise KeyboardInterrupt
                    if cmd in {"status", "list", "ls"}:
                        self._print_home_tune_status()
                    elif cmd == "nudge" and len(parts) == 3:
                        idx = self._joint_index(parts[1])
                        delta = math.radians(float(parts[2]))
                        self.home_offsets[idx] += delta
                        print(f"[tune] {idx:02d} {JOINT_NAMES[idx]} offset -> {self.home_offsets[idx]:+.3f}rad")
                    elif cmd == "auto":
                        scale = float(parts[1]) if len(parts) > 1 else 0.5
                        correction = -scale * (self.measured_q() - HOME)
                        self.home_offsets += correction
                        self.home_offsets = np.clip(self.home_offsets, -0.6, 0.6)
                        print(f"[tune] auto applied scale={scale:.2f}; offsets clipped to +/-0.6rad")
                    elif cmd == "zero":
                        self.home_offsets[:] = 0.0
                        print("[tune] offsets reset to zero")
                    elif cmd == "save":
                        self._save_home_offsets(parts[1] if len(parts) > 1 else None)
                    else:
                        print("[tune] unknown command. Use: status | nudge <idx/name> <deg> | auto <scale> | zero | save [path] | quit")
                except ValueError as exc:
                    print(f"[tune] {exc}")
            time.sleep(CONTROL_DT)

    def damping_stop(self, secs=1.0):
        print("\nstopping -> damping")
        for _ in range(int(secs / CONTROL_DT)):
            self._send(np.zeros(NU), np.zeros(NU), 2.0 * np.ones(NU), kd_only=True)
            time.sleep(CONTROL_DT)

    def build_obs(self):
        s = self.low_state
        ang = np.array([s.imu_state.gyroscope[i] for i in range(3)])
        quat = np.array([s.imu_state.quaternion[i] for i in range(4)])  # w,x,y,z
        q  = self.measured_q()
        dq = np.array([s.motor_state[i].dq for i in range(NU)])
        ph = 2 * math.pi * ((self.step % PHASE_PERIOD) / PHASE_PERIOD)
        obs = np.zeros(98, dtype=np.float64)
        obs[0:3]   = ANG_VEL_SCALE * ang
        obs[3:6]   = projected_gravity(quat)
        obs[6:9]   = self.cmd
        obs[9:38]  = q - HOME
        obs[38:67] = DOF_VEL_SCALE * dq
        obs[67:96] = self.prev_action
        obs[96], obs[97] = math.sin(ph), math.cos(ph)
        return obs

    def run(self):
        self.wait_for_state()
        try:
            if self.command_source:
                input("[0/3] browser control connected. ENTER to zero-torque and move to HOME (Ctrl-C to exit without motion)... ")
            self.zero_torque()
            self.move_to_home()
            max_err, worst = self.home_alignment_report()
            print(f"[2/3] HOME measured joint error: max={max_err:.3f}rad tolerance={self.home_tolerance:.3f}rad")
            print(f"[2/3] worst joints: {worst}")
            if max_err > self.home_tolerance:
                input("[2/3] WARNING: measured joints are not close to policy HOME. ENTER to continue anyway, Ctrl-C to stop... ")
            if self.home_tune:
                self.home_tune_loop()
            if self.home_only:
                print("[home-only] holding policy HOME. Ctrl-C -> damping stop.")
                while True:
                    self._send(self.home_target(), KP, KD)
                    if self.step % 50 == 0:
                        max_err, worst = self.home_alignment_report()
                        print(f"[home-only] max_err={max_err:.3f}rad worst={worst}")
                    self.step += 1
                    time.sleep(CONTROL_DT)
        except KeyboardInterrupt:
            self.damping_stop()
            return
        kb = KeyTeleop() if self.teleop else None
        if self.command_source:
            input("[3/3] home reached. ENTER to start browser-controlled policy loop (browser GO/STOP still gates motion)… ")
        else:
            input("[3/3] home reached. ENTER to start the policy (Ctrl-C to stop)… ")
        self.pose.reset(self.low_state.imu_state.quaternion)
        if self.command_source:
            print(f"policy loop running - browser command source={self.command_source.url}")
        elif self.waypoint:
            print(f"policy running  - go_to target=({self.waypoint.target[0]:.2f}, {self.waypoint.target[1]:.2f}) m")
        else:
            print("policy running" + ("  - WASD to drive, space to stop" if kb else "  - walking in place"))
        try:
            next_t = time.time()
            last_t = next_t
            while True:
                now = time.time()
                dt = max(0.0, min(0.1, now - last_t))
                last_t = now
                self.pose.update(self.cmd, self.low_state.imu_state.quaternion, dt)
                if kb:
                    self.cmd = kb.update(self.cmd)
                elif self.command_source:
                    self.cmd, active, reason = self.command_source.update()
                    if not active:
                        if self.step % 50 == 0:
                            print(f"browser STOP/hold: {reason}")
                        self.prev_action[:] = 0.0
                        self._send(self.home_target(), KP, KD)
                        self.step += 1
                        next_t += CONTROL_DT
                        time.sleep(max(0.0, next_t - time.time()))
                        continue
                    if self.step % 50 == 0:
                        print(f"browser GO cmd: vx={self.cmd[0]:+.3f} vy={self.cmd[1]:+.3f} wz={self.cmd[2]:+.3f}")
                elif self.waypoint:
                    self.cmd = self.waypoint.update(self.pose)
                act = self.infer(self.build_obs())
                target = self.home_target()
                for a in range(NU):
                    c = float(np.clip(act[a], -1.0, 1.0))
                    if a >= LEG_DOF:           # v3: legs only; waist+arms stay home
                        c = 0.0
                    self.prev_action[a] = c
                    t = self.home_target()[a] + ACTION_SCALE * c
                    target[a] = np.clip(t, CTRL_RANGE[a, 0], CTRL_RANGE[a, 1])
                if self.command_source and active and self.step % 50 == 0:
                    leg_action_rms = float(np.sqrt(np.mean(self.prev_action[:LEG_DOF] ** 2)))
                    leg_delta_max = float(np.max(np.abs(target[:LEG_DOF] - HOME[:LEG_DOF])))
                    measured_delta_max = float(np.max(np.abs(self.measured_q()[:LEG_DOF] - HOME[:LEG_DOF])))
                    print(
                        "policy leg output: "
                        f"action_rms={leg_action_rms:.3f} "
                        f"target_delta_max={leg_delta_max:.3f}rad "
                        f"measured_delta_max={measured_delta_max:.3f}rad"
                    )
                self._send(target, KP, KD)
                self.step += 1
                next_t += CONTROL_DT
                time.sleep(max(0.0, next_t - time.time()))
        except KeyboardInterrupt:
            self.damping_stop()


class RelativePose:
    """Relative pose estimate from commanded velocity plus IMU yaw.

    This is enough for cautious relative tests like go_to(1, 1), but it is not
    a true localization source. Swap this for VIO, motion capture, SLAM, or
    robot odometry before using waypoint goals around obstacles or people.
    """
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._start_yaw = None

    def reset(self, quat_wxyz):
        quat = np.array([quat_wxyz[i] for i in range(4)], dtype=np.float64)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._start_yaw = yaw_from_quat(quat)

    def update(self, cmd, quat_wxyz, dt):
        if self._start_yaw is None or dt <= 0.0:
            return
        quat = np.array([quat_wxyz[i] for i in range(4)], dtype=np.float64)
        self.yaw = wrap_pi(yaw_from_quat(quat) - self._start_yaw)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vx, vy = float(cmd[0]), float(cmd[1])
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt


class WaypointController:
    def __init__(self, x, y, tolerance=0.10, max_cmd=None, kp_xy=0.8, kp_yaw=1.2):
        self.target = np.array([x, y], dtype=np.float64)
        self.tolerance = float(tolerance)
        self.max_cmd = np.array(max_cmd if max_cmd is not None else WAYPOINT_CMD_MAX, dtype=np.float64)
        self.kp_xy = float(kp_xy)
        self.kp_yaw = float(kp_yaw)
        self.done = False

    def update(self, pose):
        if self.done:
            return np.zeros(3)
        dx = self.target[0] - pose.x
        dy = self.target[1] - pose.y
        dist = math.hypot(dx, dy)
        if dist <= self.tolerance:
            self.done = True
            print(f"go_to reached: pose=({pose.x:.2f}, {pose.y:.2f}) target=({self.target[0]:.2f}, {self.target[1]:.2f})")
            return np.zeros(3)

        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        ex_body = c * dx + s * dy
        ey_body = -s * dx + c * dy
        desired_heading = math.atan2(dy, dx)
        yaw_err = wrap_pi(desired_heading - pose.yaw)

        cmd = np.array([
            self.kp_xy * ex_body,
            self.kp_xy * ey_body,
            self.kp_yaw * yaw_err,
        ])
        return np.clip(cmd, -self.max_cmd, self.max_cmd)


class KeyTeleop:
    """Non-blocking WASD command teleop (raw stdin)."""
    def __init__(self):
        import termios, tty
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def update(self, cmd):
        import select
        cmd = cmd.copy()
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1).lower()
            if   ch == 'w': cmd[0] += CMD_STEP[0]
            elif ch == 's': cmd[0] -= CMD_STEP[0]
            elif ch == 'a': cmd[2] += CMD_STEP[2]
            elif ch == 'd': cmd[2] -= CMD_STEP[2]
            elif ch == ' ': cmd[:] = 0
        return np.clip(cmd, -CMD_MAX, CMD_MAX)

    def __del__(self):
        try:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        except Exception:
            pass


class DashboardCommandSource:
    def __init__(self, url, timeout=0.15, max_age=0.75, scale=1.0, max_cmd=None, poll_hz=10.0):
        self.url = url
        self.timeout = float(timeout)
        self.max_age = float(max_age)
        self.scale = float(scale)
        self.max_cmd = np.array(max_cmd if max_cmd is not None else [0.15, 0.08, 0.25], dtype=np.float64)
        self.poll_period = 1.0 / max(1.0, float(poll_hz))
        self.lock = threading.Lock()
        self.cmd = np.zeros(3, dtype=np.float64)
        self.active = False
        self.last_reason = "not_polled_yet"
        self.last_target_time = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _set(self, cmd, active, reason, target_time=None):
        with self.lock:
            self.cmd = np.array(cmd, dtype=np.float64)
            self.active = bool(active)
            self.last_reason = str(reason)
            if target_time is not None:
                self.last_target_time = float(target_time)

    def _loop(self):
        while True:
            self._poll_once()
            time.sleep(self.poll_period)

    def _poll_once(self):
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except Exception as exc:
            self._set(np.zeros(3), False, f"dashboard_fetch_failed:{type(exc).__name__}")
            return

        now = time.time()
        target_time = payload.get("time_unix")
        if target_time is None or now - float(target_time) > self.max_age:
            self._set(np.zeros(3), False, "stale_dashboard_target", target_time or 0.0)
            return

        motion = payload.get("motion_control") or {}
        if not motion.get("enabled", False):
            self._set(np.zeros(3), False, "browser_stop", target_time)
            return

        safety = payload.get("safety_stop") or {}
        if safety.get("active", False):
            self._set(np.zeros(3), False, "dashboard_safety_stop", target_time)
            return

        command = payload.get("wanted_input_vx_vy_wz_stop")
        if not isinstance(command, list) or len(command) < 4:
            self._set(np.zeros(3), False, "missing_wanted_input", target_time)
            return

        stop = float(command[3])
        if stop >= 0.5:
            self._set(np.zeros(3), False, f"policy_stop:{stop:.2f}", target_time)
            return

        cmd = self.scale * np.array([float(command[0]), float(command[1]), float(command[2])], dtype=np.float64)
        cmd = np.clip(cmd, -self.max_cmd, self.max_cmd)
        self._set(cmd, True, "browser_go", target_time)

    def update(self):
        with self.lock:
            cmd = self.cmd.copy()
            active = self.active
            reason = self.last_reason
            target_time = self.last_target_time
        if active and target_time and time.time() - target_time > self.max_age:
            return np.zeros(3), False, "stale_cached_dashboard_target"
        return cmd, active, reason


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Deploy nanoG1 on a real Unitree G1")
    ap.add_argument("--net", required=True, help="DDS network interface to the robot (e.g. eth0)")
    ap.add_argument("--bin", default=os.path.join(here, "..", "assets", "nanoG1.bin"))
    ap.add_argument("--lib", default=None, help="path to libnanog1policy.{so,dylib}")
    ap.add_argument("--teleop", action="store_true", help="WASD command teleop")
    ap.add_argument("--go-to", nargs=2, type=float, metavar=("X", "Y"),
                    help="relative target in meters from the start pose: X forward, Y left")
    ap.add_argument("--go-to-tolerance", type=float, default=0.10,
                    help="stop when the relative target is within this many meters")
    ap.add_argument("--dashboard-command-url", default=None,
                    help="Poll a follow dashboard /target.json endpoint for browser GO/STOP velocity commands")
    ap.add_argument("--dashboard-command-timeout", type=float, default=0.15,
                    help="HTTP timeout for dashboard command polling")
    ap.add_argument("--dashboard-command-max-age", type=float, default=3.0,
                    help="Hold home if the dashboard target is older than this many seconds")
    ap.add_argument("--dashboard-command-scale", type=float, default=0.50,
                    help="Scale visual follow vx/vy/wz before sending to the walking policy")
    ap.add_argument("--dashboard-command-max-vx", type=float, default=0.15)
    ap.add_argument("--dashboard-command-max-vy", type=float, default=0.08)
    ap.add_argument("--dashboard-command-max-wz", type=float, default=0.25)
    ap.add_argument("--dashboard-command-poll-hz", type=float, default=10.0)
    ap.add_argument("--home-tolerance", type=float, default=0.12,
                    help="Warn before policy start if any measured joint is farther than this from policy HOME")
    ap.add_argument("--home-settle", type=float, default=1.0,
                    help="Hold policy HOME for this many seconds before checking joint alignment")
    ap.add_argument("--home-only", action="store_true",
                    help="Move to policy HOME and hold there; do not start the walking policy loop")
    ap.add_argument("--home-tune", action="store_true",
                    help="Interactively tune HOME command offsets while continuously holding position")
    ap.add_argument("--home-offsets", default=None,
                    help="JSON file with saved HOME command offsets")
    ap.add_argument("--home-offsets-out", default="/home/unitree/nanoG1_browser_follow/home_offsets.json",
                    help="Where --home-tune saves offsets by default")
    ap.add_argument(
        "--prepare-loco",
        choices=("off", "lowstand", "highstand", "squat2stand", "damp"),
        default="off",
        help="Optional G1 high-level loco setup before switching to low-level DDS control",
    )
    ap.add_argument("--prepare-loco-wait", type=float, default=3.0)
    ap.add_argument(
        "--release-motion-mode",
        action="store_true",
        help="Call MotionSwitcher ReleaseMode before low-level DDS control",
    )
    args = ap.parse_args()
    modes = sum(bool(x) for x in (args.teleop, args.go_to, args.dashboard_command_url))
    if modes > 1:
        ap.error("--teleop, --go-to, and --dashboard-command-url are mutually exclusive")

    lib = args.lib or os.path.join(here, "libnanog1policy." + ("dylib" if sys.platform == "darwin" else "so"))
    if not os.path.exists(lib):
        sys.exit(f"{lib} not found — run: bash deploy/build_policy.sh")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, args.net)

    if args.prepare_loco != "off" or args.release_motion_mode:
        input("[pre] ENTER to run high-level prepare/release before low-level control (Ctrl-C to exit)... ")
    if args.prepare_loco != "off":
        prepare_with_loco(args.prepare_loco, args.prepare_loco_wait)
    if args.release_motion_mode:
        release_motion_mode()

    infer = load_policy(lib, os.path.abspath(args.bin))
    waypoint = None
    command_source = None
    home_offsets = np.zeros(NU)
    if args.home_offsets:
        with open(args.home_offsets) as f:
            payload = json.load(f)
        home_offsets = np.array(payload.get("offsets", payload), dtype=np.float64)
        if home_offsets.shape != (NU,):
            raise SystemExit(f"--home-offsets must contain {NU} offsets, got shape {home_offsets.shape}")
    if args.go_to:
        waypoint = WaypointController(args.go_to[0], args.go_to[1], tolerance=args.go_to_tolerance)
    if args.dashboard_command_url:
        command_source = DashboardCommandSource(
            args.dashboard_command_url,
            timeout=args.dashboard_command_timeout,
            max_age=args.dashboard_command_max_age,
            scale=args.dashboard_command_scale,
            max_cmd=[args.dashboard_command_max_vx, args.dashboard_command_max_vy, args.dashboard_command_max_wz],
            poll_hz=args.dashboard_command_poll_hz,
        )
    G1Deploy(
        infer,
        args.teleop,
        waypoint,
        command_source,
        home_tolerance=args.home_tolerance,
        home_settle=args.home_settle,
        home_only=args.home_only,
        home_tune=args.home_tune,
        home_offsets=home_offsets,
        home_offsets_out=args.home_offsets_out,
    ).run()


if __name__ == "__main__":
    main()
