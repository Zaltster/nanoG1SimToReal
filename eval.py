"""nanoG1 — quality gate: does the policy actually walk?

    python eval.py assets/nanoG1.bin

Runs the host-physics battery (the SAME MuJoCo-validated stepper the browser demo
uses — no MuJoCo/CUDA needed) on a checkpoint and checks the frozen quality bar
(falls / velocity-tracking / smoothness). Exits nonzero on FAIL.

Needs the native demo built (`build/g1demo`); this builds it via web/build_demo.sh
if missing — which needs the engine fork, so run `bash setup.sh` once first.
"""
import operator, os, re, subprocess, sys

# the frozen bar (ref 116M checkpoint, user-approved 2026-06-15)
THRESH = [
    ("battery_falls",   "<=", 1),
    ("battery_perf",    ">=", 0.90),
    ("push_falls",      "<=", 0),
    ("push_perf",       ">=", 0.70),
    ("action_jerk_rms", "<=", 0.21),
    ("ang_vel_xy_rms",  "<=", 0.21),
    ("yaw_rate_rms",    "<=", 0.20),
    ("leg_qvel_rms",    "<=", 1.22),
]
OPS = {"<=": operator.le, ">=": operator.ge}


def num(text, key):
    m = re.search(rf"{re.escape(key)}=([0-9.]+)", text)
    return float(m.group(1)) if m else None


def main(ckpt):
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    demo = "build/g1demo"
    if not os.access(demo, os.X_OK):
        print("building host demo (web/build_demo.sh)…", flush=True)
        subprocess.run(["bash", "web/build_demo.sh"], check=True)

    def run(envkey):
        return subprocess.run([f"./{demo}", ckpt], text=True, capture_output=True,
                              env={**os.environ, envkey: "1"}).stdout

    bat = run("G1_DEMO_EVAL")        # command battery -> falls, perf
    push = run("G1_DEMO_PUSH_EVAL")  # deterministic simulated pushes -> falls, perf
    dg  = run("G1_DEMO_DIAG")        # gait diagnostic -> jerk, ang, yaw, leg_qvel
    conv = next((l for l in bat.splitlines() if "RESULT conv" in l), "")
    push_res = next((l for l in push.splitlines() if "RESULT push" in l), "")
    vals = {
        "battery_falls":   num(conv, "falls"),
        "battery_perf":    num(conv, "perf"),
        "push_falls":      num(push_res, "falls"),
        "push_perf":       num(push_res, "perf"),
        "action_jerk_rms": num(dg, "action_jerk_rms"),
        "ang_vel_xy_rms":  num(dg, "ang_vel_xy_rms"),
        "yaw_rate_rms":    num(dg, "yaw_rate_rms"),
        "leg_qvel_rms":    num(dg, "leg_qvel_rms"),
    }

    print(f"QUALITY GATE: {ckpt}")
    rc = 0
    for name, op, thr in THRESH:
        v = vals[name]
        ok = v is not None and OPS[op](v, thr)
        print(f"  {name:16s} {str(v):8s} {op} {str(thr):<6} {'PASS' if ok else 'FAIL'}")
        if not ok:
            rc = 1
    if rc != 0:
        print("GATE FAIL")
        sys.exit(1)

    print("GATE PASS — it walks. Now the eye test:")
    if sys.stdout.isatty() and not os.environ.get("NANOG1_NO_VIEW"):
        print(f"  launching demo — arrows walk/turn, R reset, close window to exit…", flush=True)
        subprocess.run([f"./{demo}", ckpt])          # interactive raylib viewer (blocks)
    else:
        print(f"  open the demo:  ./{demo} {ckpt}    (set NANOG1_NO_VIEW=1 to skip auto-launch)")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets/nanoG1.bin")
