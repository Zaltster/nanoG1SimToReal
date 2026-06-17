"""nanoG1 — train a Unitree G1 to walk in <60s, on one GPU, via Modal.

    modal run train.py            # the sub-60 walk (recipe.py baked in)
    modal run train.py --smoke    # ~10M-step smoke: validate the stack + print cost (~$0.02)

Builds a G1-specialized PufferLib fork (recipe.FORK_PIN), trains the frozen recipe,
then downloads the samples-to-walk checkpoint locally as assets/nanoG1.bin and prints
T_walk = samples-to-walk / SPS.

Prereqs: a Modal account (`modal token new`). Default GPU is an RTX 5090-class card
(RTX PRO 6000); override with `NANOG1_GPU=H100 modal run train.py`.
"""
import json, os, re, shutil, subprocess, time
import modal
import recipe as R

GPU  = os.environ.get("NANOG1_GPU", "RTX-PRO-6000")
ARCH = {"RTX-PRO-6000": "sm_120", "H100": "sm_90", "L40S": "sm_89", "A100": "sm_80"}[GPU]
CUDA = "12.8.1" if ARCH in ("sm_120", "sm_100") else "12.6.3"
RATE_HR = {"RTX-PRO-6000": 3.03, "H100": 3.95, "L40S": 1.95}.get(GPU, 3.95) + 8*0.0473 + 32*0.0080
SMOKE_TIMESTEPS = 10_000_000
PUFFER, MODEL = "/root/PufferLib", "/root/envs/g1/model/g1.mjb"

app = modal.App("nanoG1")
vol = modal.Volume.from_name("nanoG1-ckpt", create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-cudnn-devel-ubuntu22.04", add_python="3.11")
    .env({"PYTHONUNBUFFERED": "1", "DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("git", "curl", "clang", "ccache", "libomp-dev")
    .pip_install("torch", "mujoco==3.9.0", "playground==0.2.0", "jax", "numpy",
                 "pybind11", "setuptools", "rich", "rich_argparse", "gpytorch",
                 "scikit-learn", "wandb")
    .add_local_file("tools/extract_g1_model.py", "/root/extract_g1_model.py", copy=True)
    .run_commands(
        f"git clone -b {R.FORK_BRANCH} {R.FORK} {PUFFER} && cd {PUFFER} && "
        f"git checkout {R.FORK_PIN} && git log --oneline -1",
        f"pip install -e {PUFFER} --no-deps",
        "G1_MODEL_DIR=/root/envs/g1/model python /root/extract_g1_model.py",
        f"cd {PUFFER} && NVCC_ARCH={ARCH} G1_TASK_FLAGS='{R.TASK_FLAGS}' "
        f"PUFFER_TRAIN_FLAGS='{R.TRAIN_FLAGS}' ./build.sh g1gpu",
    )
)


def _apply_overrides(ini, overrides, total_timesteps):
    import configparser
    cp = configparser.ConfigParser(); cp.read(ini)
    if total_timesteps > 0:
        cp["train"]["total_timesteps"] = str(total_timesteps)
    for pair in (p.strip() for p in overrides.split(",") if p.strip()):
        k, v = pair.split("=", 1); sec, opt = k.rsplit(".", 1); cp[sec][opt] = v
    with open(ini, "w") as f: cp.write(f)


def _steady_sps(ckpt_dir):
    """Median SPS from (global_step-from-filename, mtime) of saved checkpoints."""
    pts = []
    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            m = re.match(r"^(\d{16})\.bin$", f)
            if m: pts.append((int(m.group(1)), os.path.getmtime(os.path.join(root, f))))
    pts.sort()
    rates = sorted((b[0]-a[0])/(b[1]-a[1]) for a, b in zip(pts, pts[1:])
                   if b[1] > a[1] and b[0] > a[0])
    return (rates[len(rates)//2] if rates else None), pts


def _perf_curve(log_env_dir):
    if not os.path.isdir(log_env_dir): return [], []
    for fn in sorted(f for f in os.listdir(log_env_dir) if f.endswith(".json")):
        try: m = json.load(open(os.path.join(log_env_dir, fn))).get("metrics", {})
        except Exception: continue
        return ([int(x) for x in m.get("agent_steps", [])],
                [float(x) for x in m.get("env/perf", [])])
    return [], []


@app.function(image=image, gpu=GPU, cpu=8.0, memory=32*1024, timeout=2*3600,
              volumes={"/ckpt": vol})
def train(total_timesteps: int = 0, smoke: bool = False):
    t0 = time.perf_counter()
    tt = SMOKE_TIMESTEPS if (smoke and total_timesteps == 0) else (total_timesteps or R.TOTAL_TIMESTEPS)
    os.chdir(PUFFER); os.environ["G1_MODEL_PATH"] = MODEL
    print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip(), flush=True)
    ov = R.overrides_str() + (",base.checkpoint_interval=2" if smoke else "")
    _apply_overrides("config/g1gpu.ini", ov, tt)
    shutil.rmtree("checkpoints", ignore_errors=True); shutil.rmtree("logs/g1gpu", ignore_errors=True)
    print(f"training nanoG1: {tt:,} steps on {GPU} (recipe baked, ~1-2 min before the dashboard)...", flush=True)

    t1 = time.perf_counter()
    rc = subprocess.run(["puffer", "train", "g1gpu"]).returncode
    train_s = time.perf_counter() - t1

    sps, pts = _steady_sps("checkpoints")
    steps, perf = _perf_curve("logs/g1gpu")

    # samples-to-walk checkpoint: the one nearest WALK_SAMPLES
    # (filename counter ≈ agent_steps × max_counter/total_timesteps)
    walk_counter = None
    if pts:
        target = int(R.WALK_SAMPLES * pts[-1][0] / tt)
        walk_counter = min((c for c, _ in pts), key=lambda c: abs(c - target))

    run, walk_bytes = f"nanoG1-{int(t0)}", None
    if os.path.isdir("checkpoints"):
        shutil.copytree("checkpoints", f"/ckpt/{run}", dirs_exist_ok=True); vol.commit()
        if walk_counter:
            for root, _, files in os.walk("checkpoints"):
                for f in files:
                    if f == f"{walk_counter:016d}.bin":
                        walk_bytes = open(os.path.join(root, f), "rb").read()

    t_walk = R.WALK_SAMPLES / sps if sps else None
    perf_at_walk = None
    if steps and perf:
        i = min(range(len(steps)), key=lambda j: abs(steps[j] - R.WALK_SAMPLES))
        perf_at_walk = round(perf[i], 3)
    print("\n=== nanoG1 RESULT ===")
    print(json.dumps({
        "exit_code": rc, "gpu": GPU, "total_timesteps": tt,
        "steady_sps": round(sps, 1) if sps else None,
        "physics_steps_per_s": round(sps * R.DECIMATION, 1) if sps else None,
        "T_walk_s": round(t_walk, 1) if t_walk else None,
        "walk_samples": R.WALK_SAMPLES, "perf_at_walk": perf_at_walk,
        "final_perf": round(perf[-1], 3) if perf else None,
        "est_cost_usd": round((time.perf_counter()-t0)/3600*RATE_HR, 3), "run": run,
    }, indent=2))
    print("=== END RESULT ===\n", flush=True)
    return {"walk_bytes": walk_bytes, "t_walk": t_walk, "smoke": smoke}


@app.local_entrypoint()
def main(smoke: bool = False, total_timesteps: int = 0):
    r = train.remote(total_timesteps=total_timesteps, smoke=smoke)
    if smoke:
        print("\n✓ smoke OK — stack builds + trains. Now: `modal run train.py`")
    elif r.get("walk_bytes"):
        os.makedirs("assets", exist_ok=True)
        with open("assets/nanoG1.bin", "wb") as f: f.write(r["walk_bytes"])
        print(f"\n✓ trained — wrote assets/nanoG1.bin  (T_walk ≈ {r['t_walk']:.1f}s)")
        print("  verify it walks:  python eval.py assets/nanoG1.bin")
    else:
        print("\n⚠ training finished but no walk checkpoint was captured — check the log above.")
