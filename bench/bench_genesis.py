"""G1-now competitive benchmark: Genesis (Genesis-Embodied-AI) batched GPU
stepping on the Unitree G1, in our Phase-0 protocol (Modal, single GPU).

WHY this exists: Genesis is the most-hyped "fastest sim" claim in robotics RL.
Our thesis is per-robot specialization beats general-purpose GPU sim. This adds
Genesis next to warp/MJX/CPU-C — but Genesis is NOT MuJoCo physics (its own
solver/contact), so this is a *competitor* benchmark with loud caveats, not a
matched-physics datapoint like the warp/MJX wall.

HONESTY FRAMING (read before quoting any number):
  * Different physics: Genesis uses its own Newton/contact, not MuJoCo's. We do
    NOT claim "same physics, faster." Genesis is also MJCF-reparsed, not our
    md5-pinned model.
  * Units trap: raw steps/s is meaningless across engines at different dt. We
    report `sim_s_per_wall_s` = n_envs * nsteps * dt / wall (simulated robot-
    seconds generated per wall second) as the dt-NORMALIZED fair metric, plus
    raw steps/s WITH the dt stated. (Genesis locomotion ships at dt 0.02 single-
    step; we step dt 0.002-0.004 with decimation 5-10 => far more physics work
    per control step. Comparing at matched dt removes that asymmetry.)

Run:
  modal run bench/bench_genesis.py --smoke                 # ALWAYS FIRST
  modal run bench/bench_genesis.py                          # full sweep
  modal run bench/bench_genesis.py --dts "0.002" --batches "8192,32768"

Paste back everything between the ULTRA-BENCH RESULT markers.
"""

import json
import os
import time

import modal

# ---- frozen-ish spec (mirrors bench_warp.py where it can) ----
SEED = 0
JOINT_NOISE = 0.05   # rad, position-target perturbation (matches stepping-bench spirit)

# GPU tier — benchmark on the SAME silicon we quote our numbers on (mirrors
# train_g1gpu.py). Default RTX-PRO-6000 (= RTX 5090, Blackwell GB202, sm_120),
# where our headline 1.78M SPS / 8.9M physics-steps/s live.
#   ULTRA_GPU=L40S  modal run ...   # the RTX 4090 comparison (Ada, safest)
#   ULTRA_GPU=H100  modal run ...   # the Phase-0 wall GPU
GPU_TYPE = os.environ.get("ULTRA_GPU", "RTX-PRO-6000")
GPU_ARCH = {"H100": "sm_90", "L40S": "sm_89", "A100": "sm_80",
            "RTX-PRO-6000": "sm_120"}.get(GPU_TYPE, "sm_90")
RATE_GPU_HR = {"H100": 3.95, "L40S": 1.95, "A100": 3.40,
               "RTX-PRO-6000": 3.03}.get(GPU_TYPE, 3.95)
# Blackwell (sm_120) needs CUDA >= 12.8 + torch cu128; Ada/Hopper fine on cu124.
CUDA_VER = "12.8.1" if GPU_ARCH in ("sm_120", "sm_100") else "12.6.3"
TORCH_CUDA = "cu128" if GPU_ARCH in ("sm_120", "sm_100") else "cu124"
RATE_CPU_CORE_HR = 0.0473
RATE_MEM_GIB_HR = 0.0080


def _run_meta(t_start, smoke, gpu_hr, cpu_cores, mem_gib):
    elapsed = time.perf_counter() - t_start
    cpu_hr = cpu_cores * RATE_CPU_CORE_HR
    mem_hr = mem_gib * RATE_MEM_GIB_HR
    rate_hr = gpu_hr + cpu_hr + mem_hr
    cost = elapsed / 3600 * rate_hr
    print(f"\n[{'smoke' if smoke else 'full'}] wall={elapsed:.1f}s  est cost ${cost:.3f}  "
          f"(${rate_hr:.2f}/h container)")
    return {"run_mode": "smoke" if smoke else "full", "total_wall_s": round(elapsed, 1),
            "rate_usd_per_hr": {"gpu": gpu_hr, "cpu": round(cpu_hr, 3),
                                "mem": round(mem_hr, 3), "container_total": round(rate_hr, 2)},
            "est_cost_usd": round(cost, 3),
            "cost_note": "GPU+CPU+mem billed additively on Modal; rates confirmed vs pricing page 2026-06-11"}


def _emit(blob):
    print("\n=== ULTRA-BENCH RESULT ===")
    print(json.dumps(blob, indent=2))
    print("=== END RESULT ===\n")


app = modal.App("ultra-bench-genesis")

# Genesis needs a CUDA runtime (gs.gpu). torch+cuda wheels + genesis-world, and
# mujoco_menagerie for the canonical G1 MJCF (our .xml references menagerie mesh
# paths). PYTHONUNBUFFERED so the long compile is visible, not a hang.
# CUDA devel base matched to the GPU arch (Blackwell needs 12.8) — same approach
# as train_g1gpu.py, so Genesis's runtime CUDA compile targets the right sm_.
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_VER}-cudnn-devel-ubuntu22.04",
                              add_python="3.11")
    .env({"PYTHONUNBUFFERED": "1"})
    # Genesis eagerly imports its pyglet/X11 visualizer at `import genesis`
    # (even with show_viewer=False), so the X11/GL shared libs must be present
    # for the import to resolve — we never open a window.
    .apt_install("git", "libgl1", "libglib2.0-0", "libegl1",
                 "libxrender1", "libxext6", "libsm6", "libx11-6", "libxi6",
                 "libxxf86vm1", "libxfixes3", "libxcursor1", "libxrandr2", "libxinerama1")
    .pip_install("torch", index_url=f"https://download.pytorch.org/whl/{TORCH_CUDA}")
    .pip_install("genesis-world", "numpy", "mujoco")
    .run_commands("git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /menagerie")
)

DEFAULT_DTS = [0.002, 0.005]      # 0.002 matches the warp/MJX wall; 0.005 ~ Genesis-natural
DEFAULT_BATCHES = [4096, 8192, 16384, 32768, 65536]
SMOKE_DTS = [0.002]
SMOKE_BATCHES = [1024]
NSTEPS = 1000
NSTEPS_SMOKE = 100


ROBOT = os.environ.get("ULTRA_ROBOT", "g1")


def _find_robot_xml(robot="g1"):
    """menagerie MJCF for `robot`. Genesis reparses MJCF (can't load our mjb);
    we point it at the same menagerie source our go2.mjb was frozen from."""
    import glob, os
    if robot == "go2":
        cands = ["/menagerie/unitree_go2/go2.xml",
                 "/menagerie/unitree_go2/go2_mjx.xml"]
        cands += sorted(glob.glob("/menagerie/unitree_go2/go2*.xml"))
        d = "/menagerie/unitree_go2"
    else:
        cands = [
            "/menagerie/unitree_g1/g1_29dof.xml",
            "/menagerie/unitree_g1/g1_29dof_rev_1_0.xml",
            "/menagerie/unitree_g1/scene_29dof.xml",
            "/menagerie/unitree_g1/g1.xml",
        ]
        cands += sorted(glob.glob("/menagerie/unitree_g1/*29dof*.xml"))
        cands += sorted(glob.glob("/menagerie/unitree_g1/g1*.xml"))
        d = "/menagerie/unitree_g1"
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"no {robot} xml under {d} ({os.listdir(d)})")


@app.function(image=image, gpu=GPU_TYPE, cpu=8.0, memory=32768, timeout=5400)
def run(dts: str = "", batches: str = "", smoke: bool = False, robot: str = "g1"):
    import numpy as np
    import genesis as gs

    t_start = time.perf_counter()
    dt_list = [float(x) for x in dts.split(",") if x] or (SMOKE_DTS if smoke else DEFAULT_DTS)
    batch_list = [int(b) for b in batches.split(",") if b] or (SMOKE_BATCHES if smoke else DEFAULT_BATCHES)
    nsteps = NSTEPS_SMOKE if smoke else NSTEPS

    robot_name = robot   # the loop below rebinds `robot` to the Genesis entity
    gs.init(backend=gs.gpu, performance_mode=True)
    g1_xml = _find_robot_xml(robot_name)
    print(f"Genesis robot={robot_name} xml: {g1_xml}", flush=True)

    results = []
    model_info = {}
    poisoned = False
    for dt in dt_list:
        if poisoned:
            break
        for n in batch_list:
            try:
                # enable_self_collision=False — Genesis defaults it TRUE, which
                # makes the G1's mesh links all self-collide (~3112 constraints/
                # env -> huge dense Jacobian, OOM at 32k). NO locomotion setup
                # does this (legged-gym / MJX playground / Genesis's own go2_env
                # all disable it), and our engine models foot-floor contact only.
                # This is the single biggest fairness lever.
                scene = gs.Scene(
                    show_viewer=False,
                    rigid_options=gs.options.RigidOptions(
                        dt=dt, constraint_solver=gs.constraint_solver.Newton,
                        enable_self_collision=False),
                )
                scene.add_entity(gs.morphs.Plane())
                robot = scene.add_entity(gs.morphs.MJCF(file=g1_xml))
                print(f"[dt={dt} n={n}] building scene (Genesis compiles kernels on build/first step; silent ~30-90s)...", flush=True)
                t0 = time.perf_counter()
                scene.build(n_envs=n)
                build_s = time.perf_counter() - t0

                # actuated dofs -> position control at home (+ seeded noise)
                dof_idx = list(range(robot.n_dofs))
                # drop the 6 free-base dofs if present (first link is floating base)
                motor_idx = [robot.get_joint(j.name).dofs_idx_local[0]
                             for j in robot.joints if len(j.dofs_idx_local) == 1]
                if not model_info:
                    model_info = {"xml": g1_xml, "n_dofs": int(robot.n_dofs),
                                  "n_links": int(robot.n_links), "n_motors": len(motor_idx)}
                rng = np.random.RandomState(SEED)
                tgt = rng.uniform(-JOINT_NOISE, JOINT_NOISE, (n, len(motor_idx))).astype(np.float32)
                robot.set_dofs_kp(np.full(len(motor_idx), 100.0), motor_idx)
                robot.control_dofs_position(tgt, motor_idx)

                # warmup (first steps trigger remaining compilation), then time
                for _ in range(5):
                    scene.step()
                t0 = time.perf_counter()
                for _ in range(nsteps):
                    scene.step()
                run_s = time.perf_counter() - t0

                sps = n * nsteps / run_s
                sim_s_per_wall_s = n * nsteps * dt / run_s
                results.append({
                    "dt": dt, "batch": n, "nstep": nsteps,
                    "build_s": round(build_s, 1), "run_s": round(run_s, 4),
                    "steps_per_s": round(sps, 1),
                    "sim_s_per_wall_s": round(sim_s_per_wall_s, 1),
                })
                print(f"[dt={dt} n={n:6d}] {sps:,.0f} steps/s  |  {sim_s_per_wall_s:,.0f} sim-s/wall-s  (build {build_s:.0f}s)", flush=True)
                del scene, robot
            except Exception as e:
                msg = repr(e)
                results.append({"dt": dt, "batch": n, "error": msg})
                print(f"[dt={dt} n={n}] FAILED: {e!r}", flush=True)
                # OOM / FieldsBuilder corrupt the Genesis process — later configs
                # then fail spuriously. Abort the sweep so we don't record garbage.
                if "OUT_OF_MEMORY" in msg or "FieldsBuilder" in msg:
                    print("  (Genesis process poisoned; aborting remaining configs — rerun larger batches separately)", flush=True)
                    poisoned = True
                    break

    import subprocess
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        gpu = "n/a"
    from importlib import metadata
    vers = {}
    for p in ("genesis-world", "torch", "mujoco", "numpy"):
        try: vers[p] = metadata.version(p)
        except Exception: vers[p] = "n/a"

    _emit({
        "engine": "genesis",
        "robot": robot_name,
        "DISCLAIMER": "Genesis physics != MuJoCo (own solver/contact); MJCF-reparsed menagerie "
                      "model, not our md5-pinned mjb. Compare via sim_s_per_wall_s (dt-normalized), "
                      "not raw steps/s. NOT part of the matched-physics warp/MJX wall.",
        "model": model_info,
        "constraint_solver": "Genesis-Newton",
        "collision_config": "enable_self_collision=False (foot-floor contact only, matches our engine + standard locomotion)",
        "gpu_tier": {"requested": GPU_TYPE, "arch": GPU_ARCH, "cuda": CUDA_VER, "torch": TORCH_CUDA},
        "system": {"gpu": gpu, "versions": vers},
        "results": results,
        "reference_ours": {
            "RTX-PRO-6000": {"ours_physics_steps_per_s_staged": 8.9e6, "ours_training_sps": 1.78e6,
                             "ours_sim_s_per_wall_s_at_dt0.004": round(8.9e6 * 0.004, 1)},
            "L40S": {"ours_training_sps": 525e3, "note": "L40S = RTX 4090 proxy (v1)"},
            "H100": {"ours_physics_steps_per_s_staged": 5.68e6, "ours_training_sps": 568e3,
                     "warp_wall_steps_per_s": 3.34e6},
        }.get(GPU_TYPE, {}),
        "compare_note": "Use sim_s_per_wall_s (dt-normalized) for cross-engine. Ours @dt0.004: "
                        "8.9M steps/s -> 35,600 sim-s/wall-s on PRO-6000.",
        "run_meta": _run_meta(t_start, smoke, gpu_hr=RATE_GPU_HR, cpu_cores=8, mem_gib=32),
    })


@app.local_entrypoint()
def main(dts: str = "", batches: str = "", smoke: bool = False, robot: str = ""):
    run.remote(dts=dts, batches=batches, smoke=smoke, robot=robot or ROBOT)
