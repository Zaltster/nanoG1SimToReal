"""Phase 0 baseline: mujoco_warp (NVIDIA Warp CUDA kernels) batched GPU stepping (Modal, H100).

Measures CUDA-graph-captured mjwarp.step throughput on the G1 model across a
batch (nworld) sweep. Kernel JIT/compile time reported separately.

NOTE: mujoco_warp's API has moved fast historically. This harness guards the
unstable surfaces (put_data kwargs, qpos randomization) and degrades gracefully
— if something fails, the result blob says exactly what; paste it back and we
iterate.

Run:
  modal run bench/bench_warp.py --smoke               # ALWAYS FIRST: validate + time/cost
  modal run bench/bench_warp.py                       # full sweep
  modal run bench/bench_warp.py --batches "1024,4096"
  modal run bench/bench_warp.py --nconmax 200000 --njmax 600000   # capacity overrides

Paste back everything between the ULTRA-BENCH RESULT markers.
"""

import json
import os
import time

import modal

# robot/GPU selection (env vars: GPU is read locally at app-build so it reaches
# the container via the function def; ROBOT must be passed as a fn arg since
# module globals are re-read fresh in the container without our env).
GPU = os.environ.get("ULTRA_GPU", "RTX-PRO-6000")
ROBOT = os.environ.get("ULTRA_ROBOT", "g1")
ROBOT_ENV = {"g1": "G1JoystickFlatTerrain", "go2": "go2.mjb (frozen, md5-pinned)"}
GPU_HR = {"L40S": 1.95, "RTX-PRO-6000": 3.03, "H100": 3.95, "T4": 0.59}.get(GPU, 3.95)

# ======================================================================
# FROZEN BENCHMARK SPEC — Phase 0 (keep IDENTICAL across bench_*.py)
# ======================================================================
PLAYGROUND_ENV = "G1JoystickFlatTerrain"  # mujoco_playground task; all engines
SEED = 0                                  # one RNG seed for init everywhere
JOINT_NOISE = 0.05                        # rad, on non-root joint qpos
CTRL_NOISE = 0.10                        # around keyframe ctrl, clipped


def _load_g1(robot="g1"):
    """Load the canonical model for `robot` (g1|go2). All engines load this EXACT MjModel.

    impl="jax" override: playground env init calls mjx.put_model, and on GPU
    machines MJX auto-prefers its warp backend (crashes without mujoco_warp;
    and we want engine choice to be OURS, per-bench, not auto). The override
    only affects the env's internal mjx model build — the mj_model we extract
    is identical either way.
    """
    if robot == "go2":
        import mujoco
        m = mujoco.MjModel.from_binary_path("/root/go2.mjb")  # our frozen md5-pinned model
        # MATCH our engine's validated solver budget (the frozen mjb bakes
        # iterations=1; ours validates at Newton 2 / ls 5 — make every engine do
        # the SAME solver work per step so steps/s is apples-to-apples).
        m.opt.iterations = 2
        m.opt.ls_iterations = 5
        return m
    from mujoco_playground import registry

    env_name = ROBOT_ENV[robot]
    try:
        env = registry.load(env_name, config_overrides={"impl": "jax"})
    except (KeyError, TypeError) as e:  # config without impl / older signature
        print(f"(impl override not supported: {e!r}; plain load)")
        env = registry.load(env_name)
    return env.mj_model


def _model_fingerprint(m):
    """md5 of the compiled model binary — must MATCH across all bench files."""
    import hashlib

    import mujoco
    import numpy as np

    sz = mujoco.mj_sizeModel(m)
    buf = np.empty(sz, dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return hashlib.md5(buf.tobytes()).hexdigest()


def _opt_summary(m):
    import mujoco

    o = m.opt
    return {
        "timestep": float(o.timestep),
        "integrator": mujoco.mjtIntegrator(o.integrator).name,
        "solver": mujoco.mjtSolver(o.solver).name,
        "iterations": int(o.iterations),
        "ls_iterations": int(o.ls_iterations),
        "cone": mujoco.mjtCone(o.cone).name,
        "impratio": float(o.impratio),
        "jacobian": mujoco.mjtJacobian(o.jacobian).name,
        "disableflags": int(o.disableflags),
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "nbody": int(m.nbody),
        "ngeom": int(m.ngeom),
    }


def _base_state(m):
    """(qpos0, ctrl0) from keyframe 0 if present, else model defaults."""
    import numpy as np

    if m.nkey > 0:
        return m.key_qpos[0].copy(), m.key_ctrl[0].copy()
    return m.qpos0.copy(), np.zeros(m.nu)


def _randomized_batch(m, n):
    """Identical across engines: seeded joint + ctrl perturbations."""
    import numpy as np

    rng = np.random.RandomState(SEED)
    qpos0, ctrl0 = _base_state(m)
    qpos = np.tile(qpos0, (n, 1))
    if m.nq > 7:  # free-joint root at [0:7]; perturb actuated joints only
        qpos[:, 7:] += rng.uniform(-JOINT_NOISE, JOINT_NOISE, (n, m.nq - 7))
    ctrl = np.tile(ctrl0, (n, 1)) + rng.uniform(-CTRL_NOISE, CTRL_NOISE, (n, m.nu))
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    limited = m.actuator_ctrllimited.astype(bool)
    ctrl[:, limited] = np.clip(ctrl[:, limited], lo[limited], hi[limited])
    return qpos, ctrl


def _sys_info():
    import os
    import subprocess
    from importlib import metadata

    info = {"cpu_count": len(os.sched_getaffinity(0))}
    for pkg in ("mujoco", "mujoco-warp", "warp-lang", "playground", "numpy"):
        try:
            info[f"{pkg}_version"] = metadata.version(pkg)
        except Exception:
            info[f"{pkg}_version"] = "n/a"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        info["gpu"] = out.stdout.strip() or "none"
    except Exception:
        info["gpu"] = "none"
    return info


def _emit(blob):
    print("\n=== ULTRA-BENCH RESULT ===")
    print(json.dumps(blob, indent=2))
    print("=== END RESULT ===\n")


# Modal rates (USD/hr) — CONFIRMED against Modal pricing page 2026-06-11:
# H100 $3.95/h, physical core $0.0473/h, memory $0.0080/GiB/h.
# Keep IDENTICAL across bench_*.py.
RATE_H100_HR = 3.95
RATE_CPU_CORE_HR = 0.0473
RATE_MEM_GIB_HR = 0.0080


def _run_meta(t_start, smoke, gpu_hr, cpu_cores, mem_gib):
    """Wall-time + cost-estimate block appended to every result blob.

    Modal bills GPU + CPU cores + memory ADDITIVELY, so the container rate is
    higher than the bare GPU rate (e.g. H100 $3.95/h + 8 cores + 16 GiB
    ~= $4.46/h total). The breakdown below makes that explicit.
    """
    elapsed = time.perf_counter() - t_start
    cpu_hr = cpu_cores * RATE_CPU_CORE_HR
    mem_hr = mem_gib * RATE_MEM_GIB_HR
    rate_hr = gpu_hr + cpu_hr + mem_hr
    cost = elapsed / 3600 * rate_hr
    meta = {
        "run_mode": "smoke" if smoke else "full",
        "total_wall_s": round(elapsed, 1),
        "rate_usd_per_hr": {"gpu": gpu_hr, "cpu": round(cpu_hr, 3),
                            "mem": round(mem_hr, 3), "container_total": round(rate_hr, 2)},
        "est_cost_usd": round(cost, 3),
        "cost_note": "GPU+CPU+mem billed additively on Modal; rates confirmed vs pricing page 2026-06-11",
    }
    print(f"\n[{meta['run_mode']}] wall={elapsed:.1f}s  est cost ${cost:.3f}  "
          f"(GPU ${gpu_hr:.2f}/h + CPU ${cpu_hr:.2f}/h + mem ${mem_hr:.2f}/h "
          f"= ${rate_hr:.2f}/h container)")
    return meta
# ================== end frozen spec block ==================


app = modal.App("ultra-bench-warp")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONUNBUFFERED": "1"})  # stream prints immediately
    .apt_install("git")
    .pip_install("warp-lang", "mujoco", "playground", "numpy", "jax")  # jax (cpu): playground import dep
    .pip_install("git+https://github.com/google-deepmind/mujoco_warp.git")
    # pre-bake the menagerie clone into the image (no per-container download)
    .run_commands('python -c "from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists()"')
    # our exact frozen Go2 model (matched-physics: identical MjModel md5 across
    # all engines). playground ships Go1, not Go2, so for go2 we load our mjb.
    .add_local_file("envs/go2/model/go2.mjb", "/root/go2.mjb", copy=True)
)

DEFAULT_BATCHES = [1024, 2048, 4096, 8192, 16384]
SMOKE_BATCHES = [512]
NSTEPS = 1000  # measured steps after compile/capture
NSTEPS_SMOKE = 100


def _try_randomize(wd, name, np_batch):
    """Best-effort per-world randomization of a warp array; report outcome."""
    import numpy as np

    try:
        arr = getattr(wd, name)
        target_dtype = arr.numpy().dtype  # match whatever warp uses (f32/f64)
        arr.assign(np_batch.astype(target_dtype))
        return True
    except Exception as e:
        print(f"  (randomization of {name} skipped: {e!r})")
        return False


@app.function(image=image, gpu=GPU, cpu=8.0, memory=16384, timeout=5400)
def run(batches: str = "", nconmax: int = 0, njmax: int = 0, smoke: bool = False,
        robot: str = "g1"):
    import mujoco
    import numpy as np
    import warp as wp
    import mujoco_warp as mjwarp

    t_start = time.perf_counter()
    wp.init()
    batch_list = [int(b) for b in batches.split(",") if b] or (
        SMOKE_BATCHES if smoke else DEFAULT_BATCHES)
    nsteps = NSTEPS_SMOKE if smoke else NSTEPS

    m = _load_g1(robot)
    print(f"robot={robot} env={ROBOT_ENV[robot]} model md5: {_model_fingerprint(m)}")

    qpos0, ctrl0 = _base_state(m)
    mjd = mujoco.MjData(m)
    mjd.qpos[:] = qpos0
    mjd.ctrl[:] = ctrl0
    mujoco.mj_forward(m, mjd)

    results = []
    for n in batch_list:
        try:
            wm = mjwarp.put_model(m)

            # put_data kwargs have varied across versions; try richest first.
            # NOTE: at nworld>=1024 with randomized init, warp's default
            # constraint capacity overflows ("nefc overflow - please increase
            # njmax to ~85") — results with those warnings are INVALID; rerun
            # with --njmax 128 --nconmax 32 (per-world capacities).
            put_kwargs = {"nworld": n}
            if nconmax > 0:
                put_kwargs["nconmax"] = nconmax
            if njmax > 0:
                put_kwargs["njmax"] = njmax
            attempts = [dict(put_kwargs)]
            if nconmax > 0:  # newer mujoco_warp renamed nconmax -> naconmax
                alt = {k: v for k, v in put_kwargs.items() if k != "nconmax"}
                alt["naconmax"] = nconmax
                attempts.append(alt)
            attempts.append({"nworld": n})  # last resort: library defaults
            wd = None
            for kw in attempts:
                try:
                    wd = mjwarp.put_data(m, mjd, **kw)
                    print(f"  put_data accepted kwargs: {sorted(kw)}", flush=True)
                    break
                except TypeError as te:
                    print(f"  put_data({sorted(kw)}) rejected: {te}", flush=True)
            if wd is None:
                raise RuntimeError("all put_data signatures rejected")

            qpos_b, ctrl_b = _randomized_batch(m, n)
            randomized = _try_randomize(wd, "qpos", qpos_b)
            _try_randomize(wd, "ctrl", ctrl_b)

            # first step triggers warp kernel compilation — time it separately
            print(f"[nworld={n:5d}] compiling warp kernels "
                  f"(~85s first batch, fast after; silent)...", flush=True)
            t0 = time.perf_counter()
            mjwarp.step(wm, wd)
            wp.synchronize()
            compile_s = time.perf_counter() - t0

            # capture one step into a CUDA graph, then replay
            print(f"[nworld={n:5d}] capturing CUDA graph + timing...", flush=True)
            with wp.ScopedCapture() as capture:
                mjwarp.step(wm, wd)
            graph = capture.graph
            wp.capture_launch(graph)  # warmup launch
            wp.synchronize()

            t0 = time.perf_counter()
            for _ in range(nsteps):
                wp.capture_launch(graph)
            wp.synchronize()
            run_s = time.perf_counter() - t0

            sps = n * nsteps / run_s
            results.append({
                "batch": n, "nstep": nsteps, "randomized_init": randomized,
                "compile_s": round(compile_s, 2), "run_s": round(run_s, 4),
                "steps_per_s": round(sps, 1),
            })
            print(f"[nworld={n:5d}] {sps:,.0f} steps/s  "
                  f"(compile {compile_s:.1f}s, run {run_s:.2f}s)")

            del wd, wm  # free GPU memory before the next batch size
        except Exception as e:
            results.append({"batch": n, "error": repr(e)})
            print(f"[nworld={n:5d}] FAILED: {e!r}")

    _emit({
        "engine": "mujoco_warp",
        "robot": robot, "gpu": GPU,
        "spec": {"env": ROBOT_ENV[robot], "seed": SEED,
                 "joint_noise": JOINT_NOISE, "ctrl_noise": CTRL_NOISE},
        "model_md5": _model_fingerprint(m),
        "opt": _opt_summary(m),
        "system": _sys_info(),
        "capacity_overrides": {"nconmax": nconmax or None, "njmax": njmax or None},
        "results": results,
        "run_meta": _run_meta(t_start, smoke, gpu_hr=GPU_HR, cpu_cores=8, mem_gib=16),
    })


@app.local_entrypoint()
def main(batches: str = "", nconmax: int = 0, njmax: int = 0, smoke: bool = False,
         robot: str = ""):
    run.remote(batches=batches, nconmax=nconmax, njmax=njmax, smoke=smoke,
               robot=robot or ROBOT)
