"""Phase 0 baseline: MJX (JAX/XLA) batched GPU stepping (Modal, H100).

Measures jit(vmap(mjx.step)) throughput on the G1 model across a batch sweep.

METHODOLOGY NOTE (from smoke runs 2026-06-11): the primary metric is a repeated
jitted batched step (async dispatch, block at end) — NOT lax.scan. On
jax 0.10.1 + H100, XLA's while-loop path is catastrophically misoptimized for
this workload: both scan(vmap(step)) and the official-testspeed
vmap(scan(step, unroll=4)) patterns measured 250-1050 ms/step regardless of
batch size, while the identical step outside scan costs 3.6-6.1 ms/step
(~672k steps/s @ 4096 — the expected MJX band). The repeated-step loop is also
how RL training invokes the simulator per control step. A scan measurement
remains available via --with-scan for the record.

Run:
  modal run bench/bench_mjx.py --smoke                 # ALWAYS FIRST: validate + time/cost
  modal run bench/bench_mjx.py                         # full sweep
  modal run bench/bench_mjx.py --batches "1024,4096"   # optional override
  modal run bench/bench_mjx.py --smoke --with-scan     # include the (slow) scan diagnostic

Paste back everything between the ULTRA-BENCH RESULT markers.
"""

import json
import os
import time

import modal

# robot/GPU selection (GPU read locally -> reaches container via fn def; ROBOT
# passed as fn arg since module globals re-read fresh in the container).
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

    go2: our frozen md5-pinned go2.mjb (playground ships Go1, not Go2).
    g1:  playground G1 (impl="jax" override only affects the env's internal mjx
    build — the mj_model we extract is identical either way).
    """
    if robot == "go2":
        import mujoco
        m = mujoco.MjModel.from_binary_path("/root/go2.mjb")
        # MATCH our engine's validated solver budget (mjb bakes iterations=1;
        # ours validates at Newton 2 / ls 5) so steps/s is apples-to-apples.
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
    for pkg in ("mujoco", "mujoco-mjx", "jax", "playground", "numpy"):
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


app = modal.App("ultra-bench-mjx")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONUNBUFFERED": "1"})  # stream prints immediately
    .apt_install("git")  # playground clones mujoco_menagerie via git
    .pip_install("jax[cuda12]", "mujoco", "mujoco-mjx", "playground", "numpy")
    # pre-bake the menagerie clone into the image (no per-container download)
    .run_commands('python -c "from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists()"')
    .add_local_file("envs/go2/model/go2.mjb", "/root/go2.mjb", copy=True)
)

DEFAULT_BATCHES = [1024, 2048, 4096, 8192, 16384]
SMOKE_BATCHES = [512, 4096]  # two points: does throughput scale with batch?
NSTEPS = 500       # timed steps per batch (full)
NSTEPS_SMOKE = 100
WARMUP = 10
SCAN_CHUNK = 20    # only used with --with-scan (the slow diagnostic)


def _mjx_static_sizes(d0):
    """MJX pads contacts/constraints to static worst-case sizes — report them.

    These paddings are the prime suspect if throughput is unexpectedly low
    (every env pays for the padded nefc/ncon every step), and a key datapoint
    for our specialization thesis either way.
    """
    out = {}
    impl = getattr(d0, "_impl", d0)
    try:
        con = getattr(impl, "contact", None) or getattr(d0, "contact", None)
        if con is not None and hasattr(con, "dist"):
            out["ncon_padded"] = int(con.dist.shape[-1])
    except Exception as e:
        out["ncon_error"] = repr(e)
    for name in ("efc_J", "efc_D", "efc_aref", "qM"):
        try:
            arr = getattr(impl, name, None)
            if arr is None:
                arr = getattr(d0, name, None)
            if arr is not None and hasattr(arr, "shape"):
                out[f"{name}_shape"] = list(arr.shape)
        except Exception:
            pass
    return out


@app.function(image=image, gpu=GPU, cpu=8.0, memory=16384, timeout=5400)
def run(batches: str = "", smoke: bool = False, with_scan: bool = False,
        unroll_steps: int = 4, robot: str = "g1"):
    import os

    # set BEFORE importing jax (parity with the training bench)
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from mujoco import mjx

    t_start = time.perf_counter()
    batch_list = [int(b) for b in batches.split(",") if b] or (
        SMOKE_BATCHES if smoke else DEFAULT_BATCHES)
    nsteps = NSTEPS_SMOKE if smoke else NSTEPS

    m = _load_g1(robot)
    print(f"robot={robot} env={ROBOT_ENV[robot]} model md5: {_model_fingerprint(m)}")
    print(f"jax devices: {jax.devices()}")

    # impl="jax": THIS bench measures the MJX/XLA path. On GPU machines newer
    # MJX auto-prefers its warp backend — that's bench_warp.py's job, not ours.
    try:
        mjxm = mjx.put_model(m, impl="jax")
    except TypeError:  # older mjx without the impl kwarg
        mjxm = mjx.put_model(m)
    d0 = mjx.make_data(mjxm)

    static_sizes = _mjx_static_sizes(d0)
    print(f"mjx static (padded) sizes: {static_sizes}")

    results = []
    for n in batch_list:
        try:
            qpos_b, ctrl_b = _randomized_batch(m, n)
            qpos_j = jnp.asarray(qpos_b)
            ctrl_j = jnp.asarray(ctrl_b)
            ds = jax.vmap(lambda q, c: d0.replace(qpos=q, ctrl=c))(qpos_j, ctrl_j)

            # ---- PRIMARY: repeated jitted batched step (no scan) ----
            # See the methodology note in the module docstring: XLA while-loops
            # (lax.scan) are catastrophically slow on this jax/H100 stack, so
            # the honest engine speed is the repeated-dispatch jitted step.
            step_jit = jax.jit(jax.vmap(lambda d: mjx.step(mjxm, d)),
                               donate_argnums=0)

            print(f"[batch={n:5d}] compiling XLA program (~35-40s, silent)...",
                  flush=True)
            t0 = time.perf_counter()
            ds = step_jit(ds)
            jax.block_until_ready(ds.qpos)
            compile_s = time.perf_counter() - t0

            for _ in range(WARMUP):
                ds = step_jit(ds)
            jax.block_until_ready(ds.qpos)

            t0 = time.perf_counter()
            for _ in range(nsteps):
                ds = step_jit(ds)
            jax.block_until_ready(ds.qpos)
            run_s = time.perf_counter() - t0

            sps = n * nsteps / run_s
            entry = {
                "batch": n, "nstep": nsteps,
                "compile_s": round(compile_s, 2), "run_s": round(run_s, 4),
                "steps_per_s": round(sps, 1),
                "per_step_ms": round(run_s / nsteps * 1000, 2),
            }
            print(f"[batch={n:5d}] {sps:,.0f} steps/s  "
                  f"(compile {compile_s:.1f}s, run {run_s:.2f}s, "
                  f"{run_s / nsteps * 1000:.2f} ms/step)")

            # ---- optional diagnostic: the (pathological) scan path ----
            if with_scan:
                def step_one(d, _):
                    return mjx.step(mjxm, d), None

                def unroll_one(d):
                    d, _ = jax.lax.scan(step_one, d, None, length=SCAN_CHUNK,
                                        unroll=unroll_steps)
                    return d

                unroll = jax.jit(jax.vmap(unroll_one), donate_argnums=0)
                t0 = time.perf_counter()
                ds = unroll(ds)
                jax.block_until_ready(ds.qpos)
                scan_compile_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                ds = unroll(ds)
                jax.block_until_ready(ds.qpos)
                scan_run_s = time.perf_counter() - t0
                entry["scan_diag"] = {
                    "chunk": SCAN_CHUNK, "unroll": unroll_steps,
                    "compile_s": round(scan_compile_s, 2),
                    "per_step_ms": round(scan_run_s / SCAN_CHUNK * 1000, 2),
                }
                print(f"          scan diag: "
                      f"{scan_run_s / SCAN_CHUNK * 1000:.1f} ms/step "
                      f"(compile {scan_compile_s:.1f}s)")

            results.append(entry)
        except Exception as e:
            results.append({"batch": n, "error": repr(e)})
            print(f"[batch={n:5d}] FAILED: {e!r}")

    _emit({
        "engine": "mjx",
        "robot": robot, "gpu": GPU,
        "spec": {"env": ROBOT_ENV[robot], "seed": SEED,
                 "joint_noise": JOINT_NOISE, "ctrl_noise": CTRL_NOISE},
        "model_md5": _model_fingerprint(m),
        "opt": _opt_summary(m),
        "system": _sys_info(),
        "mjx_static_sizes": static_sizes,
        "results": results,
        "run_meta": _run_meta(t_start, smoke, gpu_hr=GPU_HR, cpu_cores=8, mem_gib=16),
    })


@app.local_entrypoint()
def main(batches: str = "", smoke: bool = False, with_scan: bool = False,
         unroll_steps: int = 4, robot: str = ""):
    run.remote(batches=batches, smoke=smoke, with_scan=with_scan,
               unroll_steps=unroll_steps, robot=robot or ROBOT)
