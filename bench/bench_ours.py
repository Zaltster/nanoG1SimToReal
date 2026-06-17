"""P2: g1phys kernel dev-loop + benchmark on Modal.

Code (tests/, src/g1phys/) is MOUNTED at runtime — edit locally, re-run, no
image rebuild. The image bakes the canonical model + toolchain. Each run:
  1. compile record_traj (cc) against the pip mujoco wheel
  2. generate reference trajectories (air/stand/random) + bit-exact self-check
  3. nvcc-compile the kernel target (-arch=native, GPU present)
  4. validate vs references + time batch sweeps -> blob

Run:
  modal run bench/bench_ours.py --smoke --target smooth   # T4 dev loop (~$0.002/run)
  modal run bench/bench_ours.py --target smooth --gpu h100  # gate numbers
  modal run bench/bench_ours.py --target proto_mapping --batches "4096,16384"

Targets: proto_mapping (A2 experiment), smooth (B: smooth-dynamics step),
contact (C: full physics step), env (D: full environment — physics + episode
machinery validated against the REAL CPU env g1.h; rollout = the GATE-D number).
"""

import json
import re
import subprocess
import time

import modal

RATE_GPU = {"T4": 0.59, "L4": 0.80, "H100": 3.95, "RTX-PRO-6000": 3.03}
RATE_CPU_CORE_HR = 0.0473
RATE_MEM_GIB_HR = 0.0080
CPU_CORES, MEM_GIB = 4, 8

app = modal.App("ultra-g1phys")

def _make_image(cuda_ver):
    return (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_ver}-devel-ubuntu22.04",
                              add_python="3.11")
    .env({"PYTHONUNBUFFERED": "1", "DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("git", "curl", "clang")
    .pip_install("mujoco==3.9.0", "playground==0.2.0", "jax", "numpy")
    .add_local_file("scripts/extract_g1_model.py", "/root/extract_g1_model.py",
                    copy=True)
    .run_commands("G1_MODEL_DIR=/root/envs/g1/model python /root/extract_g1_model.py")
    # runtime mounts (no rebuild on code edits):
    .add_local_dir("tests", "/work/tests")
    .add_local_dir("src/g1phys", "/work/src/g1phys")
    # the REAL CPU env (vendor submodule) — reference for the env target
    .add_local_file("vendor/PufferLib/ocean/g1/g1.h", "/work/g1env/g1.h")
    )

image = _make_image("12.6.3")
# Blackwell (sm_120, e.g. RTX PRO 6000 = RTX 5090 silicon) needs CUDA >= 12.8
image128 = _make_image("12.8.1")


def _sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    if r.stdout: print(r.stdout, flush=True)
    if r.stderr: print(r.stderr, flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed (rc={r.returncode}): {cmd}\n"
                           f"--- stderr tail ---\n{r.stderr[-3000:]}")
    return r


# Task physics variants. v1 = the frozen Phase-0 wall (dt 0.002 x 10, Newton
# 3/5). v2* are NEW TASK CONFIGS (legged-gym-standard timestep; same 50Hz
# control) — references are re-recorded at matched settings in the same run,
# so the validation contract is identical. v1 numbers stay quoted separately.
TASKS = {
    "v1":  None,
    "v2":  {"dt": 0.004, "dec": 5, "iters": 3, "ls": 5},
    "v2s": {"dt": 0.004, "dec": 5, "iters": 2, "ls": 3},
    "v25": {"dt": 0.005, "dec": 4, "iters": 3, "ls": 5},
    # v25s = v25 dt with the v2s truncated solver (Newton 2/ls 3). The v2 lesson
    # (idiot_index §6): Newton 3/5 forks near MuJoCo's early-termination boundary
    # at coarse dt; only the truncated 2/3 validates. v25 (3/5) is the suspect,
    # v25s the likely-to-pass sibling. check_oracle_v25.py says which to try.
    "v25s": {"dt": 0.005, "dec": 4, "iters": 2, "ls": 3},
    # v3 = v2s physics + gait shaping (phase obs 98, contact/swing/hip
    # rewards, 12-DOF action masking) — G1_TASK_V3 in BOTH compilers
    "v3":  {"dt": 0.004, "dec": 5, "iters": 2, "ls": 3, "extra": "-DG1_TASK_V3 -DG1_PD_UNITREE"},
    # v3i1 = v3 physics + gait shaping with Newton 2->1 (the SOL_ITER cut probe). Both
    # GPU stagedenv AND the CPU/MuJoCo refs build at iters=1, so check_traj (vs MuJoCo
    # @ Newton-1) confirms our impl is correct at this setting and stagedenv_validate
    # confirms GPU==CPU; the gpu_busy drop vs v3 = the realized solver-iter saving.
    "v3i1": {"dt": 0.004, "dec": 5, "iters": 1, "ls": 3, "extra": "-DG1_TASK_V3 -DG1_PD_UNITREE"},
}


def _run_bench(gpu_name: str, target: str, batches: str, iters: int, smoke: bool,
               nvcc_flags: str = "", task: str = "v1"):
    t0 = time.perf_counter()
    import os
    os.chdir("/work")
    tv = TASKS[task]
    rec_env = ""      # env-var prefix for the reference recorders
    cc_defs = ""      # defines for the CPU env (record_env)
    if tv:
        rec_env = (f"G1_TIMESTEP={tv['dt']} G1_SOLITER={tv['iters']} "
                   f"G1_LSITER={tv['ls']} ")
        extra = tv.get("extra", "")
        cc_defs = (f"-DG1_WALL_TIMESTEP={tv['dt']} -DG1_DECIMATION={tv['dec']} "
                   f"-DG1_WALL_ITERATIONS={tv['iters']} "
                   f"-DG1_WALL_LS_ITERATIONS={tv['ls']} {extra}")
        nvcc_flags = (f"{nvcc_flags} -DG1_DT={tv['dt']}f "
                      f"-DENV_DECIMATION={tv['dec']} -DSOL_ITER={tv['iters']} "
                      f"-DSOL_LS_ITER={tv['ls']} {extra}")
        print(f"TASK {task}: dt={tv['dt']} x decimation {tv['dec']} "
              f"(ctrl 50Hz), Newton {tv['iters']}/ls {tv['ls']}", flush=True)
    os.makedirs("/work/traj", exist_ok=True)
    _sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")

    mj = subprocess.run(
        ["python", "-c", "import mujoco,os;print(os.path.dirname(mujoco.__file__))"],
        capture_output=True, text=True).stdout.strip()
    mjlib = subprocess.run(f"ls {mj}/libmujoco.so.* | head -1", shell=True,
                           capture_output=True, text=True).stdout.strip()

    # 1-2: reference trajectories + self-check (the A1 harness, in-container)
    _sh(f"cc -O2 tests/record_traj.c -I{mj}/include {mjlib} -Wl,-rpath,{mj} "
        f"-o /work/record_traj -lm")
    _sh(f"cc -O2 tests/check_traj.c -I{mj}/include {mjlib} -Wl,-rpath,{mj} "
        f"-o /work/check_traj -lm")
    # solver constants for the contact target (fp32 sidecar from the model)
    _sh("python tests/dump_solver_consts.py /root/envs/g1/model/g1.mjb "
        "traj/g1_solver_consts.bin")

    checks = []
    for scen in ("air", "stand", "random"):
        # air is the Gate-B rollout horizon (1000 steps, constraint-disabled)
        nsteps = 100 if smoke else (1000 if scen == "air" else 400)
        _sh(f"{rec_env}G1_MODEL_PATH=/root/envs/g1/model/g1.mjb ./record_traj {scen} {nsteps} 7 "
            f"traj/{scen}.bin /root/envs/g1/model/g1.mjb")
        out = subprocess.run(f"{rec_env}./check_traj traj/{scen}.bin /root/envs/g1/model/g1.mjb",
                             shell=True, capture_output=True, text=True).stdout
        print(out, flush=True)
        checks.append("BIT-EXACT" in out)

    # 3: compile the kernel target
    print(f"nvcc compiling {target} (-arch=native)...", flush=True)
    _sh(f"nvcc -O3 -arch=native --ptxas-options=-v {nvcc_flags} -Itests "
        f"-Isrc/g1phys src/g1phys/{target}.cu -o /work/{target}")
    # SASS code size (icache-pressure diagnostic)
    _sh(f"cuobjdump --dump-sass /work/{target} | wc -l")

    # 4: validate + sweep
    batch_list = [int(b) for b in batches.split(",") if b] or ([1024] if smoke else [4096, 16384])
    results = []
    if target in ("env", "stagedenv"):
        # full-env targets: reference comes from the REAL CPU env (g1.h)
        env_steps = 100 if smoke else 300
        _sh(f"cc -O2 {cc_defs} tests/record_env.c -I{mj}/include -I/work/g1env {mjlib} "
            f"-Wl,-rpath,{mj} -o /work/record_env -lm")
        _sh(f"G1_MODEL_PATH=/root/envs/g1/model/g1.mjb ./record_env {env_steps} 7 traj/env.bin")
        for b in batch_list:
            out = subprocess.run(f"./{target} traj/env.bin {b} {iters}",
                                 shell=True, capture_output=True, text=True).stdout
            print(f"--- env batch={b} ---\n{out}", flush=True)
            for line in out.splitlines():
                if line.startswith("RESULT"):
                    entry = {"scenario": "env"}
                    entry.update(dict(kv.split("=") for kv in line.split()[2:]))
                    entry["kernel"] = line.split()[1]
                    results.append(entry)
    else:
        for scen in ("air", "stand", "random"):
            for b in batch_list:
                out = subprocess.run(f"./{target} traj/{scen}.bin {b} {iters}",
                                     shell=True, capture_output=True, text=True).stdout
                print(f"--- {scen} batch={b} ---\n{out}", flush=True)
                for line in out.splitlines():
                    if line.startswith("RESULT"):
                        entry = {"scenario": scen}
                        entry.update(dict(kv.split("=") for kv in line.split()[2:]))
                        entry["kernel"] = line.split()[1]
                        results.append(entry)

    elapsed = time.perf_counter() - t0
    rate = RATE_GPU[gpu_name] + CPU_CORES * RATE_CPU_CORE_HR + MEM_GIB * RATE_MEM_GIB_HR
    blob = {
        "engine": "g1phys", "target": target,
        "task": task, "gpu": gpu_name,
        "harness_self_check": all(checks),
        "results": results,
        "run_meta": {"run_mode": "smoke" if smoke else "full",
                     "total_wall_s": round(elapsed, 1),
                     "rate_usd_per_hr": round(rate, 2),
                     "est_cost_usd": round(elapsed / 3600 * rate, 3)},
    }
    print("\n=== ULTRA-BENCH RESULT ===")
    print(json.dumps(blob, indent=2))
    print("=== END RESULT ===\n")


@app.function(image=image, gpu="T4", cpu=float(CPU_CORES), memory=MEM_GIB * 1024,
              timeout=1800)
def run_t4(target: str = "proto_mapping", batches: str = "", iters: int = 50,
           smoke: bool = False, nvcc_flags: str = "", task: str = "v1"):
    _run_bench("T4", target, batches, iters, smoke, nvcc_flags, task)


@app.function(image=image, gpu="H100", cpu=float(CPU_CORES), memory=MEM_GIB * 1024,
              timeout=1800)
def run_h100(target: str = "proto_mapping", batches: str = "", iters: int = 50,
             smoke: bool = False, nvcc_flags: str = "", task: str = "v1"):
    _run_bench("H100", target, batches, iters, smoke, nvcc_flags, task)


@app.function(image=image128, gpu="RTX-PRO-6000", cpu=float(CPU_CORES),
              memory=MEM_GIB * 1024, timeout=3600)
def run_pro6000(target: str = "proto_mapping", batches: str = "", iters: int = 50,
                smoke: bool = False, nvcc_flags: str = "", task: str = "v1"):
    _run_bench("RTX-PRO-6000", target, batches, iters, smoke, nvcc_flags, task)


@app.local_entrypoint()
def main(gpu: str = "t4", target: str = "proto_mapping", batches: str = "",
         iters: int = 50, smoke: bool = False, nvcc_flags: str = "", task: str = "v1"):
    g = gpu.lower()
    fn = run_h100 if g == "h100" else (run_pro6000 if g in ("pro6000", "rtx-pro-6000") else run_t4)
    fn.remote(target=target, batches=batches, iters=iters, smoke=smoke,
              nvcc_flags=nvcc_flags, task=task)
