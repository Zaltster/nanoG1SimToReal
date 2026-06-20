# Results

Curated, honest numbers with provenance. The rule from day one: identical physics
settings across every engine compared, three metrics always reported together, and
compile/JIT time separated from steady-state throughput.

## Time-to-walk

| metric | value |
|---|---|
| **time-to-walk** | **58.9 s** |
| samples-to-walk | 75M control steps |
| steady SPS | 1.28M samples/s (end-to-end: env + inference + learning) |
| cost-to-walk | local Spark run; cloud billing scripts removed |
| GPU | original reference run used RTX PRO 6000-class silicon; refresh this row after a Spark run |
| method | PPO + V-trace + Muon, **pure RL from scratch** (no demos, no reference gait) |
| seed | 42 |

**What "time-to-walk" means.** The policy is trained on a 150M-step schedule (the LR
anneals over the full budget). It crosses the frozen quality gate at ~75M samples;
`time-to-walk = samples-to-walk / steady-SPS = 75M / 1.28M ≈ 59 s`. `train_local.py`
captures the checkpoint nearest 75M and ships it as `assets/nanoG1.bin`. Training is
deterministic per built binary (fixed seed + pinned engine commit).

## Quality gate (the frozen bar)

`eval.py` runs the MuJoCo-validated host-physics battery (the same stepper the
browser demo uses) and checks all six thresholds. Frozen against a reference 116M
checkpoint, approved 2026-06-15:

| check | threshold | meaning |
|---|---|---|
| `battery_falls` | ≤ 1 | falls across the command battery |
| `battery_perf` | ≥ 0.90 | velocity-tracking score |
| `action_jerk_rms` | ≤ 0.21 | action smoothness |
| `ang_vel_xy_rms` | ≤ 0.21 | torso wobble |
| `yaw_rate_rms` | ≤ 0.20 | heading stability |
| `leg_qvel_rms` | ≤ 1.22 | leg-velocity smoothness |

"Passes the gate" is re-provable by one command (`python eval.py assets/nanoG1.bin`),
not by testimony.

## Engine throughput — the wall

G1 reference run, physics steps/s, CUDA-graph-captured steady state
(JIT/compile time excluded). All MuJoCo-physics engines load the **same** G1 model;
the byte-level model fingerprints match (md5 `432c765a`) — that *is* the
apples-to-apples guarantee.

| engine | steps/s | settings | note |
|---|---|---|---|
| **nanoG1 (production)** | **8.9M** | dt 0.004, Newton 2/3 | the solver the shipped policy trains with |
| nanoG1 (matched) | 6.46M | dt 0.002, Newton 3/5 | matched to warp's settings → **1.6× warp** |
| mujoco_warp | 4.0M | dt 0.002, Newton 3/5 | needs `--nconmax 32 --njmax 128` (per-world capacities; G1 nefc≈85) |
| Genesis\* | 2.28M | its own solver | \*different physics — see caveat |
| MJX | 1.12M | dt 0.002, Newton 3/5 | jit(vmap(step)) repeated-step (not lax.scan) |

**Honesty notes.**
- The rigorous, matched-physics claim is **1.6× mujoco_warp** (6.46M vs 4.0M at
  identical dt/solver). The **8.9M** headline is our *production* solver config
  (dt 0.004, Newton 2/3) — the settings the shipped policy actually trains under,
  which is the number that matters for time-to-walk. Both are reported; don't
  conflate them.
- **Genesis runs its own (non-MuJoCo) solver and contact model** and reparses the
  MJCF — it is a *competitor* datapoint, not matched physics. Raw steps/s across
  engines at different dt is unit-mismatched; the original Genesis benchmark also
  reported the dt-normalized `sim_s_per_wall_s`. Quote with the caveat.
- warp/MJX/ours are **bit-comparable** (same model fingerprint). Genesis is not.

## Reproduce

```bash
python train_local.py --smoke
python train_local.py
python eval.py assets/nanoG1.bin
```

The paid cloud benchmark scripts have been removed from this Spark-first checkout.
Refresh this file with local Spark measurements after running the local trainer.
