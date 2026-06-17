# bench/ — the wall

Competitor benchmarks: the same Unitree G1, on the same GPU, under identical
physics settings, so the throughput numbers in [../RESULTS.md](../RESULTS.md) are
apples-to-apples. Each app runs on [Modal](https://modal.com) and prints a JSON
blob between `=== ULTRA-BENCH RESULT ===` markers.

| file | engine | physics |
|---|---|---|
| `bench_ours.py`     | nanoG1 (G1-specialized CUDA) | MuJoCo, compile-time specialized |
| `bench_warp.py`     | mujoco_warp (NVIDIA Warp)    | MuJoCo, general-purpose CUDA |
| `bench_mjx.py`      | MJX (JAX/XLA)                | MuJoCo, padded to static shapes |
| `bench_genesis.py`  | Genesis                      | **its own solver** (not MuJoCo) |

## Run — smoke first, always

Every app has a `--smoke` mode: tiny workload on the same hardware class, so
wall-time and cost extrapolate before you pay for the full run. The result blob's
`run_meta` (`total_wall_s`, `est_cost_usd`) lets you calibrate the `RATE_*` cost
constants against your Modal dashboard.

```bash
# pass 1: smoke (minutes, cents)
modal run bench/bench_warp.py --smoke
modal run bench/bench_mjx.py --smoke
modal run bench/bench_genesis.py --smoke

# pass 2: full sweeps on the same card (only after smokes are green)
NANOG1_GPU=RTX-PRO-6000 modal run bench/bench_warp.py --nconmax 32 --njmax 128
NANOG1_GPU=RTX-PRO-6000 modal run bench/bench_mjx.py
NANOG1_GPU=RTX-PRO-6000 modal run bench/bench_genesis.py
```

> **mujoco_warp capacities.** Pass `--nconmax 32 --njmax 128` — these are *per-world*
> contact/constraint capacities (the G1 has nefc≈85). The defaults are far too large
> and OOM; far too small and they silently drop contacts.

## Honesty rules (read before disputing a number)

- **Same model by construction.** The MuJoCo-physics benches (ours/warp/MJX) load
  the same G1 model; the byte-level `mj_saveModel` md5 fingerprints must match. That
  fingerprint is printed in every blob — it *is* the apples-to-apples guarantee.
- **Compile/JIT time is reported separately** from steady-state throughput (XLA jit
  for MJX, kernel compile + CUDA-graph capture for warp/ours).
- **Genesis is a competitor, not a matched datapoint.** Different solver, different
  contact, MJCF reparsed. Raw steps/s across different dt is unit-mismatched, so
  `bench_genesis.py` also reports the dt-normalized `sim_s_per_wall_s`. Quote with
  the caveat.
- **Record GPU, driver, batch size, host core count** with every number (the blobs
  do this). See [../RESULTS.md](../RESULTS.md) for the recorded results + settings.
