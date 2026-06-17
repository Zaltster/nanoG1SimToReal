# nanoG1

**Train a [Unitree G1](https://www.unitree.com/g1) humanoid to walk in under 60 seconds, on a single GPU — pure RL, from scratch.**

No demonstrations, no reference gait, no motion capture. The policy starts from noise and learns to walk from reward alone in **~59 seconds** of wall-clock training (~75M samples at 1.28M samples/s) for about **$0.17** on one GPU.

🤖 **[Live demo — drive the trained G1 in your browser](https://g1-sub60-walk.vercel.app)** &nbsp;·&nbsp; 🤗 **[Model on Hugging Face](https://huggingface.co/kingJulio/nanoG1)**

![nanoG1](assets/preview.png)

This is to robot locomotion what [nanoGPT](https://github.com/karpathy/nanoGPT) is to language models: the smallest, most legible thing that actually works, that you can read top-to-bottom and run yourself.

---

## Quickstart

```bash
git clone https://github.com/kingjulio8238/nanoG1 && cd nanoG1
bash speedrun.sh
```

That's it. `speedrun.sh` syncs the Python env, fetches the engine, trains the G1 on a GPU (via [Modal](https://modal.com)), gates the result, and drops the trained policy at `assets/nanoG1.bin`.

**Prereqs:** [`uv`](https://docs.astral.sh/uv), a [Modal](https://modal.com) account (`modal token new`), and `git`. The GPU run is the only paid part (~$0.17 on an RTX PRO 6000); everything else is local and free.

Want to turn the dials yourself instead of one-shotting it:

```bash
bash setup.sh                          # fetch the G1-specialized engine (pinned fork)
modal run train.py --smoke             # ~$0.02 — validate the whole stack first
modal run train.py                     # the <60s walk -> assets/nanoG1.bin
python eval.py assets/nanoG1.bin       # quality gate: does it actually walk?
bash web/build_demo.sh && ./build/g1demo assets/nanoG1.bin   # watch it locally
```

Train on a different card: `NANOG1_GPU=H100 modal run train.py`.

---

## What you get

| | |
|---|---|
| **Time-to-walk** | **58.9 s** (75M samples @ 1.28M SPS, single RTX PRO 6000) |
| **Cost-to-walk** | **~$0.17** |
| **Method** | PPO + V-trace, **pure RL from scratch** — no demos, no reference motion |
| **Physics** | MuJoCo-grade soft-convex contact, friction cones, domain randomization |
| **Engine throughput** | **8.9M physics steps/s** on one GPU (see chart below) |

### Engine throughput — G1, single RTX PRO 6000, physics steps/s

```
nanoG1        ████████████████████████████████████  8.9M
mujoco_warp   ████████████████                       4.0M
Genesis*      █████████                               2.3M
MJX           ████▌                                   1.1M
```

\* Genesis runs its own (non-MuJoCo) solver — a competitor datapoint, not matched-physics. See [RESULTS.md](RESULTS.md) for exact settings, the matched-physics comparison (1.6× warp), and provenance.

---

## How it works

The thesis: MuJoCo's physics isn't inherently slow for RL — it's just never been **specialized**. nanoG1 compiles the simulator *per-robot*. For a fixed G1, the kinematic tree, contact set, and solver layout are compile-time constants, so the whole step inlines into straight-line CUDA with no runtime dispatch, no broadphase, and a fixed-iteration solver. That's where the throughput comes from — not from cheapening the physics (it's validated trajectory-by-trajectory against the MuJoCo C engine).

Two ingredients make it learn to walk this fast:

1. **A G1-specialized GPU engine** — a [pinned PufferLib fork](https://github.com/kingjulio8238/PufferLib/tree/g1) that bakes the G1 in at compile time (zero Python in the hot loop). `recipe.py` pins the exact commit.
2. **A left↔right symmetry loss** (N1, after [Yu et al. 2018](https://arxiv.org/abs/1801.08093)) — regularizing the policy toward a mirror-symmetric gait cut samples-to-walk ~26% *and* smoothed the gait. That's the single biggest lever.

Everything else — the reward weights, PPO/Muon hyperparameters, the dt/decimation/solver settings — lives in **one file, [`recipe.py`](recipe.py)**. That's the dial you turn.

---

## Repo layout

```
recipe.py        the frozen winning recipe — the one dial you turn
train.py         Modal launcher: builds the engine, trains, pulls the walk checkpoint
eval.py          quality gate — runs the host-physics battery, checks it walks
speedrun.sh      one command: env -> engine -> train -> gate
setup.sh         fetch the pinned G1 engine (for local demo/eval builds)
web/             browser demo (raylib + the policy, host physics) -> WASM
bench/           competitor benchmarks (warp / MJX / Genesis) — same card, same G1
tools/           bake the G1 model + meshes from MuJoCo (assets are committed)
assets/nanoG1.bin   the trained <60s policy (655 KB)
```

---

## Credits

Built on [PufferLib](https://github.com/PufferAI/PufferLib)'s compile-time-specialization training engine, [MuJoCo](https://github.com/google-deepmind/mujoco) physics semantics, the [Unitree G1](https://github.com/google-deepmind/mujoco_menagerie) from MuJoCo Menagerie, and [raylib](https://github.com/raysan5/raylib) for the demo. Compute on [Modal](https://modal.com). Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat).

MIT licensed.
