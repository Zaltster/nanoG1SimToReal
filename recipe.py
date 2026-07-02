"""nanoG1 — the frozen winning recipe (the one dial you turn).

A Unitree G1 humanoid learns to walk from scratch in <60s with exactly these
settings. Every number was found by a hyperparameter sweep and validated
trajectory-by-trajectory against the MuJoCo C engine — change at your peril.

The speed comes from per-robot *compile-time specialization*: the G1's kinematics,
contact set, and solver layout are baked into the engine as constants (the pinned
fork below), and the policy is regularized with a left<->right symmetry loss (N1).
"""

# --- the engine: a G1-specialized fork of PufferLib (built at train time) ---
FORK        = "https://github.com/kingjulio8238/PufferLib"
FORK_BRANCH = "g1"
FORK_PIN    = "e3825cea"            # pinned commit -> deterministic builds

# --- compile-time flags ---
# v3 task physics: dt 0.004 x decimation 5 (50 Hz control), truncated Newton solver
# (2 iters / 3 line-search — validated sufficient for the G1), Unitree PD gains.
TASK_FLAGS  = (
    "-DG1_DT=0.004f -DENV_DECIMATION=5 -DSOL_ITER=2 -DSOL_LS_ITER=3 "
    "-DG1_TASK_V3 -DG1_PD_UNITREE "
    "-DG1_DOMAIN_RANDOMIZATION "
    "-DG1_REWARD_ANTISTALL "
    "-DG1_DR_RESET_QPOS=0.10f "
    "-DG1_DR_RESET_LINVEL=0.30f -DG1_DR_RESET_ANGVEL=0.40f "
    "-DG1_DR_CMD_MIN=0.55f -DG1_DR_CMD_MAX=1.25f "
    "-DG1_DR_MOTOR_MIN=0.80f -DG1_DR_MOTOR_MAX=1.20f "
    "-DG1_DR_FRICTION_MIN=0.55f -DG1_DR_FRICTION_MAX=1.25f "
    "-DG1_ZERO_CMD_PROB=0.02f "
    "-DG1_NONZERO_CMD_MIN=0.20f -DG1_ANTISTALL_VEL=0.12f -DG1_W_ANTISTALL=-4.0f "
    "-DG1_V3_MIN_FOOT_SEP=0.10f -DG1_V3_W_FOOT_SEP=-12.0f "
    "-DG1_PUSH_MIN_TICKS=90 -DG1_PUSH_MAX_TICKS=240 "
    "-DG1_PUSH_PROB=0.50f -DG1_PUSH_LINVEL=0.35f -DG1_PUSH_ANGVEL=0.50f "
    "-DG1_PUSH_ZVEL=0.08f"
)
# N1: left<->right symmetry loss, coefficient 0.25 — the breakthrough that cut
# samples-to-walk ~26% AND smoothed the gait (quality-positive).
TRAIN_FLAGS = "-DG1_MIRROR_LOSS=0.25"
DECIMATION  = 5

# --- budget ---
TOTAL_TIMESTEPS = 150_000_000      # the schedule (LR anneals over this)
WALK_SAMPLES    = 75_000_000       # samples-to-walk: the G1 passes the gate here ≈ 59s

# --- runtime config (config/g1gpu.ini [env]/[train]) — the sub-60 recipe ---
RECIPE = {
    # reward (gated velocity-tracking; w_ang_vel_xy=-1.3 is the torso-wobble bump
    # that pulled the gate-crossing under 60s)
    "env.action_scale": 0.25, "env.max_episode_len": 1000,
    "env.w_track_lin": 5.0, "env.w_track_ang": 1.25, "env.w_lin_vel_z": -2.0,
    "env.w_ang_vel_xy": -1.3, "env.w_orientation": -10.0, "env.w_torque": -2e-05,
    "env.w_action_rate": -0.01, "env.w_alive": 0.5, "env.w_termination": -3.0,
    # PPO + V-trace + prioritized replay, Muon optimizer (the swept winner)
    "train.seed": 42, "train.learning_rate": 0.02, "train.anneal_lr": 1,
    "train.min_lr_ratio": 0, "train.gamma": 0.97, "train.gae_lambda": 0.9,
    "train.replay_ratio": 3.0, "train.clip_coef": 0.2, "train.vf_coef": 0.5,
    "train.vf_clip_coef": 20, "train.max_grad_norm": 0.3, "train.ent_coef": 1e-05,
    "train.anneal_ent_coef": 0, "train.min_ent_coef_ratio": 0.1,
    "train.beta1": 0.9, "train.beta2": 0.999, "train.eps": 1e-12,
    "train.minibatch_size": 32768, "train.horizon": 64,
    "train.vtrace_rho_clip": 3.0, "train.vtrace_c_clip": 3.0,
    "train.prio_alpha": 0.4, "train.prio_beta0": 1,
    "base.checkpoint_interval": 10,
}


def overrides_str() -> str:
    """Comma-separated section.key=val string the trainer applies to config/g1gpu.ini."""
    return ",".join(f"{k}={v}" for k, v in RECIPE.items())
