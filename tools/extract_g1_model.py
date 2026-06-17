"""Extract the frozen G1 benchmark model (playground G1JoystickFlatTerrain).

Saves the COMPILED model as envs/g1/model/g1.mjb (binary — byte-identical
physics across loads) plus a reference XML, and verifies the md5 against the
Phase-0 wall fingerprint. Pin: mujoco==3.9.0, playground==0.2.0.

Run:  .venv/bin/python scripts/extract_g1_model.py
"""

import hashlib
import os
import pathlib
import sys

import mujoco
import numpy as np
from mujoco_playground import registry

# Canonical fingerprint (linux-x86_64, the Phase-0 wall; docs/baselines.md).
# The Modal training image runs this script at BUILD time and hard-fails on
# mismatch — that's where canonical identity is enforced.
WALL_MD5_LINUX = "432c765a0ac7b68800af4a22d446f7d3"
# macOS-arm64 compiles the same model to different bytes (libm 1-ulp diffs in
# compile-time constants; structure nq=36/nv=35/nu=29/ngeom=72 identical).
# Local mjb is DEV-ONLY (gitignored, 83 MB w/ embedded visual meshes).
KNOWN_MD5_MACOS_ARM64 = "1cbf77af62c1bc69ea3f4e2d8a32af0a"

WALL_MD5 = WALL_MD5_LINUX
OUT_DIR = pathlib.Path(os.environ.get(
    "G1_MODEL_DIR",
    pathlib.Path(__file__).resolve().parents[1] / "envs" / "g1" / "model"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = registry.load("G1JoystickFlatTerrain", config_overrides={"impl": "jax"})
    m = env.mj_model

    sz = mujoco.mj_sizeModel(m)
    buf = np.empty(sz, dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    md5 = hashlib.md5(buf.tobytes()).hexdigest()

    mjb = OUT_DIR / "g1.mjb"
    mjb.write_bytes(buf.tobytes())
    print(f"wrote {mjb}  ({sz/1e6:.2f} MB)")

    try:  # best-effort human-readable reference
        mujoco.mj_saveLastXML(str(OUT_DIR / "g1_reference.xml"), m)
        print("wrote g1_reference.xml")
    except Exception as e:
        print(f"(xml reference skipped: {e!r})")

    # structural identity always required
    assert (m.nq, m.nv, m.nu, m.nbody, m.ngeom) == (36, 35, 29, 31, 72), \
        f"structure mismatch: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}"
    print(f"model: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} (structure ✓)")
    print(f"mjb md5:       {md5}")
    print(f"canonical:     {WALL_MD5_LINUX} (linux-x86_64 — enforced in the Modal image build)")

    if md5 == WALL_MD5_LINUX:
        print("MATCH — canonical model identity verified against the Phase-0 wall")
        return 0
    if md5 == KNOWN_MD5_MACOS_ARM64:
        print("macOS-arm64 dev build — known platform bytes; OK for LOCAL DEV ONLY")
        return 0
    print("UNKNOWN fingerprint — investigate before using this model anywhere")
    return 1


if __name__ == "__main__":
    sys.exit(main())
