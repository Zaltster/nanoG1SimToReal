#!/usr/bin/env python3
"""Bake the G1 visual meshes from g1.mjb into a compact binary asset
(web/g1_meshes.bin) the host/WASM demo loads at runtime. The C/WASM demo can't
call libmujoco, so the render geometry travels as a preloaded file.

Only the UNIQUE meshes are stored (35); the demo instances them per mesh-geom
via hc_geom_dataid (baked into g1_model_const.h) and the geom world pose.

Layout (little-endian):
  int32  magic = 'G1MS' (0x47314D53)
  int32  nmesh
  nmesh x { int32 vertnum, int32 facenum }
  verts: for each mesh, vertnum*3  float32   (mesh-local coords)
  faces: for each mesh, facenum*3  int32     (vertex indices into that mesh)

Usage: python scripts/dump_g1_meshes.py [g1.mjb] [out.bin]
"""
import sys, struct, mujoco, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
mjb = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "envs/g1/model/g1.mjb")
out = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "web/g1_meshes.bin")
m = mujoco.MjModel.from_binary_path(mjb)

buf = bytearray()
buf += struct.pack("<ii", 0x47314D53, m.nmesh)
for i in range(m.nmesh):
    buf += struct.pack("<ii", int(m.mesh_vertnum[i]), int(m.mesh_facenum[i]))
# verts (mesh-local, float32) — m.mesh_vert is already mesh-frame == geom-frame
for i in range(m.nmesh):
    a, n = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
    buf += m.mesh_vert[a:a+n].astype("<f4").tobytes()
# faces (int32 triplets, local to each mesh's vert block)
for i in range(m.nmesh):
    a, n = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
    buf += m.mesh_face[a:a+n].astype("<i4").tobytes()

Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_bytes(buf)
print(f"wrote {out}: nmesh={m.nmesh} verts={m.mesh_vert.shape[0]} "
      f"faces={m.mesh_face.shape[0]} bytes={len(buf)} ({len(buf)/1e6:.1f}MB)")
