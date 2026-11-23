import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, ARM5

r = Ref("block_stack", 411)
cb = r.body_pos("cube_b")
ca = r.body_pos("cube_a")
print("cube_b", np.round(cb, 3), "cube_a", np.round(ca, 3))
grips = [f[5] for f in r.gt]
gthr = min(grips) + 0.6 * (max(grips) - min(grips))
rows = []
for f in range(len(r.gt)):
    for i, j in enumerate(ARM5):
        r.data.qpos[r.qadr[j]] = math.radians(r.gt[f][i])
    r.data.qpos[r.qadr["gripper"]] = math.radians(r.gt[f][5])
    mujoco.mj_forward(r.model, r.data)
    pc = r.pinch_centre()
    rows.append((f, float(np.linalg.norm(pc - cb)), float(np.linalg.norm(pc - ca)), r.gt[f][5], np.round(pc, 3)))
rows.sort(key=lambda x: x[1])
print("closest to cube_b (all):", rows[:3])
rows_o = [x for x in rows if x[3] >= gthr]
rows_o.sort(key=lambda x: x[1])
print("closest to cube_b (open):", rows_o[:3])
rows.sort(key=lambda x: x[2])
print("closest to cube_a (all):", rows[:3])
