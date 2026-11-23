import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, ARM5

r = Ref("block_stack", 411)
m, d = r.model, r.data
print("qpos arm", [round(float(d.qpos[r.qadr[j]]),4) for j in ARM5], "grip", round(float(d.qpos[r.qadr['gripper']]),3))
print("pc0", r.pinch_centre(), "site", r.site_pos())
q0 = d.qpos.copy()
for eps in (1e-6, 1e-4, 1e-3):
    d.qpos[r.qadr['shoulder_pan']] = q0[r.qadr['shoulder_pan']] + eps
    mujoco.mj_forward(m, d)
    print("eps", eps, "pc", r.pinch_centre(), "delta", r.pinch_centre() - np.array([0.1563,0.0079,0.0316]))
    d.qpos[:] = q0; mujoco.mj_forward(m, d)
print("geom xpos fixed", d.geom_xpos[r.pad_fixed], "moving", d.geom_xpos[r.pad_moving])
print("geom bodyids", m.geom_bodyid[r.pad_fixed], m.geom_bodyid[r.pad_moving])
print("body names", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[r.pad_fixed])), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[r.pad_moving])))
