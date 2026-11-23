import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref

r = Ref("block_stack", 411)
m, d = r.model, r.data
# orient gripper identity-ish via a known solve: use current pose; measure pad face gap along gripper body x axis
for g in (1.45, 1.2, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, -0.1, -0.17):
    d.qpos[r.qadr['gripper']] = g
    mujoco.mj_forward(m, d)
    R = d.xmat[r.gbody].reshape(3, 3)
    pf = d.geom_xpos[r.pad_fixed]; pm = d.geom_xpos[r.pad_moving]
    # face planes: fixed face at centre + R@(0.0025,0,0) (half-thick toward +x local? face plane local x=-0.008, centre -0.0105 -> face at +0.0025 from centre)
    # moving face: centre -0.009 local in moving body; use world offset along pinch axis (R x-axis)
    axis = R[:, 0]
    gap = float((pm - pf) @ axis)
    print(f"grip={g:+.2f} pad-centre distance along pinch axis = {gap*1000:.1f} mm  pinch={np.round(r.pinch_centre(),3)}")
