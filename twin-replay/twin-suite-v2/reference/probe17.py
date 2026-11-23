import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, OPEN_RAD, ARM5

r = Ref("eye_drops_to_basket", 1630)
print("start pinch", np.round(r.pinch_centre(), 4), "qpos_deg", np.round(np.degrees([r.data.qpos[r.qadr[j]] for j in ARM5]), 1))
tgt = r.body_pos("carton") + [0, 0, 0.08]
sols = r.continuation(tgt)
print("continuation sols:", None if sols is None else len(sols))
if sols:
    saved = r.data.qpos.copy()
    for i, j in enumerate(ARM5):
        r.data.qpos[r.qadr[j]] = sols[-1][i]
    mujoco.mj_forward(r.model, r.data)
    print("pinch at final sol", np.round(r.pinch_centre(), 4), "target", np.round(tgt, 4))
    r.data.qpos[:] = saved; mujoco.mj_forward(r.model, r.data)
    r.execute_path(sols, OPEN_RAD)
    print("pinch after execute_path", np.round(r.pinch_centre(), 4))
    print("qpos_deg now", np.round(np.degrees([r.data.qpos[r.qadr[j]] for j in ARM5]), 1), "ctrl_deg", np.round(np.degrees([r.data.ctrl[r.act[j]] for j in ARM5]), 1))
