import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import SR
from refcore import Ref

r = Ref("eye_drops_to_basket", 1630)
C0 = r.body_pos("carton")
print("n gt frames", len(r.gt), "carton", np.round(C0, 3))
saved = r.data.qpos.copy()
dists = []
for f in range(len(r.gt)):
    c = SR.action_deg_to_ctrl(r.gt[f])
    for k, j in enumerate(SR.JOINTS):
        r.data.qpos[r.qadr[j]] = c[k]
    mujoco.mj_forward(r.model, r.data)
    dists.append((float(np.linalg.norm(r.pinch_centre() - C0)), f, r.pinch_centre().copy(), float(r.gt[f][5])))
r.data.qpos[:] = saved; mujoco.mj_forward(r.model, r.data)
dists.sort()
for d, f, pc, gd in dists[:6]:
    print(f"frame {f} dist {d*1000:.0f}mm pinch {np.round(pc,3)} grip_deg {gd:.1f}")
print("grip_deg trace sample:", [round(float(r.gt[f][5]),1) for f in range(0, len(r.gt), 10)])
