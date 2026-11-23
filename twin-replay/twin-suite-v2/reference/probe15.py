import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, ARM5

r = Ref("block_stack", 411)
cur = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
print("cur q_deg", np.round(np.degrees(cur), 1), "pinch", np.round(r.pinch_centre(), 3))
tgt = np.array([0.066, -0.023, 0.035])
q, pe, oe, w = r._ik(tgt, np.eye(3), cur, iters=300, w_ori=0.0)
print("from cur: pe_mm", round(pe * 1000, 2), "oe", round(oe, 1), "within", w, "q_deg", np.round(np.degrees(q), 1))
# manual: what does the old-style plain iteration do? emulate with w_ori=0 and no line search via iters and print intermediate by monkeypatching: just report pinch at q
saved = r.data.qpos.copy()
for i, j in enumerate(ARM5):
    r.data.qpos[r.qadr[j]] = q[i]
mujoco.mj_forward(r.model, r.data)
print("pinch at q", np.round(r.pinch_centre(), 4))
r.data.qpos[:] = saved; mujoco.mj_forward(r.model, r.data)
