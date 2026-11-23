import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
import refcore as _rc; SR = _rc.SR
from refcore import Ref, ARM5

r = Ref("block_stack", 411)
B0 = r.body_pos("cube_b")
tgt = B0 + [0, 0, 0.030]
m, d = r.model, r.data
saved = d.qpos.copy()
best = []
for f in range(len(r.gt)):
    c = SR.action_deg_to_ctrl(r.gt[f])
    for k, j in enumerate(SR.JOINTS):
        d.qpos[r.qadr[j]] = c[k]
    mujoco.mj_forward(m, d)
    pc = r.pinch_centre()
    best.append((float(np.linalg.norm(pc - tgt)), f, pc.copy(), d.xmat[r.gbody].reshape(3,3).copy(), np.radians(r.gt[f][:5])))
best.sort(key=lambda x: x[0])
for dist, f, pc, R, q in best[:5]:
    print(f"frame {f} dist_mm={dist*1000:.1f} pc={np.round(pc,4)} grip_deg={r.gt[f][5]:.1f} q_deg={np.round(np.degrees(q),1)}")
d.qpos[:] = saved; mujoco.mj_forward(m, d)
# now IK with Rt from best frame and its seed
dist, f, pc, R, q = best[0]
qq, pe, oe, within = r._ik(tgt, R, q)
print("IK from demo frame:", "pos_mm", round(pe*1000,2), "ori_deg", round(oe,2), "within", within, "q_deg", np.round(np.degrees(qq),1))
# position-only scan: what orientation does the closest demo frame have vs identity?
print("R_demo=\n", np.round(R, 3))
