import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, Rz, ARM5

r = Ref("eye_drops_to_basket", 1630)
C0 = r.body_pos("carton")
gf = r.demo_frame_nearest(C0)
print("grasp frame", gf)
r.follow_demo(gf)
print("pinch after demo", np.round(r.pinch_centre(), 4), "carton", np.round(C0, 4))
tgt = C0 + [0, 0, 0.006]
# position-only candidates without torque filter
cur = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
cands = []
for s in [cur] + [np.radians(r.gt[f][:5]) for f in (gf, gf + 10, 90, 110)]:
    q, pe, oe, w = r._ik(tgt, np.eye(3), s, iters=200, w_ori=0.0)
    print("seed pe_mm", round(pe * 1000, 2), "within", w, "torque_ok", r.torque_ok(q) if w else None, "q_deg", np.round(np.degrees(q), 1))
    if w:
        cands.append((pe, q))
# gravity torques at best candidate
if cands:
    cands.sort(key=lambda x: x[0])
    q = cands[0][1]
    sc = mujoco.MjData(r.model)
    sc.qpos[:] = r.data.qpos[:]
    for i, j in enumerate(ARM5):
        sc.qpos[r.qadr[j]] = q[i]
    sc.qvel[:] = 0; sc.qacc[:] = 0
    mujoco.mj_inverse(r.model, sc)
    print("gravity torques at best:", [round(float(sc.qfrc_inverse[r.vadr[j]]), 2) for j in ARM5])
print("solve_position_only:", r.solve_position_only(tgt) is not None)
print("continuation:", r.continuation(tgt) is not None)
