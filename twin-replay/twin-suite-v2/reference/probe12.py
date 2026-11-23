import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, Rz, Ry, ARM5

r = Ref("block_stack", 411)
A0, B0 = r.body_pos("cube_a"), r.body_pos("cube_b")
g = r.grasp(A0, yaw=0.0, dz=0.0, close_target=-0.16)
print("grasp", g["stall"] if g else None)
r.lift(g["hold"], 0.07, Rt=g["Rt"])
o_body, Rg = r.held_offset_body("cube_a")
print("o_body", np.round(o_body, 4), "|o| mm", round(float(np.linalg.norm(o_body)) * 1000, 1))
obj_tgt = B0 + [0, 0, 0.033]
# position-only at pinch_tgt for Rt=I
pinch_tgt = obj_tgt - o_body
print("pinch_tgt (Rt=I)", np.round(pinch_tgt, 4))
cur = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
rng = np.random.RandomState(20261124)
seeds = [cur] + [cur + np.array([0, rng.uniform(-0.8, 0.8), rng.uniform(-1, 1), rng.uniform(-1.2, 1.2), 0]) for _ in range(10)]
best = None
for s in seeds:
    q, pe, oe, within = r._ik(pinch_tgt, np.eye(3), s, w_ori=0.0)
    if within and (best is None or pe < best[1]):
        best = (q, pe)
print("pos-only best pe_mm", None if best is None else round(best[1] * 1000, 2))
if best:
    saved = r.data.qpos.copy()
    for i, j in enumerate(ARM5):
        r.data.qpos[r.qadr[j]] = best[0][i]
    mujoco.mj_forward(r.model, r.data)
    Rq = r.data.xmat[r.gbody].reshape(3, 3).copy()
    print("R at solution\n", np.round(Rq, 3), "q_deg", np.round(np.degrees(best[0]), 1))
    r.data.qpos[:] = saved; mujoco.mj_forward(r.model, r.data)
    pinch2 = obj_tgt - Rq @ o_body
    q2, pe2, oe2, w2 = r._ik(pinch2, Rq, best[0])
    print("full IK from sol seed: pe_mm", round(pe2 * 1000, 2), "oe_deg", round(oe2, 2), "within", w2)
    sol = r.solve(pinch2, Rq)
    print("solve():", None if sol is None else (round(sol["pos_err"] * 1000, 2), round(sol["ori_err_deg"], 2)))
