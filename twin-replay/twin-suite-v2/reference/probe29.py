import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz, OPEN_RAD, PRE_RAD, ARM5

r = Ref("eye_drops_to_basket", 1630)
C0 = r.body_pos("carton")
tgt = C0 + [0, 0, 0.006]
pc0 = r.pinch_centre()
print("raise:", r.move_pos(pc0 + [0, 0, 0.08], OPEN_RAD))
print("above:", r.move_pos(C0 + [0, 0, 0.08], OPEN_RAD))
print("pinch now", np.round(r.pinch_centre(), 4))
for name, Rt in [("yaw90", Rz(math.pi / 2)), ("yaw0", Rz(0))]:
    seed = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
    for zt in (tgt + [0, 0, 0.025], tgt + [0, 0, 0.012], tgt):
        q, pe, oe, w = r._ik(zt, Rt, seed, iters=500)
        print(f"{name} link z={zt[2]:.3f}: pe={pe*1000:.1f} oe={oe:.1f} within={w} torque={r.torque_ok(q) if w else '-'}")
        if not (w and pe <= 0.006 and oe <= 6.0 and r.torque_ok(q)):
            sol = r.solve_link(zt, Rt)
            print("   solve_link:", None if sol is None else (round(sol["pos_err"] * 1000, 1), round(sol["ori_err_deg"], 1)))
            if sol is None:
                break
            q = sol["q"]
        seed = q
