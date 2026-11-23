import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, Rz, OPEN_RAD, PRE_RAD

r = Ref("eye_drops_to_basket", 1630)
C0 = r.body_pos("carton")
print("carton", np.round(C0, 4), "size x .090 y .035 z .035 (geom half .045/.0175/.0175)")
Rt = Rz(math.pi / 2)
tgt = C0 + [0, 0, 0.006]
print("move_pos above:", r.move_pos(C0 + [0, 0, 0.08], OPEN_RAD))
sol = r.solve(tgt, Rt)
print("solve grasp:", None if sol is None else (round(sol["pos_err"] * 1000, 2), round(sol["ori_err_deg"], 2), sol["q_deg"]))
if sol:
    r.execute_sol(sol, PRE_RAD)
print("pinch actual", np.round(r.pinch_centre(), 4), "target", np.round(tgt, 4))
print("pad_fixed world", np.round(r.data.geom_xpos[r.pad_fixed], 4), "pad_moving world", np.round(r.data.geom_xpos[r.pad_moving], 4))
print("gripper R\n", np.round(r.data.xmat[r.gbody].reshape(3, 3), 3))
print("carton contacts", r.contacts("carton"))
q6 = [r.data.ctrl[r.act[j]] for j in __import__("refcore").SR.JOINTS]; q6[5] = -0.16
r.run_to(q6, 1.2)
print("after close: stall", round(r.grip_q(), 3), "carton", np.round(r.body_pos("carton"), 4), "contacts", r.contacts("carton"))
