import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, Rz, OPEN_RAD, PRE_RAD, SR

r = Ref("eye_drops_to_basket", 1630)
C0 = r.body_pos("carton")
print("carton", np.round(C0, 4), "x span", C0[0] - 0.045, C0[0] + 0.045, "y span", C0[1] - 0.0175, C0[1] + 0.0175, "z span", C0[2] - 0.0175, C0[2] + 0.0175)
Rt = Rz(math.pi / 2)
tgt = C0 + [0, 0, 0.006]
r.move_pos(C0 + [0, 0, 0.08], OPEN_RAD)
sol = r.solve(tgt, Rt)
print("solve:", None if sol is None else (round(sol["pos_err"] * 1000, 2), round(sol["ori_err_deg"], 2)))
if sol:
    r.execute_sol(sol, PRE_RAD)
else:
    sols = r.continuation(tgt)
    print("continuation:", sols is not None)
    if sols: r.execute_path(sols, PRE_RAD)
print("pinch", np.round(r.pinch_centre(), 4), "target", np.round(tgt, 4))
print("pad_fixed", np.round(r.data.geom_xpos[r.pad_fixed], 4), "pad_moving", np.round(r.data.geom_xpos[r.pad_moving], 4))
print("gripper R\n", np.round(r.data.xmat[r.gbody].reshape(3, 3), 3))
print("carton contacts", r.contacts("carton"))
q6 = [r.data.ctrl[r.act[j]] for j in SR.JOINTS]; q6[5] = -0.16
r.run_to(q6, 1.2)
print("stall", round(r.grip_q(), 3), "carton", np.round(r.body_pos("carton"), 4), "contacts", r.contacts("carton"))
print("pad_fixed", np.round(r.data.geom_xpos[r.pad_fixed], 4), "pad_moving", np.round(r.data.geom_xpos[r.pad_moving], 4))
