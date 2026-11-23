import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz, Ry

r = Ref("block_stack", 411)
B0 = r.body_pos("cube_b")
tgt = B0 + [0, 0, 0.030]
print("place pinch target", np.round(tgt, 4))
for name, Rt in [("I", Rz(0)), ("pitch-15", Rz(0) @ Ry(math.radians(-15))), ("pitch-30", Rz(0) @ Ry(math.radians(-30))),
                 ("pitch-45", Rz(0) @ Ry(math.radians(-45))), ("pitch+15", Rz(0) @ Ry(math.radians(15))),
                 ("pitch+30", Rz(0) @ Ry(math.radians(30))), ("yaw90 pitch-30", Rz(math.pi/2) @ Ry(math.radians(-30)))]:
    sol = r.solve(tgt, Rt)
    print(name, None if sol is None else {k: sol[k] for k in ("pos_err", "ori_err_deg", "wrist_flex_deg", "q_deg")})
