import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz, Ry, Rx, OPEN_RAD

r = Ref("tool_insert", 683)
T0 = r.body_pos("tool")
print("tool at", np.round(T0, 4))
tgt = T0 + [0, 0, 0.020]
print("above:", r.move_pos(T0 + [0, 0, 0.08], OPEN_RAD))
cands = [("yaw90", Rz(math.pi / 2)), ("yaw0", Rz(0)), ("yaw-90", Rz(-math.pi / 2)),
         ("tiltY25", Rz(0) @ Ry(math.radians(25))), ("free", None)]
got = r.solve_position_only(tgt)
print("solve_position_only:", None if got is None else (round(got[1] * 1000, 1), np.round(np.degrees(got[0]), 0)))
if got is not None:
    cands[4] = ("free", got[2])
for name, Rt in cands:
    if Rt is None:
        continue
    sol = r.solve_link(tgt, Rt)
    sol2 = r.solve(tgt, Rt)
    print(name, "solve_link:", None if sol is None else (round(sol["pos_err"] * 1000, 1), round(sol["ori_err_deg"], 1)),
          "| solve:", None if sol2 is None else (round(sol2["pos_err"] * 1000, 1), round(sol2["ori_err_deg"], 1)))
sols = r.continuation(tgt)
print("continuation:", None if sols is None else len(sols))
