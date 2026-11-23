import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz

r = Ref("block_stack", 411)
B0 = r.body_pos("cube_b")
for z in (0.015, 0.025, 0.035, 0.045):
    tgt = np.array([B0[0], B0[1], z])
    got = r.solve_position_only(tgt)
    print(f"z={z}: pos-only", None if got is None else round(got[1] * 1000, 2), "mm")
    if got:
        sols = r.solve_path([tgt], got[2])
        print("   path to free-Rt:", bool(sols))
    sol = r.solve(tgt, Rz(0))
    print("   full solve Rt=I:", None if sol is None else round(sol["pos_err"] * 1000, 2))
