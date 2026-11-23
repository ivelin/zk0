import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz

r = Ref("block_stack", 411)
B0 = r.body_pos("cube_b")
for dz in (0.030, 0.045, 0.060):
    tgt = B0 + [0, 0, dz]
    for yaw in (0, 30, 45, 60, 90, 120, 150, 180, -45, -90):
        sol = r.solve(tgt, Rz(math.radians(yaw)))
        if sol:
            print(f"dz={dz} yaw={yaw}: pos_mm={sol['pos_err']*1000:.1f} ori={sol['ori_err_deg']:.1f} wrist={sol['wrist_flex_deg']:.0f} q={sol['q_deg']}")
    print(f"dz={dz} done")
