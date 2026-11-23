import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref

r = Ref("block_stack", 411)
cb = r.body_pos("cube_b")
for dz in (0.05, 0.03):
    got = r.solve_position_only(cb + [0, 0, dz])
    print(f"solve_position_only +{dz}:", None if got is None else (round(got[1] * 1000, 1), np.round(np.degrees(got[0]), 0)))
print("move_pos +0.05:", r.move_pos(cb + [0, 0, 0.05], 1.2))
print("pinch now:", np.round(r.pinch_centre(), 4))
