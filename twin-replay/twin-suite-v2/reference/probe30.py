import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
import refcore  # noqa
from refcore import Ref


r = Ref("block_stack", 411)
print("pinch at HOME:", np.round(r.pinch_centre(), 4))
cb = r.body_pos("cube_b")
print("cube_b", np.round(cb, 4))
tgt = cb + [0, 0, 0.08]
print("above move_pos:", r.move_pos(tgt, 1.2))
print("pinch now:", np.round(r.pinch_centre(), 4))
sols = r.continuation(cb + [0, 0, 0.05])
print("continuation to +0.05:", None if sols is None else len(sols))
got = r.solve_position_only(cb + [0, 0, 0.08])
print("solve_position_only +0.08:", None if got is None else (round(got[1] * 1000, 1), np.round(np.degrees(got[0]), 0)))
