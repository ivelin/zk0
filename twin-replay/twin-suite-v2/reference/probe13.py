import sys, time
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref

r = Ref("block_stack", 411)
for tgt in ([0.0657, -0.0202, 0.0584], [0.066, -0.023, 0.045], [0.066, -0.023, 0.075], [0.10, -0.023, 0.058]):
    t0 = time.time()
    got = r.solve_position_only(np.array(tgt))
    print(tgt, "->", None if got is None else (round(got[1] * 1000, 2), np.round(np.degrees(got[0]), 1)), f"{time.time()-t0:.0f}s")
    if got:
        print("   R=\n", np.round(got[2], 3))
