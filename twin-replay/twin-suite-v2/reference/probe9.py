import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz, Ry, ARM5

r = Ref("block_stack", 411)
B0 = r.body_pos("cube_b")
cur = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
rng = np.random.RandomState(0)
seeds = [cur] + [cur + rng.uniform(-1, 1, 5) for _ in range(8)]
for name, tgt in [("place pinch", B0 + [0, 0, 0.030]), ("place+40mm x", B0 + [0.04, 0, 0.05]),
                  ("place+40z", B0 + [0, 0, 0.07]), ("mid", np.array([0.15, 0.0, 0.05]))]:
    best = None
    for s in seeds:
        q, pe, oe, within = r._ik(tgt, Rz(0), s, w_ori=0.0)
        if within and (best is None or pe < best[1]):
            best = (q, pe, oe)
    print(name, np.round(tgt, 3), "pos-only best pos_mm", None if best is None else round(best[1]*1000, 2),
          "ori_at_sol_deg", None if best is None else round(best[2], 1),
          "q_deg", None if best is None else np.round(np.degrees(best[0]), 1))
