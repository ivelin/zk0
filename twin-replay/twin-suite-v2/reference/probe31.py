import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, ARM5

r = Ref("block_stack", 411)
cb = r.body_pos("cube_b")
rng = np.random.RandomState(3)
lo = np.array([r.lim[j][0] for j in ARM5]); hi = np.array([r.lim[j][1] for j in ARM5])
for dz in (0.03, 0.05, 0.08):
    tgt = cb + [0, 0, dz]
    best = (9e9, None)
    seeds = [rng.uniform(lo, hi) for _ in range(120)]
    seeds += [np.radians(r.gt[f][:5]) for f in range(0, len(r.gt), 7)]
    for s in seeds:
        q, pe, oe, w = r._ik(tgt, np.eye(3), s, iters=300, w_ori=0.0)
        if w and pe < best[0]:
            best = (pe, q)
    print(f"dz={dz}: best pe_mm={best[0]*1000:.1f} q_deg={None if best[1] is None else np.round(np.degrees(best[1]),0)} torque_ok={r.torque_ok(best[1]) if best[1] is not None else '-'}")
