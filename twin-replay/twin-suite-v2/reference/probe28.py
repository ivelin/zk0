import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz, Ry, ARM5

r = Ref("eye_drops_to_basket", 1630)
C0 = r.body_pos("carton")
tgt = C0 + [0, 0, 0.006]
r.move_pos(C0 + [0, 0, 0.08], 1.2)
cur = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
rng = np.random.RandomState(1)
seeds = [cur] + [np.radians(r.gt[f][:5]) for f in (0, 30, 57, 80, 110)]
for pan in (0.9, 1.5, 2.1):
    for lift in (-1.2, -0.5, 0.2):
        for elb in (-0.3, 0.5, 1.2):
            for wr in (-1.0, 0.0, 1.0):
                seeds.append(np.array([pan, lift, elb, wr, 0.0]))
seeds += [cur + rng.uniform(-0.7, 0.7, 5) for _ in range(12)]
for name, Rt in [("yaw90", Rz(math.pi / 2)), ("yaw0", Rz(0)), ("yaw-90", Rz(-math.pi / 2)), ("yaw45", Rz(math.pi / 4))]:
    best = None
    for s in seeds:
        q, pe, oe, w = r._ik(tgt, Rt, s, iters=400)
        if w and (best is None or pe + 0.005 * oe < best[0]):
            best = (pe + 0.005 * oe, pe, oe, q)
    print(name, None if best is None else (f"pe_mm={best[1]*1000:.1f} oe_deg={best[2]:.1f} torque_ok={r.torque_ok(best[3])} q={np.round(np.degrees(best[3]),0)}"))
