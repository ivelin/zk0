import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, math
from refcore import Ref, Rz, Ry, ARM5

r = Ref("block_stack", 411)
A0 = r.body_pos("cube_a")
cur = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
for dz in (0.05, 0.08, 0.10, 0.12, 0.15):
    tgt = A0 + [0, 0, dz]
    for tname, Rt in [("I", Rz(0)), ("pitch-20", Rz(0) @ Ry(math.radians(-20))), ("pitch+20", Rz(0) @ Ry(math.radians(20))), ("pitch-40", Rz(0) @ Ry(math.radians(-40)))]:
        q, pe, oe, within = r._ik(tgt, Rt, cur)
        print(f"dz={dz} {tname}: pos_mm={pe*1000:.1f} ori_deg={oe:.1f} wrist={np.degrees(q[3]):.0f} within={within}")
