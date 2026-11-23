import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, ARM5

r = Ref("block_stack", 411)
cb = r.body_pos("cube_b")
print("gt frames:", len(r.gt), "grip range:", min(f[5] for f in r.gt), max(f[5] for f in r.gt))
demo = r.demo_approach(cb)
print("demo_approach:", None if demo is None else (np.round(np.degrees(demo[0]), 0), "frame", demo[2]))
if demo:
    q, Rt, fr = demo
    # where is the pinch at that frame?
    import mujoco
    for i, j in enumerate(ARM5):
        r.data.qpos[r.qadr[j]] = q[i]
    mujoco.mj_forward(r.model, r.data)
    print("pinch at demo frame:", np.round(r.pinch_centre(), 4), "target:", np.round(cb, 4))
    ik = r._ik(cb, Rt, q, iters=500)
    print("polish IK: pe_mm=%.1f oe=%.1f within=%s torque=%s" % (ik[1] * 1000, ik[2], ik[3], r.torque_ok(ik[0])))
