import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz, ARM5

r = Ref("block_stack", 411)
A0 = r.body_pos("cube_a")
for name, tgt, Rt in [("above", A0+[0,0,0.12], Rz(0)), ("grasp", A0.copy(), Rz(0))]:
    print(name, "target", np.round(tgt,4))
    cur = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
    seeds = [cur] + [np.radians(r.gt[f][:5]) for f in (130, 190)]
    for i, s in enumerate(seeds):
        q, pe, oe, within = r._ik(tgt, Rt, s)
        print(" seed", i, "pos_err_mm", round(pe*1000,2), "ori_err_deg", round(oe,2), "within", within, "q_deg", np.round(np.degrees(q),1))
