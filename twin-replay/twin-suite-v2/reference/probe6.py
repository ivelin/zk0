import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, ARM5

r = Ref("block_stack", 411)
m, d = r.model, r.data
pc_step = r.pinch_centre().copy(); R_step = d.xmat[r.gbody].reshape(3,3).copy()
mujoco.mj_forward(m, d)
pc_fwd = r.pinch_centre().copy(); R_fwd = d.xmat[r.gbody].reshape(3,3).copy()
print("pc step", pc_step); print("pc fwd ", pc_fwd); print("diff", pc_fwd - pc_step)
print("R diff max", np.abs(R_fwd - R_step).max())
print("time", d.time)
