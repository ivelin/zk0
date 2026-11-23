import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, ARM5

r = Ref("block_stack", 411)
m, d = r.model, r.data
cols = [r.vadr[j] for j in ARM5]
Jp = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv))
for gi, w in ((r.pad_fixed, 0.5), (r.pad_moving, 0.5)):
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jacGeom(m, d, jp, jr, gi)
    Jp += w * jp; Jr += w * jr
Ja = np.vstack([Jp[:, cols], Jr[:, cols]])
# numeric
eps = 1e-6
Jn = np.zeros((6, 5))
q0 = d.qpos.copy()
R0 = d.xmat[r.gbody].reshape(3, 3).copy(); pc0 = r.pinch_centre()
for i, j in enumerate(ARM5):
    d.qpos[r.qadr[j]] += eps
    mujoco.mj_forward(m, d)
    pc1 = r.pinch_centre(); R1 = d.xmat[r.gbody].reshape(3, 3)
    dR = R1 @ R0.T
    ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    if ang < 1e-9: ax = np.zeros(3)
    else: ax = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) / (2*np.sin(ang)) * ang
    Jn[:, i] = np.concatenate([(pc1 - pc0) / eps, ax / eps])
    d.qpos[:] = q0; mujoco.mj_forward(m, d)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print("analytic\n", Ja)
print("numeric\n", Jn)
print("max diff", np.abs(Ja - Jn).max())
