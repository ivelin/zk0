import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, ARM5

r = Ref("eye_drops_to_basket", 1630)
for home in ([0, -35, 55, 40, 0], [-10, -55, 75, 55, 0], [0, -70, 90, 60, 0], [-20, -45, 65, 50, 0], [0, -50, 40, 70, 0]):
    for i, j in enumerate(ARM5):
        r.data.qpos[r.qadr[j]] = np.radians(home[i])
    mujoco.mj_forward(r.model, r.data)
    print(home, "pinch", np.round(r.pinch_centre(), 3), "bias torques", np.round([r.data.qfrc_bias[r.vadr[j]] for j in ARM5], 2))
