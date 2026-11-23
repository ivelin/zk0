import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, ARM5, SR

r = Ref("eye_drops_to_basket", 1630)
print("start qpos", np.round(np.degrees([r.data.qpos[r.qadr[j]] for j in ARM5]), 1))
for chunk in (15, 30, 45, 57):
    r.follow_demo(chunk)
    q = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
    c = np.array([r.data.ctrl[r.act[j]] for j in ARM5])
    print(f"frame {r.frame}: qpos {np.round(np.degrees(q),1)} ctrl {np.round(np.degrees(c),1)} maxerr {np.degrees(np.max(np.abs(q-c))):.1f} deg, forces {np.round(r.data.actuator_force[:5],2)}")
