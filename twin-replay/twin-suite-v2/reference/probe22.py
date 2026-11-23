import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, ARM5

for mode in ("fingers-on", "fingers-off"):
    r = Ref("eye_drops_to_basket", 1630)
    if mode == "fingers-off":
        for gid in range(r.model.ngeom):
            bid = int(r.model.geom_bodyid[gid])
            bnm = mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bnm in ("gripper", "moving_jaw_so101_v1"):
                r.model.geom_contype[gid] = 0; r.model.geom_conaffinity[gid] = 0
    r.follow_demo(57)
    q = np.array([r.data.qpos[r.qadr[j]] for j in ARM5])
    c = np.array([r.data.ctrl[r.act[j]] for j in ARM5])
    print(mode, "maxerr deg", round(float(np.degrees(np.max(np.abs(q - c)))), 1), "pinch", np.round(r.pinch_centre(), 3))
    # what are the fingers touching at the end?
    print("   carton contacts", r.contacts("carton"))
