import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, ARM5

for task, ep in (("eye_drops_to_basket", 1630), ("block_stack", 411)):
    r = Ref(task, ep)
    print(task, "qpos", np.round(np.degrees([r.data.qpos[r.qadr[j]] for j in ARM5]), 1))
    print("  pinch", np.round(r.pinch_centre(), 3), "forces", np.round(r.data.actuator_force, 2))
    for b in ("gripper", "moving_jaw_so101_v1"):
        print(" ", b, "contacts", r.contacts(b))
    # finger geom lowest points
    zs = []
    for gid in range(r.model.ngeom):
        bid = int(r.model.geom_bodyid[gid])
        bnm = mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bnm in ("gripper", "moving_jaw_so101_v1") and int(r.model.geom_group[gid]) == 3:
            zs.append((float(r.data.geom_xpos[gid][2]), mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_GEOM, gid)))
    zs.sort()
    print("  lowest finger geoms:", [(n, round(z, 4)) for z, n in zs[:4]])
