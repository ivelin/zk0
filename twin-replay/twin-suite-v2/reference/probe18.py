import sys
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, OPEN_RAD, ARM5, SR

r = Ref("eye_drops_to_basket", 1630)
tgt = r.body_pos("carton") + [0, 0, 0.08]
sols = r.continuation(tgt)
q = sols[-1]
print("final sol q_deg", np.round(np.degrees(q), 1))
# drive ONLY to the final solution, slowly
r.run_to(list(q) + [OPEN_RAD], 4.0)
for s in range(3):
    r.settle(0.5)
    print(f"t+{s} qpos_deg", np.round(np.degrees([r.data.qpos[r.qadr[j]] for j in ARM5]), 1),
          "pinch", np.round(r.pinch_centre(), 3))
    print("   actuator forces", np.round(r.data.actuator_force, 2))
    print("   contacts involving arm bodies:")
    for i in range(r.data.ncon):
        c = r.data.contact[i]
        b1 = mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_BODY, int(r.model.geom_bodyid[c.geom1]))
        b2 = mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_BODY, int(r.model.geom_bodyid[c.geom2]))
        if b1 in ("lower_arm", "wrist", "upper_arm", "shoulder") or b2 in ("lower_arm", "wrist", "upper_arm", "shoulder"):
            print("    ", b1, b2)
