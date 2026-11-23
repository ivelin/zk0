import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np, mujoco
from refcore import Ref, Rz, SR

r = Ref("eye_drops_to_basket", 1630)
C0 = r.body_pos("carton")
g = r.grasp(C0, yaw=math.pi / 2, dz=0.006, close_target=-0.16, obj_body="carton")
print("grasp returned:", None if g is None else round(g["stall"], 3))
C1 = r.body_pos("carton")
print("carton before/after:", np.round(C0, 4), np.round(C1, 4))
print("carton contacts (bodies):", r.contacts("carton"))
# geom-level contacts with the carton
cb = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "carton")
for i in range(r.data.ncon):
    c = r.data.contact[i]
    b1 = int(r.model.geom_bodyid[c.geom1]); b2 = int(r.model.geom_bodyid[c.geom2])
    if cb in (b1, b2):
        g1 = mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print("  contact geoms:", g1, "<->", g2, "pos", np.round(c.pos, 4))
print("pinch", np.round(r.pinch_centre(), 4))
print("pad_fixed", np.round(r.data.geom_xpos[r.pad_fixed], 4), "pad_moving", np.round(r.data.geom_xpos[r.pad_moving], 4))
print("gripper R\n", np.round(r.data.xmat[r.gbody].reshape(3, 3), 3))
