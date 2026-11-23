import sys, json
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref

r = Ref("block_stack", 411)
A0 = r.body_pos("cube_a"); B0 = r.body_pos("cube_b")
print("cube_a", np.round(A0,4), "cube_b", np.round(B0,4))
print("pinch at start", np.round(r.pinch_centre(),4), "site", np.round(r.site_pos(),4), "offset", np.round(r.pinch_offset_from_site(),4))
g = r.grasp(A0, yaw=0.0, dz=0.0)
print("grasp", g and {k:g[k] for k in ("stall","hold")})
print("after close cube_a", np.round(r.body_pos("cube_a"),4), "contacts", r.contacts("cube_a"))
print("pinch offset at grasp", np.round(r.pinch_offset_from_site(),4))
sol = r.lift(g["hold"], 0.07, Rt=g["Rt"])
print("lift sol", bool(sol), "cube_a", np.round(r.body_pos("cube_a"),4), "contacts", r.contacts("cube_a"))
print(json.dumps(r.log, indent=1, default=str))
