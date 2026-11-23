import sys, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore import Ref, Rz

r = Ref("tool_insert", 683)
T0 = r.body_pos("tool")
print("tool at", np.round(T0, 4))
g = r.grasp(T0, dz=0.020, Rt=Rz(math.pi / 2), obj_body="tool")
print("grasp:", None if g is None else {k: (np.round(v, 3) if isinstance(v, np.ndarray) else round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k != "Rt"})
print("tool now", np.round(r.body_pos("tool"), 4), "contacts:", r.contacts("tool"))
print("pinch:", np.round(r.pinch_centre(), 4))
for e in r.log:
    print(e)
