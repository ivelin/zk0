import numpy as np, json
W="/home/openclaw/muse-work/zk0-study001-twin-20261122"; OUT=W+"/scenes"
cal=json.load(open(OUT+"/calibration_v2.json")); H=np.array(cal["H_pixel_to_matlocal"])
def p2w(px):
    p=H@np.array([px[0],px[1],1.0]); return [round(float(p[0]/p[2]+0.05),4), round(float(p[1]/p[2]-0.155),4)]
G={"eye_drops_to_basket__ep1780":[528,123],"eye_drops_to_basket__ep1783":[603,141],"eye_drops_to_basket__ep1814":[558,121]}
for ep,px in G.items():
    f=OUT+"/"+ep+".json"; r=json.load(open(f)); w=p2w(px)
    r["objects"]["carton"]={"pose_xyz_yaw":[w[0],w[1],0.1075,None],"confidence":"medium",
      "method":"visual grounding pick on top_first, cross-checked by Builder B visual read","sigma_xy_cm":3.5,"sigma_yaw_deg":None}
    json.dump(r,open(f,"w"),indent=1)
import glob,os
for f in glob.glob(OUT+"/*__ep*.json"):
    if "_variants" in f: continue
    r=json.load(open(f))
    if r["task"]=="cable_clip" and "cable" in r["objects"]:
        r["objects"]["cable"]["sigma_xy_cm"]=8.0  # on coil_centre_xy descriptor
    if r["task"]=="ring_insert":
        rf=r["objects"].get("ring_final",{})
        rf.pop("dist_to_peg_px",None)
    json.dump(r,open(f,"w"),indent=1)
print("cleanup done")
