import numpy as np, json, glob, os
from PIL import Image, ImageDraw
W="/home/openclaw/muse-work/zk0-study001-twin-20261122"
FR=W+"/frames"; OUT=W+"/scenes"; OVL=OUT+"/overlays"
det=json.load(open(OUT+"/detections_px.json"))
cal=json.load(open(OUT+"/calibration_v2.json")); H=np.array(cal["H_pixel_to_matlocal"]); Hinv=np.linalg.inv(H)
X0,V0=0.05,0.155
def px2world(px):
    p=H@np.array([px[0],px[1],1.0]); u,v=p[0]/p[2],p[1]/p[2]; return np.array([u+X0,v-V0])
def pose_from_px(px,z):
    w=px2world(px); return [round(float(w[0]),4),round(float(w[1]),4),z,None]
# scale cross-checks
def world_dist(px1,px2): return float(np.hypot(*(px2world(px1)-px2world(px2))))
cube_ext=[]; ring_ext=[]
for ep,v in det.items():
    for k,b in v["first"].items():
        if "extent" not in b: continue
        d=world_dist(b["px"],[b["px"][0]+max(b["extent"]),b["px"][1]])
        if ep.startswith("block"): cube_ext.append(d)
        if ep.startswith("ring") and k=="ring": ring_ext.append(d)
print("cube major extent world median m:", round(float(np.median(cube_ext)),4), "n", len(cube_ext))
print("ring major extent world median m:", round(float(np.median(ring_ext)),4), "n", len(ring_ext))
json.dump({"cube_extent_world_median_m":float(np.median(cube_ext)),"ring_extent_world_median_m":float(np.median(ring_ext))},
          open(OUT+"/scale_crosschecks.json","w"),indent=1)
CABLE_BLOCK_GROUND={"cable_clip__ep1896":[372,298],"cable_clip__ep1903":[530,247],"cable_clip__ep1905":[637,338],
 "cable_clip__ep1939":[679,309],"cable_clip__ep1959":[538,151],"cable_clip__ep1960":[536,139],
 "cable_clip__ep1968":[391,185],"cable_clip__ep1969":[376,198],"cable_clip__ep2020":[524,147],
 "cable_clip__ep2032":[377,184],"cable_clip__ep2042":[636,279],"cable_clip__ep2046":[643,274],
 "cable_clip__ep2059":[371,214],"cable_clip__ep2068":[257,135],"cable_clip__ep2085":[522,150],
 "cable_clip__ep2092":[637,241],"cable_clip__ep2101":[620,266],"cable_clip__ep2125":[240,131],
 "cable_clip__ep2130":[233,134],"cable_clip__ep2146":[510,170]}
RECESS_OFFSETS=[[-62,-42],[-7,-67],[78,-38],[93,23],[28,73],[-47,47]]  # rel to peg, from ep33 geometry (6 recesses)
SIG={"high":(2.5,12),"medium":(3.5,20)}
def add_sigma(o,task,name):
    if not o.get("pose_xyz_yaw"): 
        o["sigma_xy_cm"]=None; o["sigma_yaw_deg"]=None; return
    conf=o["confidence"]
    if conf=="low": o["sigma_xy_cm"]=None; o["sigma_yaw_deg"]=None; return
    sxy,syaw=SIG[conf]
    if name=="ring_final": sxy=2.0; syaw=None
    if name=="tool_final": sxy=2.5; syaw=15
    if name=="clip_block": sxy=4.0
    if name.startswith("cube"): syaw=25
    if o["pose_xyz_yaw"][3] is None and name not in ("tool_final",): syaw=None
    o["sigma_xy_cm"]=sxy; o["sigma_yaw_deg"]=syaw
peg_positions={}
for f in sorted(glob.glob(OUT+"/*.json")):
    ep=os.path.basename(f)[:-5]
    if "__ep" not in ep: continue
    if not os.path.exists(FR+"/"+ep): continue
    rec=json.load(open(f)); task=rec["task"]; objs=rec["objects"]
    if task=="ring_insert":
        fin=det[ep]["final"].get("ring")
        if fin:
            peg=fin["px"]; peg_positions[ep]=peg
            board=[peg[0]+28,peg[1]-7]
            rec["fixtures"]={"board":{"pose_xyz_yaw":pose_from_px(board,0.0),"note":"PER-EPISODE board (board is re-placed between episodes; centre = final-ring peg position + median offset (+28,-7)px measured on ep7/ep33)"},
                             "peg":{"pose_xyz_yaw":pose_from_px(peg,0.015),"note":"peg position = ring final position (ring seated on peg in recorded final frame)"}}
            rf=objs.get("ring_final")
            if rf:
                rf["pose_xyz_yaw"]=pose_from_px(peg,0.021)
                rf["confidence"]="high" if fin["area"]>=1200 else "medium"
                rf["method"]="colour+median-diff segmentation on top_final; ring seated on peg (recorded demo success); peg anchor for this episode"
            ri=objs.get("ring")
            if ri and ri.get("pose_xyz_yaw"):
                d=[(ri["pose_xyz_yaw"],None)]
                best=None;bd=1e9
                for i,off in enumerate(RECESS_OFFSETS):
                    rp=[peg[0]+off[0],peg[1]+off[1]]
                    dd=float(np.hypot(fin["px"][0]-0,0)) # placeholder
                    # distance ring-start px to recess px: need ring start px from det
                    sp=det[ep]["first"]["ring"]["px"]; dd=float(np.hypot(sp[0]-rp[0],sp[1]-rp[1]))
                    if dd<bd: bd=dd; best=i
                ri["start_recess"]=(best if bd<42 else "board_surface_or_unmatched")
                ri["start_recess_dist_px"]=round(bd,1)
                ri["start_recess_note"]="6 recesses (corrects scene-inventory count of 5; counted in ring_insert__ep33 frames); board rotation per episode not estimated, recess identity approximate"
    if task=="cable_clip" and ep in CABLE_BLOCK_GROUND:
        objs["clip_block"]={"pose_xyz_yaw":pose_from_px(CABLE_BLOCK_GROUND[ep],0.0275),"confidence":"medium",
          "method":"visual grounding pick on top_first (white-blob segmentation rejected: block highlights merge with mat border stripe and split by lighting; grounding cross-checked vs Builder B full-res read on ep1939: agreement <5px)"}
        if "cable" in objs: objs["cable"]["sigma_xy_cm"]=8.0
    for name,o in objs.items(): add_sigma(o,task,name)
    json.dump(rec,open(f,"w"),indent=1)
    # regenerate overlay
    d=FR+"/"+ep
    ov=Image.open(d+"/top_first.png").convert("RGB"); dr=ImageDraw.Draw(ov)
    for name,o in objs.items():
        if o.get("pose_xyz_yaw"):
            w=np.array(o["pose_xyz_yaw"][:2]); p=Hinv@np.array([w[0]-X0,w[1]+V0,1.0]); px=p[:2]/p[2]
            col="lime" if o["confidence"]=="high" else ("yellow" if o["confidence"]=="medium" else "red")
            dr.ellipse([px[0]-9,px[1]-9,px[0]+9,px[1]+9],outline=col,width=3); dr.text((px[0]+11,px[1]-6),name,fill=col)
    ov.save(OVL+"/"+ep+".png")
# fixtures.json: ring becomes per-episode summary
fx=json.load(open(OUT+"/fixtures.json"))
pegs=np.array(list(peg_positions.values()))
fx["ring_insert"]={"board_varies_per_episode":True,
  "peg_positions_px":{k:[round(float(x),1),round(float(y),1)] for k,(x,y) in peg_positions.items()},
  "peg_px_spread_std":[round(float(pegs[:,0].std()),1),round(float(pegs[:,1].std()),1)],
  "note":"Board/peg are re-placed between episodes (spread std ~65x45 px). Per-episode board+peg poses are in each episode JSON under fixtures. Recess count = 6 (see pose-estimation.md correction)."}
json.dump(fx,open(OUT+"/fixtures.json","w"),indent=1)
print("pegs n=",len(peg_positions))
print("patched")
