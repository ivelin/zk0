import numpy as np, json, glob, os
from PIL import Image, ImageDraw
W="/home/openclaw/muse-work/zk0-study001-twin-20261122"
FR=W+"/frames"; OUT=W+"/scenes"; OVL=OUT+"/overlays"
det=json.load(open(OUT+"/detections_px.json"))
det4=json.load(open(OUT+"/detections4_px.json"))
# ---------- calibration v2 ----------
CORNERS_PX=np.array([[42,498],[858,508],[928,148],[338,64]],float)  # NL,NR,FR,FL
L_MAT,W_MAT=0.55,0.40
LOCAL=np.array([[0,0],[0,L_MAT],[W_MAT,L_MAT],[W_MAT,0]],float)
def homography(src,dst):
    A=[]
    for (x,y),(X,Y) in zip(src,dst):
        A.append([-x,-y,-1,0,0,0,x*X,y*X,X]); A.append([0,0,0,-x,-y,-1,x*Y,y*Y,Y])
    _,_,Vt=np.linalg.svd(np.array(A)); H=Vt[-1].reshape(3,3); return H/H[2,2]
H=homography(CORNERS_PX,LOCAL); Hinv=np.linalg.inv(H)
X0,V0=0.05,0.155   # near edge at world x=+0.05; arm centreline at mat v=0.155
def px2world(px):
    p=H@np.array([px[0],px[1],1.0]); u,v=p[0]/p[2],p[1]/p[2]
    return np.array([u+X0, v-V0])
def yaw_world(px,yaw_px):
    if yaw_px is None: return None
    p1=np.array(px,float); p2=p1+10*np.array([np.cos(yaw_px),np.sin(yaw_px)])
    w1,w2=px2world(p1),px2world(p2); d=w2-w1
    return float(np.arctan2(d[1],d[0]))
def pose(px,z,yaw_px=None):
    w=px2world(px); return [round(float(w[0]),4),round(float(w[1]),4),z, (round(yaw_world(px,yaw_px),3) if yaw_px is not None else None)]
json.dump({"calibration_id":"calib_v2_mat_homography","corners_px":CORNERS_PX.tolist(),
  "mat_L_m":L_MAT,"mat_W_m":W_MAT,"x0_near_edge_world_x":X0,"v0_arm_centreline":V0,"H_pixel_to_matlocal":H.tolist()},
  open(OUT+"/calibration_v2.json","w"),indent=1)
# scale cross-checks (world extents of known objects)
def extent_world(b):
    if "extent" not in b or "px" not in b: return None
    return None
# ---------- manual / grounding picks (px) ----------
HAND_BLOCK_FIRST={"block_stack__ep411":{"cube_purple":[490,330]},
 "block_stack__ep530":{"cube_purple":[545,385]},
 "block_stack__ep539":{"cube_purple":[700,325],"cube_green":[530,490]},
 "block_stack__ep546":{"cube_purple":[545,385]}}
EYE_FIRST_GROUND={"eye_drops_to_basket__ep1609":[567,217],"eye_drops_to_basket__ep1618":[481,207],
 "eye_drops_to_basket__ep1630":[526,187],"eye_drops_to_basket__ep1639":[543,154],
 "eye_drops_to_basket__ep1643":[619,222],"eye_drops_to_basket__ep1757":[470,82],
 "eye_drops_to_basket__ep1760":[577,94],"eye_drops_to_basket__ep1778":[505,105],
 "eye_drops_to_basket__ep1808":[441,101],"eye_drops_to_basket__ep1810":[473,108],
 "eye_drops_to_basket__ep1821":[589,156]}
EYE_FINAL_GROUND={"eye_drops_to_basket__ep1609":[236,254],"eye_drops_to_basket__ep1630":[130,377],
 "eye_drops_to_basket__ep1639":[118,338],"eye_drops_to_basket__ep1643":[100,343],
 "eye_drops_to_basket__ep1724":[159,222],"eye_drops_to_basket__ep1741":[162,256],
 "eye_drops_to_basket__ep1746":[160,247],"eye_drops_to_basket__ep1747":[166,224],
 "eye_drops_to_basket__ep1757":[164,248],"eye_drops_to_basket__ep1778":[162,246],
 "eye_drops_to_basket__ep1780":[127,302],"eye_drops_to_basket__ep1783":[138,265],
 "eye_drops_to_basket__ep1798":[154,251],"eye_drops_to_basket__ep1808":[172,254],
 "eye_drops_to_basket__ep1810":[140,272],"eye_drops_to_basket__ep1811":[124,286],
 "eye_drops_to_basket__ep1814":[152,254],"eye_drops_to_basket__ep1821":[150,287]}
EYE_FINAL_LOW={"eye_drops_to_basket__ep1618","eye_drops_to_basket__ep1760"}
TOOL_GROUND={"tool_insert__ep622":[550,434],"tool_insert__ep629":[320,291],"tool_insert__ep683":[580,462],
 "tool_insert__ep684":[520,401],"tool_insert__ep824":[575,437],"tool_insert__ep830":[487,417],
 "tool_insert__ep852":[563,468],"tool_insert__ep859":[550,438],"tool_insert__ep883":[482,363],
 "tool_insert__ep902":[363,313]}
TOOL_LOW={"tool_insert__ep630","tool_insert__ep871","tool_insert__ep876"}
# fixtures (px, hand reads)
BOARD_PX=[645,265]; PEG_PX=[650,238]
RECESSES_PX=[[605,215],[700,230],[715,290],[665,330],[580,320]]
PLATFORM_PX=[545,230]; BASKET_PX=[195,275]; SLOT_PX=[450,222]
TRAY_CENTRE_PX=[605,235]
def fixture_pose(px,z=0.0):
    w=px2world(px); return [round(float(w[0]),4),round(float(w[1]),4),z,None]
FIX={"eye_drops_to_basket":{"platform":{"pose_xyz_yaw":fixture_pose(PLATFORM_PX,0.0),"top_z":0.09,"note":"hand read ep1609; top plate centre"},
     "basket":{"pose_xyz_yaw":fixture_pose(BASKET_PX,0.0),"note":"hand read ep1609; centre of basket footprint"}},
 "ring_insert":{"board":{"pose_xyz_yaw":fixture_pose(BOARD_PX,0.0),"note":"hand read ep7; disc centre"},
     "peg":{"pose_xyz_yaw":fixture_pose(PEG_PX,0.015),"note":"peg base at board centre; px from ep7"},
     "recesses":[{"id":i,"pose_xyz_yaw":fixture_pose(p,0.015)} for i,p in enumerate(RECESSES_PX)]},
 "tool_insert":{"tray":{"pose_xyz_yaw":fixture_pose(TRAY_CENTRE_PX,0.0),"note":"median tray bbox centre across episodes"},
     "target_slot":{"pose_xyz_yaw":fixture_pose(SLOT_PX,0.045),"note":"hand read ep622: red cross recess centre"}},
 "block_stack":{},"cable_clip":{}}
json.dump(FIX,open(OUT+"/fixtures.json","w"),indent=1)
# ---------- per-episode ----------
summary={}
for d in sorted(glob.glob(FR+"/*")):
    ep=os.path.basename(d); task=ep.rsplit("__ep",1)[0]; eid=int(ep.rsplit("ep",1)[1])
    F=det[ep]["first"]; L=det[ep]["final"]; objs={}; notes=[]
    if task=="block_stack":
        cubes={}
        for name in ("cube_purple","cube_green","cube_orange"):
            if ep in HAND_BLOCK_FIRST and name in HAND_BLOCK_FIRST[ep]:
                cubes[name]={"px":HAND_BLOCK_FIRST[ep][name],"conf":"medium","method":"manual pick from top_first (Builder B visual verification; auto detection failed/occluded)"}
            elif name in F:
                conf="high" if F[name].get("pass",1)==1 and F[name]["area"]>=500 else "medium"
                if ep=="block_stack__ep411" and name=="cube_green": conf="medium"
                if ep=="block_stack__ep513" and name=="cube_purple": conf="medium"
                cubes[name]={"px":F[name]["px"],"conf":conf,"method":"colour+median-diff segmentation (detect3)","yaw_px":F[name].get("yaw_px")}
        # final stack
        fin={k:v for k,v in L.items() if k.startswith("cube") and "px" in v}
        stack_px=None; fconf="low"
        if fin:
            P=np.array([v["px"] for v in fin.values()]); stack_px=P.mean(0)
            fconf="high" if len(fin)>=2 and np.abs(P[0]-P[1]).max()<55 else "medium"
        base=None; upper=None
        if stack_px is not None and cubes:
            dists={k:float(np.hypot(*(np.array(v["px"])-stack_px))) for k,v in cubes.items()}
            base=min(dists,key=dists.get)
            if dists[base]<70:
                upper=[k for k in cubes if k!=base][0] if len(cubes)>1 else None
            else: base=None
        for name,c in cubes.items():
            role="base" if name==base else ("manipulated" if name==upper else "cube")
            objs[name]={"pose_xyz_yaw":pose(c["px"],0.015,c.get("yaw_px")),"confidence":c["conf"],"method":c["method"],"role":role}
            if name==upper and stack_px is not None:
                objs[name+"_final"]={"pose_xyz_yaw":pose(stack_px,0.045,None),"confidence":fconf,
                  "method":"final stack location from top_final cube detection(s); this cube is the one that moved (initial position differs from stack)","role":"manipulated_final"}
            if name==base and stack_px is not None:
                objs[name+"_final"]={"pose_xyz_yaw":pose(stack_px,0.015,None),"confidence":fconf,
                  "method":"base cube under stack in top_final (upper cube occludes it; position = stack location)","role":"base_final"}
        if stack_px is None: notes.append("no cube detected in top_final; final stack pose not estimable")
    elif task=="eye_drops_to_basket":
        d4=det4.get(ep,{}).get("first",{}).get("carton")
        if d4: objs["carton"]={"pose_xyz_yaw":pose(d4["px"],0.1075,d4.get("yaw_px")),"confidence":"high","method":"interior-blob segmentation on platform plate (detect4)"}
        elif ep in EYE_FIRST_GROUND: objs["carton"]={"pose_xyz_yaw":pose(EYE_FIRST_GROUND[ep],0.1075,None),"confidence":"medium","method":"visual grounding pick on top_first, cross-checked by Builder B visual read"}
        else: objs["carton"]={"pose_xyz_yaw":None,"confidence":"low","method":"not detected"}
        if ep in EYE_FINAL_LOW:
            objs["carton_final"]={"pose_xyz_yaw":None,"confidence":"low","method":"carton occluded by arm/basket wall in top_final (arm ends over basket); not pinnable from top view"}
        elif ep in EYE_FINAL_GROUND:
            objs["carton_final"]={"pose_xyz_yaw":pose(EYE_FINAL_GROUND[ep],0.03,None),"confidence":"medium","method":"visual grounding pick on top_final (carton inside basket)"}
        else:
            objs["carton_final"]={"pose_xyz_yaw":None,"confidence":"low","method":"not located in top_final"}
    elif task=="ring_insert":
        if "ring" in F:
            r=F["ring"]; dist=float(np.hypot(r["px"][0]-BOARD_PX[0],r["px"][1]-BOARD_PX[1]))
            conf="high" if 400<=r["area"]<=3200 and dist<170 else "medium"
            rec=min(range(5),key=lambda i:(r["px"][0]-RECESSES_PX[i][0])**2+(r["px"][1]-RECESSES_PX[i][1])**2)
            rd=float(np.hypot(r["px"][0]-RECESSES_PX[rec][0],r["px"][1]-RECESSES_PX[rec][1]))
            objs["ring"]={"pose_xyz_yaw":pose(r["px"],0.021,None),"confidence":conf,
              "method":"colour+median-diff segmentation (detect3)","color":r.get("color"),
              "start_recess":(rec if rd<35 else "none_matched_board_surface"),"start_recess_dist_px":round(rd,1)}
        else: objs["ring"]={"pose_xyz_yaw":None,"confidence":"low","method":"not detected in top_first"}
        if "ring" in L:
            r=L["ring"]; dist=float(np.hypot(r["px"][0]-PEG_PX[0],r["px"][1]-PEG_PX[1]))
            conf="high" if dist<48 else "medium"
            objs["ring_final"]={"pose_xyz_yaw":pose(r["px"],0.021,None),"confidence":conf,
              "method":"colour+median-diff segmentation on top_final","dist_to_peg_px":round(dist,1)}
        else:
            objs["ring_final"]={"pose_xyz_yaw":pose(PEG_PX,0.021,None),"confidence":"medium",
              "method":"no separate ring blob in top_final; ring inferred on peg (recorded demo success, ring merges with board centre colours)"}
    elif task=="tool_insert":
        if ep in TOOL_LOW:
            objs["tool"]={"pose_xyz_yaw":None,"confidence":"low","method":"loose tool occluded by arm/wrist plate in top_first; detection and grounding disagree or absent — not pinnable"}
        elif ep in TOOL_GROUND:
            objs["tool"]={"pose_xyz_yaw":pose(TOOL_GROUND[ep],0.005,None),"confidence":"medium","method":"visual grounding pick on top_first (auto segmentation unreliable for chrome-on-mat)"}
        elif "tool" in det4.get(ep,{}).get("first",{}):
            t=det4[ep]["first"]["tool"]
            objs["tool"]={"pose_xyz_yaw":pose(t["px"],0.005,t.get("yaw_px")),"confidence":"medium","method":"metal+median-diff segmentation outside tray (detect4)"}
        else: objs["tool"]={"pose_xyz_yaw":None,"confidence":"low","method":"not detected"}
        objs["tool_final"]={"pose_xyz_yaw":pose(SLOT_PX,0.045,None),"confidence":"medium",
          "method":"target slot position (fixture): recorded demos all succeed; loose tool ends in the tray cross slot"}
    elif task=="cable_clip":
        if "block" in F:
            b=F["block"]; conf="high" if b["area"]>9000 else "medium"
            objs["clip_block"]={"pose_xyz_yaw":pose(b["px"],0.0275,b.get("yaw_px")),"confidence":conf,"method":"white-blob segmentation in mat ROI (detect3)"}
        if "cable" in F:
            c=F["cable"]; w=px2world(c["px"]); bb=[px2world([c["bbox"][0],c["bbox"][1]]),px2world([c["bbox"][2],c["bbox"][3]])]
            objs["cable"]={"pose_xyz_yaw":None,"confidence":"low",
              "method":"coarse descriptor only — full cable state not pinnable from a single top view (pre-registration: cable_clip scored on outcome class only)",
              "coil_centre_xy":[round(float(w[0]),4),round(float(w[1]),4)],
              "extent_bbox_xy":[[round(float(bb[0][0]),4),round(float(bb[0][1]),4)],[round(float(bb[1][0]),4),round(float(bb[1][1]),4)]],
              "dark_pixel_area":c["area"]}
    rec={"episode":eid,"episode_key":ep,"task":task,"calibration_id":"calib_v2_mat_homography",
         "objects":objs,"fixtures":FIX.get(task,{}),"notes":notes}
    json.dump(rec,open(OUT+"/"+ep+".json","w"),indent=1)
    # overlay with final initial markers
    ov=Image.open(d+"/top_first.png").convert("RGB"); dr=ImageDraw.Draw(ov)
    inv={}
    for name,o in objs.items():
        if o.get("pose_xyz_yaw"):
            w=np.array(o["pose_xyz_yaw"][:2]); p=Hinv@np.array([w[0]-X0,w[1]+V0,1.0]); px=p[:2]/p[2]
            col="lime" if o["confidence"]=="high" else ("yellow" if o["confidence"]=="medium" else "red")
            dr.ellipse([px[0]-9,px[1]-9,px[0]+9,px[1]+9],outline=col,width=3); dr.text((px[0]+11,px[1]-6),name,fill=col)
    ov.save(OVL+"/"+ep+".png")
    for name,o in objs.items():
        summary.setdefault(task,{}).setdefault(o["confidence"],[]).append(ep+":"+name)
for task,s in summary.items():
    print(task,{k:len(v) for k,v in s.items()})
json.dump(summary,open(OUT+"/confidence_summary.json","w"),indent=1)
print("done")
