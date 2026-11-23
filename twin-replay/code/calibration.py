import numpy as np, json
# Mat corners in top-view pixels, Builder B visual read of block_stack__ep411/top_first.png
# (cleanest frame), cross-checked on tool_insert__ep622. Order: NL(near-left), NR(near-right), FR(far-right), FL(far-left)
CORNERS_PX=np.array([[42,498],[858,508],[928,148],[338,64]],float)
L_MAT=0.55  # long axis (near/far edges), world y
W_MAT=0.40  # short axis (near->far), world x
# mat-local: u in [0,W] near->far, v in [0,L] NL->NR along near edge
LOCAL=np.array([[0,0],[0,L_MAT],[W_MAT,L_MAT],[W_MAT,0]],float)
def homography(src,dst):
    A=[]
    for (x,y),(X,Y) in zip(src,dst):
        A.append([-x,-y,-1,0,0,0,x*X,y*X,X]); A.append([0,0,0,-x,-y,-1,x*Y,y*Y,Y])
    _,_,Vt=np.linalg.svd(np.array(A)); H=Vt[-1].reshape(3,3); return H/H[2,2]
H=homography(CORNERS_PX,LOCAL)
def px2local(px):
    p=H@np.array([px[0],px[1],1.0]); return (p[:2]/p[2])  # (u,v)
# Anchors: FK home gripper world (0.115,0.005) median over episodes; plate centre px in ep411 ~(440,255)
fk=json.load(open("/home/openclaw/muse-work/zk0-study001-twin-20261122/scenes/fk_tiepoints.json"))
G=np.array([v["first"]["gripper_xyz"] for v in fk.values()]); gmed=np.median(G,0)
print("FK home gripper median xyz",gmed, "spread std",G.std(0))
plate_px=np.array([440,255]); base_px=np.array([285,565])
ug,vg=px2local(plate_px); ub,vb=px2local(base_px)
print("plate -> local u,v",ug,vg," base -> local u,v",ub,vb)
# Placement: world x = u + x0, world y = v - v0. Gripper: x0 = gx - ug ; v0 = vg - gy
x0=gmed[0]-ug; v0=vg-gmed[1]
print("placement x0,v0 from gripper anchor:",x0,v0)
print("base implied world:", ub+x0, vb-v0, "(should be ~(0,0); base px is rough, treat as check)")
def px2world(px):
    u,v=px2local(px); return np.array([u+x0, v-v0])
print("mat corners world:", [px2world(c).round(3).tolist() for c in CORNERS_PX])
json.dump({"corners_px":CORNERS_PX.tolist(),"L":L_MAT,"W":W_MAT,"x0":float(x0),"v0":float(v0),
           "H":H.tolist(),"fk_home_gripper_median":gmed.tolist(),"plate_px_ep411":plate_px.tolist()},
          open("/home/openclaw/muse-work/zk0-study001-twin-20261122/scenes/calibration_v1.json","w"),indent=1)
print("saved calibration_v1")
