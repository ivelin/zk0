import numpy as np, json, glob, os
from PIL import Image, ImageDraw
from scipy import ndimage
from matplotlib.path import Path
FR="/home/openclaw/muse-work/zk0-study001-twin-20261122/frames"
OUT="/home/openclaw/muse-work/zk0-study001-twin-20261122/scenes"
OVL=OUT+"/overlays"; os.makedirs(OVL,exist_ok=True)
CORNERS=np.array([[42,498],[858,508],[928,148],[338,64]],float)
yy,xx=np.mgrid[0:576,0:1024]
ROI=Path(CORNERS).contains_points(np.stack([xx.ravel(),yy.ravel()],1)).reshape(576,1024)
ROI=ndimage.binary_erosion(ROI,np.ones((11,11)))
def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(float)
def chans(im):
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]; V=(R+G+B)/3; mx=im.max(2); S=(mx-im.min(2))/(mx+1e-6)
    return R,G,B,V,S
def blobs(mask,min_area=60,max_area=100000):
    mask=ndimage.binary_opening(mask,np.ones((3,3)))
    lab,n=ndimage.label(mask); out=[]
    for i in range(1,n+1):
        ys,xs=np.nonzero(lab==i)
        if not (min_area<=len(xs)<=max_area): continue
        out.append({"cx":float(xs.mean()),"cy":float(ys.mean()),"area":int(len(xs)),
                    "bbox":[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],"pts":(xs,ys)})
    return sorted(out,key=lambda b:-b["area"])
def pca(b):
    xs,ys=b["pts"]; X=np.stack([xs-xs.mean(),ys-ys.mean()],1).astype(float)
    if len(xs)<10: return None,None,None
    C=np.cov(X.T); w,V=np.linalg.eigh(C); ax=V[:,-1]; P=X@V
    return float(np.arctan2(ax[1],ax[0])), float(np.sqrt(w[-1]/max(w[0],1e-9))), [float(P[:,i].max()-P[:,i].min()) for i in range(2)]
def pack(b):
    y,r,ext=pca(b); return {"px":[b["cx"],b["cy"]],"area":b["area"],"yaw_px":y,"elong":r,"extent":ext,"bbox":b["bbox"]}
tasks=["block_stack","eye_drops_to_basket","ring_insert","tool_insert","cable_clip"]
med={}
for task in tasks:
    for which in ("first","final"):
        ims=[load(d+f"/top_{which}.png") for d in sorted(glob.glob(FR+"/"+task+"__*"))]
        med[(task,which)]=np.median(np.stack(ims),0)
# median colour blobs (fixed print) for block second pass
def colour_masks(im):
    R,G,B,V,S=chans(im)
    return {"cube_purple":(B>R+18)&(B>G+28)&(S>0.30)&(V>45)&(V<200),
            "cube_green":(G>R+22)&(G>B+8)&(S>0.30)&(V>55)&(V<210),
            "cube_orange":(R>135)&(R>G+38)&(G>B+25)&(S>0.35)&(V<230)}
medblobs={}
mb=colour_masks(med[("block_stack","first")])
for k,m in mb.items(): medblobs[k]=[(b["cx"],b["cy"]) for b in blobs(m&ROI,150,20000)]
# tray bbox per tool episode (dark|red largest)
def tray_bbox(im):
    R,G,B,V,S=chans(im); m=((V<78)|((R>95)&(G<78)&(B<78)))&ROI
    bs=blobs(m,20000,300000)
    return bs[0]["bbox"] if bs else [240,50,870,370]
# target slot: red cross in tool median-first inside tray
tm=med[("tool_insert","first")]; R,G,B,V,S=chans(tm)
redm=(R>95)&(G<80)&(B<80)&ROI
slot=None
for b in blobs(redm,300,6000):
    if 380<b["cx"]<540 and 150<b["cy"]<290: slot={"px":[b["cx"],b["cy"]],"area":b["area"],"bbox":b["bbox"]}; break
print("slot:",slot)
json.dump({"slot_px":slot},open(OUT+"/slot_px.json","w"))

def detect(task,im,which):
    R,G,B,V,S=chans(im); diff=np.abs(im-med[(task,which)]).max(2)>28
    diffF=np.abs(im-med[(task,"first")]).max(2)>28
    res={}
    if task=="block_stack":
        masks=colour_masks(im)
        for name,mask in masks.items():
            m=mask&ROI&(diff if which=="first" else (diff|diffF))
            bs=blobs(m,250,6000)
            if bs: res[name]=pack(bs[0])
        # second pass for missing colours: no diff, exclude print blobs & found cubes
        for name,mask in masks.items():
            if name in res: continue
            for b in blobs(mask&ROI,250,6000):
                if any((b["cx"]-px)**2+(b["cy"]-py)**2<32**2 for px,py in medblobs[name]): continue
                if any((b["cx"]-v["px"][0])**2+(b["cy"]-v["px"][1])**2<22**2 for v in res.values()): continue
                r=pack(b); r["pass"]=2; res[name]=r; break
    elif task=="eye_drops_to_basket":
        blue=(B>R+20)&(B>G+2)&(S>0.22)&(V>50)&(V<220)
        region=np.zeros_like(ROI)
        if which=="first": region[120:320,370:710]=True
        else: region[110:430,15:375]=True
        m=(blue|(diff if which=="first" else (diff|diffF)))&ROI&region
        bs=blobs(m,250,12000)
        if bs: res["carton"]=pack(bs[0])
    elif task=="ring_insert":
        yellow=(R>160)&(G>135)&(B<125)&(R>B+65)&(S>0.30)
        red=(R>130)&(R>G+55)&(R>B+45)&(S>0.30)&(V<200)
        for name,mask in (("yellow",yellow),("red",red)):
            m=mask&ROI&(diff if which=="first" else (diff|diffF))
            bs=blobs(m,120,3500)
            if bs: r=pack(bs[0]); r["color"]=name; res["ring"]=r; break
    elif task=="tool_insert":
        tb=tray_bbox(im); res["tray_bbox"]=tb
        if which=="first":
            metal=(V>130)&(S<0.32)
            m=metal&ROI&diff
            cands=[]
            for b in blobs(m,150,6000):
                ox=max(0,min(b["bbox"][2],tb[2])-max(b["bbox"][0],tb[0])); oy=max(0,min(b["bbox"][3],tb[3])-max(b["bbox"][1],tb[1]))
                inter=ox*oy; area_b=(b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1])
                if area_b>0 and inter/area_b>0.25: continue
                y,r,ext=pca(b)
                if r and r>1.9: cands.append(b)
            if cands: res["tool"]=pack(cands[0])
    elif task=="cable_clip":
        white=(V>185)&(S<0.20)
        bs=blobs(white&ROI,2500,26000)
        if bs: res["block"]=pack(bs[0])
        darkm=(V<50)&ROI
        lab,n=ndimage.label(ndimage.binary_opening(darkm,np.ones((3,3))))
        arm=np.zeros_like(darkm); bot=set(np.unique(lab[555:,:]))-{0}
        for i in bot: arm|=(lab==i)
        cable=darkm&(~arm)&(diff|diffF)
        ys,xs=np.nonzero(cable)
        if len(xs)>150:
            res["cable"]={"px":[float(xs.mean()),float(ys.mean())],"area":int(len(xs)),
                          "bbox":[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]}
    return res

allres={}
for d in sorted(glob.glob(FR+"/*")):
    ep=os.path.basename(d); task=ep.rsplit("__ep",1)[0]
    entry={"first":detect(task,load(d+"/top_first.png"),"first"),
           "final":detect(task,load(d+"/top_final.png"),"final")}
    allres[ep]=entry
    ov=Image.open(d+"/top_first.png").convert("RGB"); dr=ImageDraw.Draw(ov)
    cols={"cube_purple":"magenta","cube_green":"lime","cube_orange":"orange","carton":"cyan","ring":"yellow","tool":"cyan","block":"cyan","cable":"yellow"}
    for name,b in entry["first"].items():
        if "px" not in b: continue
        x,y=b["px"]; c=cols.get(name,"red")
        dr.ellipse([x-9,y-9,x+9,y+9],outline=c,width=3); dr.text((x+11,y-6),name,fill=c)
        if "bbox" in b: dr.rectangle(b["bbox"],outline=c,width=2)
    ov.save(OVL+"/"+ep+".png")
json.dump(allres,open(OUT+"/detections_px.json","w"),indent=1)
from collections import Counter
for task in tasks:
    eps={k:v for k,v in allres.items() if k.startswith(task)}; keys=Counter()
    for v in eps.values():
        for w in ("first","final"): keys.update([w+":"+k for k in v[w] if k!="tray_bbox"])
    print(task,dict(keys))
for ep in sorted(allres):
    if ep.startswith("block"):
        f=allres[ep]["first"]
        print(ep,{k:(v["area"],[round(c) for c in v["px"]],v.get("pass",1)) for k,v in f.items()})
