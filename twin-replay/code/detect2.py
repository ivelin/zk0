import numpy as np, json, glob, os
from PIL import Image, ImageDraw
from scipy import ndimage
FR="/home/openclaw/muse-work/zk0-study001-twin-20261122/frames"
OUT="/home/openclaw/muse-work/zk0-study001-twin-20261122/scenes"
OVL=OUT+"/overlays"; os.makedirs(OVL,exist_ok=True)
CORNERS=np.array([[42,498],[858,508],[928,148],[338,64]],float)
# mat ROI polygon mask
yy,xx=np.mgrid[0:576,0:1024]
from matplotlib.path import Path
ROI=Path(CORNERS).contains_points(np.stack([xx.ravel(),yy.ravel()],1)).reshape(576,1024)
ROI=ndimage.binary_erosion(ROI,np.ones((9,9)))

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
    C=np.cov(X.T); w,V=np.linalg.eigh(C); ax=V[:,-1]
    P=X@V; ext=[float(P[:,i].max()-P[:,i].min()) for i in range(2)]
    return float(np.arctan2(ax[1],ax[0])), float(np.sqrt(w[-1]/max(w[0],1e-9))), ext
def pack(b):
    y,r,ext=pca(b); return {"px":[b["cx"],b["cy"]],"area":b["area"],"yaw_px":y,"elong":r,"extent":ext,"bbox":b["bbox"]}

tasks=["block_stack","eye_drops_to_basket","ring_insert","tool_insert","cable_clip"]
med={}
for task in tasks:
    for which in ("first","final"):
        ims=[load(d+f"/top_{which}.png") for d in sorted(glob.glob(FR+"/"+task+"__*"))]
        med[(task,which)]=np.median(np.stack(ims),0)

def detect(task,im,which):
    R,G,B,V,S=chans(im); diff=np.abs(im-med[(task,which)]).max(2)>28
    diffF=np.abs(im-med[(task,"first")]).max(2)>28  # vs first-median (for finals: stack/carton-in-basket stand out)
    res={}
    if task=="block_stack":
        purple=(B>R+18)&(B>G+28)&(S>0.30)&(V>45)&(V<200)
        green=(G>R+22)&(G>B+8)&(S>0.30)&(V>55)&(V<210)
        orange=(R>135)&(R>G+38)&(G>B+25)&(S>0.35)&(V<230)
        for name,mask in (("cube_purple",purple),("cube_green",green),("cube_orange",orange)):
            m=mask&ROI&(diff if which=="first" else (diff|diffF))
            bs=blobs(m,250,6000)
            if bs: res[name]=pack(bs[0])
    elif task=="eye_drops_to_basket":
        blue=(B>R+20)&(B>G+2)&(S>0.22)&(V>50)&(V<220)
        m=blue&ROI&(diff if which=="first" else (diff|diffF))
        bs=blobs(m,200,9000)
        if bs: res["carton"]=pack(bs[0])
    elif task=="ring_insert":
        yellow=(R>160)&(G>135)&(B<125)&(R>B+65)&(S>0.30)
        red=(R>130)&(R>G+55)&(R>B+45)&(S>0.30)&(V<200)
        for name,mask in (("yellow",yellow),("red",red)):
            m=mask&ROI&(diff if which=="first" else (diff|diffF))
            bs=blobs(m,120,3500)
            if bs: r=pack(bs[0]); r["color"]=name; res["ring"]=r; break
    elif task=="tool_insert":
        metal=(V>130)&(S<0.32)
        m=metal&ROI&(diff if which=="first" else diffF)
        cands=[]
        for b in blobs(m,150,6000):
            y,r,ext=pca(b)
            if r and r>1.9: cands.append((b,y,r))
        if cands: res["tool"]=pack(cands[0][0])
    elif task=="cable_clip":
        white=(V>185)&(S<0.20)
        m=white&ROI&(diff|diffF)
        bs=blobs(m,900,26000)
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
        x,y=b["px"]; c=cols.get(name,"red")
        dr.ellipse([x-9,y-9,x+9,y+9],outline=c,width=3); dr.text((x+11,y-6),name,fill=c)
        if "bbox" in b: dr.rectangle(b["bbox"],outline=c,width=2)
    ov.save(OVL+"/"+ep+".png")
json.dump(allres,open(OUT+"/detections_px.json","w"),indent=1)
from collections import Counter
for task in tasks:
    eps={k:v for k,v in allres.items() if k.startswith(task)}; keys=Counter()
    for v in eps.values():
        for w in ("first","final"): keys.update([w+":"+k for k in v[w]])
    print(task,dict(keys))
