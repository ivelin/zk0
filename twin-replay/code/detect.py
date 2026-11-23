import numpy as np, json, glob, os, sys
from PIL import Image, ImageDraw
from scipy import ndimage
FR="/home/openclaw/muse-work/zk0-study001-twin-20261122/frames"
OUT="/home/openclaw/muse-work/zk0-study001-twin-20261122/scenes"
OVL=OUT+"/overlays"; os.makedirs(OVL,exist_ok=True)

def load(p):
    im=np.asarray(Image.open(p).convert("RGB")).astype(float)
    return im
def blobs(mask,min_area=60):
    mask=ndimage.binary_opening(mask,np.ones((3,3)))
    lab,n=ndimage.label(mask); out=[]
    for i in range(1,n+1):
        ys,xs=np.nonzero(lab==i)
        if len(xs)<min_area: continue
        out.append({"cx":float(xs.mean()),"cy":float(ys.mean()),"area":int(len(xs)),
                    "bbox":[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
                    "pts":(xs,ys)})
    return sorted(out,key=lambda b:-b["area"])
def pca_yaw(b):
    xs,ys=b["pts"]; X=np.stack([xs-xs.mean(),ys-ys.mean()],1)
    if len(xs)<10: return None,None
    C=np.cov(X.T); w,V=np.linalg.eigh(C); ax=V[:,-1]
    return float(np.arctan2(ax[1],ax[0])), float(np.sqrt(w[-1]/max(w[0],1e-9)))
def extent(b):
    xs,ys=b["pts"]; X=np.stack([xs,ys],1).astype(float)
    C=np.cov((X-X.mean(0)).T); w,V=np.linalg.eigh(C)
    P=(X-X.mean(0))@V; return [float(P[:,i].max()-P[:,i].min()) for i in range(2)]

def channels(im):
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]; V=(R+G+B)/3; mx=im.max(2); mn=im.min(2); S=(mx-mn)/(mx+1e-6)
    return R,G,B,V,S

def detect(task,im,which):
    R,G,B,V,S=channels(im); H,Wd=V.shape
    res={}
    if task=="block_stack":
        purple=(B>R+15)&(B>G+25)&(V>50)&(V<190)
        green=(G>R+18)&(G>B+10)&(V>60)&(V<200)
        orange=(R>140)&(G>70)&(G<175)&(B<95)&(R>G+35)
        for name,mask in (("cube_purple",purple),("cube_green",green),("cube_orange",orange)):
            bs=[b for b in blobs(mask,50) if b["cy"]<520]
            if bs: b=bs[0]; y,r=pca_yaw(b); res[name]={"px":[b["cx"],b["cy"]],"area":b["area"],"yaw_px":y,"elong":r,"extent":extent(b)}
    elif task=="eye_drops_to_basket":
        blue=(B>R+22)&(B>G+5)&(V>55)&(V<215)&(S>0.18)
        bs=blobs(blue,40)
        if bs: b=bs[0]; y,r=pca_yaw(b); res["carton"]={"px":[b["cx"],b["cy"]],"area":b["area"],"yaw_px":y,"elong":r,"extent":extent(b)}
        if which=="first":
            white=(V>185)&(S<0.16)
            bs=[b for b in blobs(white,800) if b["cx"]>380]
            if bs: res["platform"]={"px":[bs[0]["cx"],bs[0]["cy"]],"area":bs[0]["area"],"bbox":bs[0]["bbox"],"extent":extent(bs[0])}
    elif task=="ring_insert":
        cream=(R>175)&(G>150)&(B>105)&(R>B+25)&(V>150)
        bs=blobs(cream,3000)
        if bs: res["board"]={"px":[bs[0]["cx"],bs[0]["cy"]],"area":bs[0]["area"],"bbox":bs[0]["bbox"],"extent":extent(bs[0])}
        yellow=(R>165)&(G>145)&(B<115)&(R>B+70)
        red=(R>135)&(G<95)&(B<95)&(R>G+60)
        for name,mask in (("ring_yellow",yellow),("ring_red",red)):
            bs=[b for b in blobs(mask,40) if b["area"]<2500]
            if bs: b=bs[0]; res["ring"]={"px":[b["cx"],b["cy"]],"area":b["area"],"color":name,"extent":extent(b)}; break
    elif task=="tool_insert":
        dark=(V<75); red=(R>95)&(G<75)&(B<75)
        traym=dark|red
        bs=blobs(traym,5000)
        if bs: res["tray"]={"px":[bs[0]["cx"],bs[0]["cy"]],"area":bs[0]["area"],"bbox":bs[0]["bbox"],"extent":extent(bs[0])}
        tb=res.get("tray",{}).get("bbox",[0,0,1024,576])
        metal=(V>135)&(S<0.30)&(B>R-25)
        bs=[]
        for b in blobs(metal,25):
            inside = tb[0]-5<=b["cx"]<=tb[2]+5 and tb[1]-5<=b["cy"]<=tb[3]+5
            y,r=pca_yaw(b)
            if not inside and b["cy"]>tb[3]-40 and r and r>2.2:
                bs.append((b,y,r))
        if bs:
            b,y,r=bs[0]; res["tool"]={"px":[b["cx"],b["cy"]],"area":b["area"],"yaw_px":y,"elong":r,"extent":extent(b)}
    elif task=="cable_clip":
        white=(V>188)&(S<0.18)
        bs=[b for b in blobs(white,350) if b["area"]<20000]
        if bs: b=bs[0]; y,r=pca_yaw(b); res["block"]={"px":[b["cx"],b["cy"]],"area":b["area"],"yaw_px":y,"extent":extent(b),"bbox":b["bbox"]}
        darkm=(V<48)
        lab,n=ndimage.label(ndimage.binary_opening(darkm,np.ones((3,3))))
        arm=np.zeros_like(darkm)
        # arm = dark component touching bottom edge rows
        bot=set(np.unique(lab[560:,:]))-{0}
        for i in bot: arm|= (lab==i)
        cable=darkm&(~arm)
        ys,xs=np.nonzero(cable)
        keep=(ys<540)
        if keep.sum()>200:
            res["cable"]={"px":[float(xs[keep].mean()),float(ys[keep].mean())],"area":int(keep.sum()),
                          "bbox":[int(xs[keep].min()),int(ys[keep].min()),int(xs[keep].max()),int(ys[keep].max())]}
    return res

allres={}
for d in sorted(glob.glob(FR+"/*")):
    ep=os.path.basename(d); task=ep.rsplit("__ep",1)[0]
    entry={}
    imf=load(d+"/top_first.png"); iml=load(d+"/top_final.png")
    entry["first"]=detect(task,imf,"first"); entry["final"]=detect(task,iml,"final")
    allres[ep]=entry
    # overlay on top_first
    ov=Image.open(d+"/top_first.png").convert("RGB"); dr=ImageDraw.Draw(ov)
    cols={"cube_purple":"magenta","cube_green":"lime","cube_orange":"orange","carton":"cyan","ring":"yellow",
          "board":"white","tray":"red","tool":"cyan","block":"cyan","cable":"yellow","platform":"white"}
    for name,b in entry["first"].items():
        if "px" not in b: continue
        x,y=b["px"]; c=cols.get(name,"red")
        dr.ellipse([x-9,y-9,x+9,y+9],outline=c,width=3); dr.text((x+11,y-6),name,fill=c)
        if "bbox" in b: dr.rectangle(b["bbox"],outline=c,width=2)
    ov.save(OVL+"/"+ep+".png")
json.dump(allres,open(OUT+"/detections_px.json","w"),indent=1)
# summary
from collections import Counter
for task in ["block_stack","eye_drops_to_basket","ring_insert","tool_insert","cable_clip"]:
    eps={k:v for k,v in allres.items() if k.startswith(task)}
    keys=Counter()
    for v in eps.values():
        for w in ("first","final"): keys.update([w+":"+k for k in v[w]])
    print(task, dict(keys))
