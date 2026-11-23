import numpy as np, json, glob, os
from PIL import Image
from scipy import ndimage
from matplotlib.path import Path
FR="/home/openclaw/muse-work/zk0-study001-twin-20261122/frames"
OUT="/home/openclaw/muse-work/zk0-study001-twin-20261122/scenes"
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
def interior(bs,reg,tol=3):
    x0,y0,x1,y1=reg
    return [b for b in bs if b["bbox"][0]>x0+tol and b["bbox"][1]>y0+tol and b["bbox"][2]<x1-tol and b["bbox"][3]<y1-tol]
med={}
for task in ("eye_drops_to_basket","tool_insert"):
    for which in ("first","final"):
        ims=[load(d+f"/top_{which}.png") for d in sorted(glob.glob(FR+"/"+task+"__*"))]
        med[(task,which)]=np.median(np.stack(ims),0)
res={}
for d in sorted(glob.glob(FR+"/*")):
    ep=os.path.basename(d); task=ep.rsplit("__ep",1)[0]
    if task not in ("eye_drops_to_basket","tool_insert"): continue
    res[ep]={}
    for which in ("first","final"):
        im=load(d+f"/top_{which}.png"); R,G,B,V,S=chans(im)
        diff=(np.abs(im-med[(task,which)]).max(2)>26)|(np.abs(im-med[(task,"first")]).max(2)>26)
        out={}
        if task=="eye_drops_to_basket":
            blue=(B>R+18)&(B>G+0)&(S>0.20)&(V>45)&(V<225)
            if which=="first":
                reg=(398,138,688,302)
                region=np.zeros_like(ROI); region[reg[1]:reg[3],reg[0]:reg[2]]=True
                cand=interior(blobs((blue|diff)&ROI&region,300,12000),reg)
                if not cand: cand=interior(blobs(diff&ROI&region,300,12000),reg)
                if cand: out["carton"]=pack(cand[0])
            else:
                reg=(55,150,345,405)
                region=np.zeros_like(ROI); region[reg[1]:reg[3],reg[0]:reg[2]]=True
                cand=interior(blobs((blue|diff)&ROI&region,300,12000),reg)
                if not cand: cand=blobs((blue|diff)&ROI&region,300,12000)
                if cand: out["carton"]=pack(cand[0])
        else:
            if which=="first":
                mtr=((V<78)|((R>95)&(G<78)&(B<78)))&ROI
                tb=blobs(mtr,20000,300000)
                tbb=tb[0]["bbox"] if tb else [240,50,870,370]
                metal=(V>128)&(S<0.34)
                cands=[]
                for b in blobs(metal&ROI&diff,200,6000):
                    ox=max(0,min(b["bbox"][2],tbb[2]+4)-max(b["bbox"][0],tbb[0]-4)); oy=max(0,min(b["bbox"][3],tbb[3]+4)-max(b["bbox"][1],tbb[1]-4))
                    inter=ox*oy; ab=(b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1])
                    if ab>0 and inter/ab>0.10: continue
                    y,r,ext=pca(b)
                    if r and r>1.8: cands.append(b)
                if cands: out["tool"]=pack(cands[0])
                out["tray_bbox"]=tbb
        res[ep][which]=out
    print(ep,res[ep]["first"].get("carton" if task=="eye_drops_to_basket" else "tool",{}).get("px"),
          res[ep]["final"].get("carton",{}).get("px") if task=="eye_drops_to_basket" else "")
json.dump(res,open(OUT+"/detections4_px.json","w"),indent=1)
