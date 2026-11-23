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
def blobs(mask,min_area=60,max_area=200000):
    mask=ndimage.binary_opening(mask,np.ones((3,3)))
    lab,n=ndimage.label(mask); out=[]
    for i in range(1,n+1):
        ys,xs=np.nonzero(lab==i)
        if not (min_area<=len(xs)<=max_area): continue
        out.append({"cx":float(xs.mean()),"cy":float(ys.mean()),"area":int(len(xs)),
                    "bbox":[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]})
    return sorted(out,key=lambda b:-b["area"])
res={}
for d in sorted(glob.glob(FR+"/*")):
    ep=os.path.basename(d); task=ep.rsplit("__ep",1)[0]; res[ep]={}
    im=load(d+"/top_first.png"); R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]; V=(R+G+B)/3; S=(im.max(2)-im.min(2))/(im.max(2)+1e-6)
    if task=="ring_insert":
        cream=(V>178)&(R>B+22)&(B>125)&(S<0.38)
        bs=[b for b in blobs(cream&ROI,12000,70000)]
        if bs: res[ep]["board"]=bs[0]
    if task=="cable_clip":
        white=(V>185)&(S<0.20)
        bs=blobs(white&ROI,2500,26000)
        if bs:
            b=bs[0]; top=np.zeros_like(white); top[b["bbox"][1]:b["bbox"][1]+150, b["bbox"][0]:b["bbox"][2]]=True
            sub=[x for x in blobs(white&ROI&top,1500,26000)]
            cx,cy=(sub[0]["cx"],sub[0]["cy"]) if sub else (b["cx"],b["bbox"][1]+60)
            # black clip: dark blob in top part of block bbox
            dark=(V<62); reg=np.zeros_like(white); reg[max(0,b["bbox"][1]-15):b["bbox"][1]+95, b["bbox"][0]:b["bbox"][2]]=True
            cl=[x for x in blobs(dark&reg,250,3000)]
            res[ep]["block_refined"]={"px":[cx,cy],"raw_px":[b["cx"],b["cy"]],"bbox":b["bbox"],
                                      "clip_px":[cl[0]["cx"],cl[0]["cy"]] if cl else None}
    print(ep, res[ep])
json.dump(res,open(OUT+"/detections5_px.json","w"),indent=1)
