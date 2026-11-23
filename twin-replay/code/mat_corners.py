import numpy as np, json, glob, os
from PIL import Image
from scipy import ndimage
FR="/home/openclaw/muse-work/zk0-study001-twin-20261122/frames"
res={}
for d in sorted(glob.glob(FR+"/*")):
    ep=os.path.basename(d)
    im=np.asarray(Image.open(d+"/top_first.png").convert("RGB")).astype(float)
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
    # mat: dark slate blue-grey: low value, B>=R, not tan cardboard (R high), not black arm (very low)
    v=(R+G+B)/3
    mat=(v>40)&(v<130)&(B>=R-5)&(R<120)
    mat=ndimage.binary_opening(mat,np.ones((5,5)))
    lab,n=ndimage.label(mat)
    if n==0: res[ep]=None; continue
    sizes=ndimage.sum(mat,lab,range(1,n+1)); k=1+int(np.argmax(sizes))
    ys,xs=np.nonzero(lab==k)
    pts=np.stack([xs,ys],1).astype(float)
    # corners: extremes along diagonals
    s=pts.sum(1); df=pts[:,0]-pts[:,1]
    corners={"TL":pts[np.argmin(s)].tolist(),"BR":pts[np.argmax(s)].tolist(),
             "TR":pts[np.argmax(df)].tolist(),"BL":pts[np.argmin(df)].tolist(),
             "centroid":[float(xs.mean()),float(ys.mean())],"area":int(len(xs))}
    res[ep]=corners
json.dump(res,open("/home/openclaw/muse-work/zk0-study001-twin-20261122/scenes/mat_corners.json","w"),indent=1)
arr={k:v for k,v in res.items() if v}
import collections
for c in ["TL","TR","BR","BL","centroid"]:
    P=np.array([v[c] for v in arr.values()]); print(c,"mean",P.mean(0).round(1),"std",P.std(0).round(1),"n",len(P))
print("areas", np.array([v["area"] for v in arr.values()]).mean())
