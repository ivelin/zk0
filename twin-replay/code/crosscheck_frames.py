"""Cross-check twin frames vs eval sample/ frames and judgments/frames.
Metric: mean absolute pixel difference (0-255) and fraction of pixels differing >10.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
TWIN = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122/frames")
EVAL = Path("/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122")
def mad(a, b):
    x = np.asarray(Image.open(a), dtype=np.int16); y = np.asarray(Image.open(b), dtype=np.int16)
    if x.shape != y.shape: return (f"SHAPE {x.shape} vs {y.shape}", 1.0)
    d = np.abs(x - y)
    return (f"MAD={d.mean():.3f}", float((d > 10).mean()))
# 1) vs sample/ for the episodes present there that are in the frozen set
checks = [("block_stack",489),("eye_drops_to_basket",1810),("tool_insert",852),("block_stack",499),("ring_insert",165)]
import glob
for task, epi in checks:
    sdirs = glob.glob(str(EVAL / "sample" / f"*__ep{epi}"))
    if not sdirs: print(task, epi, "no sample dir"); continue
    sd = Path(sdirs[0])
    for cam in ["front","top","wrist"]:
        for tag in ["first","final"]:
            mine = TWIN / f"{task}__ep{epi}" / f"{cam}_{tag}.png"
            ref = sd / f"{cam}_{tag}.png"
            print(f"sample {task} ep{epi} {cam}_{tag}:", mad(mine, ref) if mine.exists() and ref.exists() else "MISSING")
# 2) vs judgments/frames for ALL episodes
import json
split = json.loads((EVAL/"split/split.json").read_text())
JF = EVAL/"judgments/frames"
tot=[]; worst=[]
for task, v in split["tasks"].items():
    for rec in v["episodes"]:
        epi = rec["episode_index"]
        for cam in ["front","top","wrist"]:
            for tag in ["first","final"]:
                mine = TWIN/f"{task}__ep{epi}"/f"{cam}_{tag}.png"; ref = JF/f"ep{epi}_{cam}_{tag}.png"
                if not mine.exists() or not ref.exists(): worst.append((999,task,epi,cam,tag,"MISSING")); continue
                s, frac = mad(mine, ref); val = float(s.split("=")[1]) if "=" in s else 999
                tot.append(val); worst.append((val,task,epi,cam,tag,frac))
worst.sort(reverse=True)
print(f"judgments check n={len(tot)} mean MAD={np.mean(tot):.3f} max MAD={np.max(tot):.3f}")
print("worst 10:", worst[:10])
