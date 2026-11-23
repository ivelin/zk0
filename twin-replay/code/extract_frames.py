"""Extract first/final frames for the 100 frozen Study001 eval episodes.
Mirrors eval scripts/frames.py mapping: episode -> per-cam chunk/file +
from/to timestamps in meta/episodes parquet; first = from_timestamp,
final = to_timestamp - 1/FPS (FPS=20). Decoding via imageio-ffmpeg binary
(PyAV not available in any current venv; ffmpeg input-seek is frame-accurate).
Only episodes listed in split/split.json (5 in-scope tasks) are touched.
"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

DS = Path("/home/openclaw/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101")
EVAL = Path("/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122")
OUT = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122/frames")
FFMPEG = "/home/openclaw/muse-work/zk0-sim-v01-20261115/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v9.0.2"
FPS = 20
ep = pd.read_parquet(DS / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")

def grab(epi, cam, frac, dest):
    r = ep.loc[epi]
    ci, fi = int(r[f"videos/observation.images.{cam}/chunk_index"]), int(r[f"videos/observation.images.{cam}/file_index"])
    t0, t1 = float(r[f"videos/observation.images.{cam}/from_timestamp"]), float(r[f"videos/observation.images.{cam}/to_timestamp"])
    path = DS / f"videos/observation.images.{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4"
    target = t0 + frac * max(t1 - t0 - 1 / FPS, 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run([FFMPEG, "-loglevel", "error", "-ss", f"{target:.3f}", "-i", str(path),
                          "-frames:v", "1", "-y", str(dest)], capture_output=True, text=True)
    if res.returncode != 0 or not dest.exists():
        return (str(dest), res.stderr[:300])
    return None

def main():
    split = json.loads((EVAL / "split/split.json").read_text())
    jobs = []
    for task, v in split["tasks"].items():
        for rec in v["episodes"]:
            epi = rec["episode_index"]
            for cam in ["front", "top", "wrist"]:
                for frac, tag in [(0.0, "first"), (1.0, "final")]:
                    dest = OUT / f"{task}__ep{epi}" / f"{cam}_{tag}.png"
                    if not dest.exists():
                        jobs.append((epi, cam, frac, dest))
    print(f"jobs: {len(jobs)}", flush=True)
    fails = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, f in enumerate(ex.map(lambda j: grab(*j), jobs)):
            if f: fails.append(f)
            if (i + 1) % 100 == 0: print(f"{i+1}/{len(jobs)} fails={len(fails)}", flush=True)
    print("FAILURES:", fails)
    print("done, failures:", len(fails))

if __name__ == "__main__":
    main()
