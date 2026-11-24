#!/usr/bin/env python3
"""Filtered variant of make_sheets.py: render sheets only for episodes
whose episode_index falls in the given ranges.
Usage: make_sheets_filtered.py <dataset_dir> <cam_row1> <cam_row2>
       <out_dir> <prefix> <ranges> [workers]
<ranges>: comma-separated lo-hi pairs, e.g. 1200-1249,1549-1598,2449-2498
Emits <out_dir>/<prefix>__ep<idx>.jpg and <out_dir>/<prefix>_episodes.csv
(episode_index, task_string, length_frames) for the filtered episodes.
Idempotent: skips existing sheets."""
import sys, os, glob, csv
import pandas as pd
from PIL import Image
import av
from multiprocessing import Pool

FRACS = [0.0, 0.35, 0.60, 0.80, 0.92, 0.999]
TW, TH = 320, 193

def load_episodes(ds):
    files = sorted(glob.glob(os.path.join(ds, "meta", "episodes", "**", "*.parquet"), recursive=True))
    if not files:
        files = [os.path.join(ds, "meta", "episodes.parquet")]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def frame_at(path, tsec):
    with av.open(path) as c:
        s = c.streams.video[0]
        tb = s.time_base
        target = int(tsec / float(tb))
        c.seek(target, stream=s, backward=True)
        last = None
        for frame in c.decode(s):
            last = frame
            if float(frame.pts * tb) >= tsec:
                break
        if last is None:
            raise RuntimeError(f"no frame at {tsec} in {path}")
        return last.to_image()

def build_sheet(args):
    ds, cam1, cam2, out, prefix, row = args
    idx = int(row["episode_index"]); length = int(row["length"])
    dest = os.path.join(out, f"{prefix}__ep{idx}.jpg")
    if os.path.exists(dest):
        return (idx, "skip")
    sheet = Image.new("RGB", (TW * 6, TH * 2))
    for r, cam in enumerate([cam1, cam2]):
        ch = int(row[f"videos/{cam}/chunk_index"]); fi = int(row[f"videos/{cam}/file_index"])
        fts = float(row[f"videos/{cam}/from_timestamp"])
        path = os.path.join(ds, "videos", cam, f"chunk-{ch:03d}", f"file-{fi:03d}.mp4")
        fps = length / max(float(row[f"videos/{cam}/to_timestamp"]) - fts, 1e-9)
        for j, fr in enumerate(FRACS):
            t = fts + fr * (length - 1) / fps
            img = frame_at(path, t).resize((TW, TH))
            sheet.paste(img, (j * TW, r * TH))
    sheet.save(dest, quality=88)
    return (idx, "ok")

def main():
    ds, cam1, cam2, out, prefix, ranges = sys.argv[1:7]
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    keep = set()
    for part in ranges.split(","):
        lo, hi = part.split("-")
        keep.update(range(int(lo), int(hi) + 1))
    os.makedirs(out, exist_ok=True)
    eps = load_episodes(ds)
    eps = eps[eps["episode_index"].isin(keep)].sort_values("episode_index")
    with open(os.path.join(out, f"{prefix}_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode_index", "task_string", "length_frames"])
        for _, r in eps.iterrows():
            tasks = r["tasks"]
            tstr = tasks[0] if hasattr(tasks, "__len__") and len(tasks) else str(tasks)
            w.writerow([int(r["episode_index"]), tstr, int(r["length"])])
    jobs = [(ds, cam1, cam2, out, prefix, r) for _, r in eps.iterrows()]
    done = {"ok": 0, "skip": 0, "err": 0}
    with Pool(workers) as p:
        for i, res in enumerate(p.imap_unordered(build_sheet, jobs, chunksize=4)):
            done[res[1] if res[1] in done else "err"] += 1
            if (i + 1) % 50 == 0:
                print(f"{prefix}: {i+1}/{len(jobs)} {done}", flush=True)
    print(f"{prefix}: DONE {done} of {len(jobs)}", flush=True)

if __name__ == "__main__":
    main()
