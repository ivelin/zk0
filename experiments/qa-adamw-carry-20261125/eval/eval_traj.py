"""Study 001 eval layer: open-loop trajectory matching for one checkpoint.
See eval-plan.md. Deterministic: fixed seeded noise per (episode, chunk)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch, os
DEV = os.environ.get("EVAL_DEVICE", "cuda")

# Dataset root: a local copy of HF zk0bot/armnetbench_v01_lerobot_so101 @ 89eef614
# (download the snapshot, then point QA_DATASET_ROOT at it).
DS_ROOT = os.environ.get("QA_DATASET_ROOT", "armnetbench_v01_lerobot_so101")
REPO_ID = "armnet/armnetbench_v01_lerobot_so101"
CHUNK = 50

def load_gt_actions(episodes_wanted):
    gt = {}
    for f in sorted(Path(DS_ROOT, "data").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "action", "frame_index"]).to_pandas()
        for epi, g in t.groupby("episode_index"):
            if int(epi) in episodes_wanted:
                gt[int(epi)] = np.asarray(g.sort_values("frame_index")["action"].tolist(), dtype=np.float64)
    return gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()

    # P1 eval gate (audit F2, 2026-11-22): refuse to start without a
    # PASS eval gate report covering this checkpoint (identical declared
    # input features across the comparison set + frozen split).
    # EVAL_GATE_OVERRIDE=1 is the deliberate, logged escape.
    import hashlib as _hl
    _rep = os.environ.get("EVAL_GATE_REPORT") or str(
        Path(args.split).resolve().parent.parent / "eval_gate_report.json")
    if os.environ.get("EVAL_GATE_OVERRIDE") == "1":
        print("[eval] WARNING: EVAL GATE OVERRIDDEN DELIBERATELY "
              "(EVAL_GATE_OVERRIDE=1)", flush=True)
    else:
        _ok, _why = False, f"no eval gate report at {_rep}"
        try:
            _r = json.loads(Path(_rep).read_text())
            _sha = _hl.sha256(
                (Path(args.ckpt_path) / "config.json").read_bytes()).hexdigest()
            _covered = [e for e in _r.get("checkpoints", {}).values()
                        if e.get("config_sha256") == _sha]
            _ok = _r.get("verdict") == "PASS" and bool(_covered)
            if not _ok:
                _why = (f"report at {_rep} is not a PASS covering this "
                        f"checkpoint (verdict={_r.get("verdict")}, "
                        f"config sha matched={bool(_covered)})")
        except Exception as e:
            _why = f"unreadable eval gate report at {_rep}: {e}"
        if not _ok:
            print(f"[eval] REFUSING to start: P1 eval gate — {_why}. "
                  "Run study_prep/eval_gate_check.py over the full "
                  "comparison set first, or set EVAL_GATE_OVERRIDE=1 to "
                  "override deliberately.", flush=True)
            sys.exit(3)
        print(f"[eval] eval gate PASS verified ({_rep})", flush=True)

    split = json.loads(Path(args.split).read_text())
    tasks = args.tasks.split(",") if args.tasks else list(split["tasks"].keys())
    ep_list = []  # (task, episode_index, length)
    for t in tasks:
        for r in split["tasks"][t]["episodes"]:
            ep_list.append((t, int(r["episode_index"]), int(r["length_frames"])))
    if args.max_episodes:
        ep_list = ep_list[: args.max_episodes]
    wanted = {e for _, e, _ in ep_list}
    gt = load_gt_actions(wanted)
    print(f"[eval] {args.ckpt_name}: {len(ep_list)} episodes", flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    ds = LeRobotDataset(REPO_ID, root=DS_ROOT, episodes=sorted(wanted))
    # relative index map: verify via episode_index column
    ep_col = ds.hf_dataset["episode_index"]
    rel_start = {}
    for i, e in enumerate(ep_col):
        if int(e) not in rel_start:
            rel_start[int(e)] = i
    for e in wanted:
        assert int(ep_col[rel_start[e]]) == e

    DS_CAMS = ["observation.images.front", "observation.images.top", "observation.images.wrist"]
    policy = SmolVLAPolicy.from_pretrained(args.ckpt_path)
    expected = sorted(policy.config.image_features.keys())
    if set(expected) == set(DS_CAMS):
        CAM_MAP = {c: c for c in DS_CAMS}
    else:  # e.g. base smolvla_base expects camera1/camera2/camera3 (prereg sec.5 mapping)
        CAM_MAP = dict(zip(DS_CAMS, expected))
    print(f"[eval] image key map: {CAM_MAP}", flush=True)
    policy.to(DEV).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt_path)
    # A5: drop rename steps that rename AWAY from the policy's declared features
    try:
        want = set(policy.config.image_features.keys()) | {"observation.state"}
        kept = []
        for st in pre.steps:
            rn = getattr(st, "rename_map", None)
            if rn and any(k in want and v not in want for k, v in rn.items()):
                print(f"[eval] dropping rename step {rn}", flush=True)
                continue
            kept.append(st)
        pre.steps = kept
    except Exception as e:
        print(f"[eval] rename-step filter skipped: {e}", flush=True)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for task, epi, length in ep_list:
        t0 = time.time()
        preds = np.zeros((length, 6), dtype=np.float64)
        n_chunks = (length + CHUNK - 1) // CHUNK
        for c in range(n_chunks):
            start = c * CHUNK
            item = ds[rel_start[epi] + start]
            batch = {}
            for ds_key, pol_key in CAM_MAP.items():
                batch[pol_key] = item[ds_key].unsqueeze(0).to(DEV)
            batch["observation.state"] = item["observation.state"].unsqueeze(0).to(DEV)
            batch["task"] = [item["task"]]
            batch = pre(batch)
            batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
            g = torch.Generator(device=DEV).manual_seed(epi * 1000 + c)
            noise = torch.randn((1, CHUNK, policy.config.max_action_dim), generator=g, device=DEV, dtype=batch["observation.state"].dtype)
            with torch.no_grad():
                chunk = policy.predict_action_chunk(batch, noise=noise)
            try:
                act = post({"action": chunk})["action"]
            except Exception:
                act = post(chunk)
            act = act[0].float().cpu().numpy()
            if act.shape[-1] > 6: act = act[:, :6]
            preds[start:start + CHUNK] = act[: length - start]
        gta = gt[epi][:length]
        err = np.linalg.norm(preds - gta, axis=1)
        np.savez_compressed(out_dir / f"{args.ckpt_name}__{task}__ep{epi}.npz",
                            gt=gta, pred=preds, err=err)
        horizon = [float(np.mean(err[c * CHUNK:(c + 1) * CHUNK].reshape(-1)[
            np.arange(min(CHUNK, length - c * CHUNK)) == h])) if False else None for h in []]
        # horizon profile: mean err by within-chunk position
        hor = np.full(CHUNK, np.nan)
        for h in range(CHUNK):
            vals = [err[c * CHUNK + h] for c in range(n_chunks) if c * CHUNK + h < length]
            if vals: hor[h] = float(np.mean(vals))
        chunk_prof = [float(np.mean(err[c * CHUNK:(c + 1) * CHUNK])) for c in range(n_chunks)]
        s = {"image_key_map": CAM_MAP, "checkpoint": args.ckpt_name, "task": task, "episode_index": epi,
             "length": length, "mean_err": float(np.mean(err)), "p50_err": float(np.percentile(err, 50)),
             "p90_err": float(np.percentile(err, 90)), "mean_l1": float(np.mean(np.abs(preds - gta))),
             "horizon_profile": [None if np.isnan(x) else x for x in hor],
             "chunk_profile": chunk_prof, "wall_s": round(time.time() - t0, 2)}
        summaries.append(s)
        print(f"[eval] {args.ckpt_name} {task} ep{epi} chunk {c+1}/{n_chunks}", flush=True)
        print(f"[eval] {args.ckpt_name} {task} ep{epi}: mean_err={s['mean_err']:.3f} ({s['wall_s']}s)", flush=True)
    (out_dir / f"summary__{args.ckpt_name}.json").write_text(json.dumps(summaries, indent=1))
    print(f"[eval] {args.ckpt_name} DONE {len(summaries)} episodes", flush=True)

if __name__ == "__main__":
    main()
