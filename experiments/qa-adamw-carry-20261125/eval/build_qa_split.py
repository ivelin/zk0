"""Q-A eval setup (E-2 + E-4): build the 90-episode eval split from the verified
lists (qa-eval-prep-20261126.md S1), diff vs Study 001 split.json, verify
disjointness vs the pinned 239 training lists, fix the canary subset, and
derive the Q-A eval gate manifest with the canonical split hash."""
import json, hashlib, os, random
from pathlib import Path
import pandas as pd

S001_SPLIT = Path(__file__).with_name("study001_split.json")  # Study 001 split, shipped in this folder
DS = Path(os.environ.get("QA_DATASET_ROOT", "armnetbench_v01_lerobot_so101"))  # HF zk0bot/armnetbench_v01_lerobot_so101 @ 89eef614
TRAIN = Path(__file__).resolve().parent.parent / "episode_lists"  # pinned 239 lists, in this package
BASE_MANIFEST = Path(__file__).with_name("qa_gate_manifest_239_20261125.json")  # shipped in this folder
EVAL = Path(os.environ.get("QA_EVAL_TREE", Path(__file__).resolve().parent))

LISTS = {
 "block_stack": [411,453,476,479,489,498,499,513,514,525,526,527,530,531,539,545,546,548,549,557],
 "eye_drops_to_basket": [1609,1618,1630,1639,1643,1724,1741,1746,1747,1757,1760,1778,1780,1783,1798,1808,1810,1811,1814,1821],
 "ring_insert": [7,11,30,31,33,48,50,160,165,167,168,177,179,188,195,198,227,240,242,253],
 "tool_insert": [622,629,630,683,684,697,779,784,824,830,852,859,871,876,883,885,888,892,902,903],
 "cable_clip": [2164,2169,2170,2183,2190,2193,2194,2195,2200,2208],
}
TASKS = list(LISTS.keys())

s001 = json.loads(S001_SPLIT.read_text())
ep = pd.read_parquet(DS / "meta/episodes/chunk-000/file-000.parquet")
ep = ep.set_index("episode_index")

split = {"seed": 20261126, "per_task": 20, "tasks": {}, "clusters": s001["clusters"],
         "n_episodes": 90, "disjoint_from_training": True, "training_index_count": 239,
         "qa_note": ("Q-A eval split, built 2026-11-27 from the verified lists in "
                     "qa-eval-prep-20261126.md S1 (HF certified manifest rows 239-328, "
                     "split=heldout; cable = founder re-split 2026-11-23, seed 20261123). "
                     "seed field records the prep/verification date; the set was enumerated "
                     "from the certified manifest, not seed-drawn. Clusters copied verbatim "
                     "from the Study 001 split (dataset-level successful-execution stats).")}

report = {}
for t in TASKS:
    want = LISTS[t]
    if t != "cable_clip":
        src = {int(r["episode_index"]): r for r in s001["tasks"][t]["episodes"]}
        s001_set = set(src.keys())
        assert set(want) == s001_set, f"E-2 DIFF {t}: qa-only={sorted(set(want)-s001_set)} s001-only={sorted(s001_set-set(want))}"
        entries = [src[e] for e in want]
        pool = s001["tasks"][t]
    else:
        entries = []
        for e in want:
            row = ep.loc[e]
            entries.append({"episode_index": int(e),
                            "source_policy_type": str(row["policy_type"]),
                            "recorded_success": int(row["success"]),
                            "length_frames": int(row["length"])})
        pool = {"pool_success_only": True}
    split["tasks"][t] = {"pool_success_only": pool.get("pool_success_only", True),
                         "pool_size": len(entries), "episodes": entries}
    report[t] = len(entries)

# disjointness vs pinned training lists + cable cross-check vs run_set_manifest
rsm = json.loads((TRAIN / "run_set_manifest.json").read_text())
assert sorted(rsm["cable_heldout_excluded"]) == sorted(LISTS["cable_clip"]), "cable != run_set_manifest exclusion"
for t in TASKS:
    train_idx = set(json.loads((TRAIN / f"episodes_{t}.json").read_text()))
    inter = train_idx & set(LISTS[t])
    assert not inter, f"OVERLAP {t}: {sorted(inter)}"
    assert len(train_idx) + len(LISTS[t]) == len(train_idx | set(LISTS[t]))
total_train = sum(len(json.loads((TRAIN / f"episodes_{t}.json").read_text())) for t in TASKS)
assert total_train == 239, total_train
assert sum(report.values()) == 90

(EVAL / "split.json")  # package layout (as-run tree used split/split.json).write_text(json.dumps(split, indent=1))

# canonical rows hash, exactly the gate's formula
rows = []
for t in sorted(split["tasks"]):
    for r in split["tasks"][t]["episodes"]:
        rows.append([t, int(r["episode_index"]), int(r["length_frames"])])
rows = sorted(rows)
rows_sha = hashlib.sha256(json.dumps(rows).encode()).hexdigest()

# canary (E-4): fixed stratified subset, 8/task, seeded, recorded before judging
rng = random.Random(20261127)
canary = {t: sorted(rng.sample(sorted(LISTS[t]), 8)) for t in TASKS}
(EVAL / "canary.json").write_text(json.dumps({
    "method": "random.Random(20261127).sample(sorted(eval list), 8) per task; fixed before judging; same subset at every read (canary standard, FL_SEARCH_QUEUE item 5)",
    "n": 40, "per_task": 8, "canary": canary,
    "u6_note": "block slice membership of eps 489/499 to be re-checked against the founder's U6 ruling when given (eval prep S3/E-4)",
}, indent=1))

# Q-A eval manifest: derive from the Q-A gate manifest; replace eval_gate.split; register the cable-counts waiver
man = json.loads(BASE_MANIFEST.read_text())
man["eval_gate"]["split"] = {
    "canonical_rows": "sha256 of json list of [task, episode_index, length_frames] sorted by task then episode_index",
    "canonical_rows_sha256": rows_sha,
    "disjoint_from_training": True,
    "n_episodes": 90,
    "per_task": 20,
    "seed": 20261126,
    "training_index_count": 239,
}
man["registered_waivers"] = man.get("registered_waivers", []) + [{
    "id": "W-cable-resplit-counts",
    "diff": "eval split per-task counts are 20/20/20/20/10 (cable_clip 10) instead of a uniform 20 per task",
    "source": "founder cable re-split decision 2026-11-23 (original held-out cable 20 were 20/20 FAIL and removed; new certified held-out cable set = 10, seed 20261123); Q-A run spec denominator 90 (qa-eval-prep-20261126.md S1)",
}]
man["qa_eval_manifest_note"] = ("Derived 2026-11-27 from qa_gate_manifest_239_20261125.json for the Q-A eval only: "
    "eval_gate.split now pins the verified 90-episode Q-A split (canonical_rows_sha256 over split/split.json in "
    "zk0-qa-resume-20261126/eval); expected_policy_config, expected_preprocessor and all other sections are "
    "carried over unchanged from the Q-A training gate manifest.")
(EVAL / "qa_eval_manifest_90.json").write_text(json.dumps(man, indent=1))

print("SPLIT OK:", report, "total=90 train=239 rows_sha=", rows_sha[:16])
print("CANARY:", json.dumps(canary))
