#!/usr/bin/env python3
"""eval_gate_check.py — Study 001 P1 EVAL-side gate (audit finding F2).

The camera defect (Incident 1 / defect D9) did its damage at EVAL time:
the FL checkpoint declared two views while every other arm declared
three, and the eval fed each checkpoint its own declared inputs, so
the comparison was silently unequal. The training-side gate
(config_equivalence_check.py) cannot see that. This gate can.

Before any trajectory pass or judging run over a comparison set, run:

  venv/bin/python study_prep/eval_gate_check.py \
      --manifest study_prep/fl_rerun_study_manifest.json \
      --split /path/to/eval/split/split.json \
      --ckpt base=/path/to/base_snapshot \
      --ckpt cexp=/path/to/checkpoint \
      --ckpt fl_r30=/path/to/checkpoint_round_30 \
      --out /path/to/eval_gate_report.json

Checks (any failure exits 1 and names the offending checkpoint):
- every checkpoint's config.json declares input_features that, in the
  post-rename namespace, are SHAPE-EXACTLY equal to the registered
  design (expected_policy_config.input_features) — hence identical
  across the whole comparison set (asserted pairwise as well);
- every checkpoint declares the registered output_features;
- every checkpoint directory carries its packaged
  policy_preprocessor.json (the A5 packaging defect made a checkpoint
  whose own preprocessor contradicted its config);
- the eval split is the frozen one: seed, per-task count, total
  count, disjoint-from-training flag, and the canonical episode-list
  hash all match the manifest's eval_gate section.

The eval harness (eval_traj.py trajectory pass, judge_packets.py
judging pass) refuses to start without a PASS report from this gate
covering the checkpoint being evaluated (EVAL_GATE_OVERRIDE=1 is the
deliberate, logged escape). The launcher also refuses eval launches
without one (launch_dgx_job.sh study mode).

Exit codes: 0 = PASS, 1 = DIFF-FAIL, 2 = error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "config_equivalence_check", _HERE / "config_equivalence_check.py"
)
cec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cec)

Gate = cec.Gate
sha256_bytes = cec.sha256_bytes
sha256_file = cec.sha256_file
norm_features = cec.norm_features
features_of_config = cec.features_of_config
exact_features_match = cec.exact_features_match


def canonical_split_rows(split: dict) -> list:
    rows = []
    for t in sorted(split["tasks"]):
        for r in split["tasks"][t]["episodes"]:
            rows.append([t, int(r["episode_index"]), int(r["length_frames"])])
    return sorted(rows)  # fully canonical: task, then episode, then length




# ---------------------------------------------------------------------------
# Gap G-12 (rounds 10/20 prep G3): admission requires a standard-load
# proof, not existence of policy_preprocessor.json. A raw round
# checkpoint HAS a preprocessor file (empty rename map, no normalizer
# state, no tokenizer) and passed the old existence check; a
# config-correct but unpackaged checkpoint would pass the whole gate
# and fail only at load time (defect A5's class). The proof below
# constructs the packaged processors in a fresh process and runs a
# dummy batch through them, under the eval harness's own loading
# contract: every declared image feature must be produced. It runs
# on a temp copy whose device step is rewritten to
# CPU, with CUDA hidden - the proof never touches the GPU and never
# modifies the artifact of record.
# ---------------------------------------------------------------------------

_PROBE_SRC = """
import json, sys
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

ckpt = sys.argv[1]
declared = json.loads(sys.argv[2])
cfg = PreTrainedConfig.from_pretrained(ckpt)
pre, _post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=ckpt)
# Mirror the eval harness contract (eval_traj.py): the harness feeds
# batches already keyed by the policy's declared image features and
# drops rename steps that rename a declared feature (or the state)
# AWAY from the declared set (the A5 filter). The proof therefore
# tests exactly what the harness will do at load time: processors
# construct, the batch runs through, every declared image feature
# survives.
want = set(declared) | {"observation.state"}
kept = []
for st in pre.steps:
    rn = getattr(st, "rename_map", None)
    if rn and any(k in want and v not in want for k, v in rn.items()):
        continue
    kept.append(st)
pre.steps = kept
batch = {k: torch.zeros((1, 3, 240, 320), dtype=torch.uint8) for k in declared}
batch["observation.state"] = torch.zeros((1, 6))
batch["task"] = ["standard load proof"]
out = pre(batch)
missing = [k for k in declared if k not in out]
print(json.dumps({"missing": missing, "produced": sorted(out.keys())}))
raise SystemExit(1 if missing else 0)
"""


def rewrite_device_steps_cpu(preprocessor: dict) -> dict:
    """Return a copy of a serialized preprocessor pipeline with every
    device step set to CPU (the artifact of record keeps its device;
    only a temp copy used by proofs is rewritten)."""
    import copy

    out = copy.deepcopy(preprocessor)
    steps = out.get("steps", out if isinstance(out, list) else [])
    for step in steps:
        if isinstance(step, dict) and step.get("registry_name") == "device_processor":
            cfg = step.get("config") or {}
            cfg["device"] = "cpu"
            step["config"] = cfg
    return out


def standard_load_proof(ckpt_path, declared_image_features,
                        timeout_s: int = 600) -> tuple:
    """(ok, detail): construct ckpt's packaged processors in a fresh
    CPU-only process and verify a dummy batch keyed by the
    checkpoint's declared image features (the eval harness's own
    contract) runs through with every declared feature produced."""
    import os
    import subprocess
    import tempfile

    ckpt = Path(ckpt_path)
    if not (ckpt / "policy_preprocessor.json").exists():
        return False, f"MISSING: {ckpt / 'policy_preprocessor.json'}"
    with tempfile.TemporaryDirectory(prefix="loadproof_") as td:
        tmp = Path(td) / "ckpt"
        tmp.mkdir()
        for entry in sorted(ckpt.iterdir()):
            os.symlink(entry, tmp / entry.name)
        pre_json = json.loads((ckpt / "policy_preprocessor.json").read_text())
        rewritten = rewrite_device_steps_cpu(pre_json)
        (tmp / "policy_preprocessor.json").unlink()
        (tmp / "policy_preprocessor.json").write_text(json.dumps(rewritten))
        probe = Path(td) / "probe.py"
        probe.write_text(_PROBE_SRC)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        try:
            proc = subprocess.run(
                [sys.executable, str(probe), str(tmp),
                 json.dumps(list(declared_image_features))],
                capture_output=True, text=True, timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            return False, f"load proof timed out after {timeout_s}s"
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if proc.returncode == 0 and lines:
            return True, f"processors construct; produced {lines[-1]}"
        if lines:
            # the probe's own verdict line ({"missing": [...]})
            return False, f"load proof failed: {lines[-1]}"
        tail = (proc.stderr or "").strip().splitlines()
        return False, "load proof failed: " + (
            tail[-1] if tail else f"exit {proc.returncode}")


def assert_gate_report_covers(report: dict, required_names) -> list:
    """Gap G-11 (rounds 10/20 prep G2): the canonical gate report is
    the artifact the eval harness reads before judging. Return the
    list of coverage problems (empty = the report is a PASS and
    covers every required checkpoint with a recorded config hash)."""
    problems = []
    if report.get("verdict") != "PASS":
        problems.append(f"gate verdict is {report.get('verdict')!r}, not PASS")
    covered = report.get("checkpoints") or {}
    for name in required_names:
        entry = covered.get(name)
        if entry is None:
            problems.append(f"{name}: not in gate report checkpoints")
        elif not entry.get("config_sha256"):
            problems.append(f"{name}: gate report records no config_sha256")
    return problems


def stale_active_records(records_dir, covered_names) -> list:
    """Gap G-6: trajectory records in the ACTIVE eval set whose
    checkpoint is not covered by the gate report (e.g. the stale
    two-view fl_r30 records that sat in the active set until the
    2026-11-22 cleanup moved them to superseded_20261122/). Record
    files are named <ckpt>__<task>__ep<N>.npz."""
    records = Path(records_dir)
    if not records.is_dir():
        return []
    covered = set(covered_names)
    stale = []
    for f in sorted(records.iterdir()):
        if not f.is_file() or "__" not in f.name:
            continue
        if f.name.startswith("summary__"):
            # per-checkpoint summaries are named summary__<ckpt>.json
            ckpt = f.name[len("summary__"):]
            if ckpt.endswith(".json"):
                ckpt = ckpt[:-len(".json")]
        else:
            ckpt = f.name.split("__", 1)[0]
        if ckpt not in covered:
            stale.append(f.name)
    return stale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True, help="the eval split.json")
    ap.add_argument("--ckpt", action="append", default=[],
                    help="NAME=PATH of a checkpoint in the comparison set; "
                         "repeatable, at least one required")
    ap.add_argument("--out", required=True, help="eval gate report JSON path")
    ap.add_argument("--waive", action="append", default=[],
                    help="CHECK_ID=WAIVER_ID for a registered waiver")
    args = ap.parse_args()

    if not args.ckpt:
        print("ERROR: at least one --ckpt NAME=PATH is required")
        return 2
    ckpts = dict(kv.split("=", 1) for kv in args.ckpt)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    waivers = dict(kv.split("=", 1) for kv in args.waive)
    registered = {w["id"] for w in manifest.get("registered_waivers", [])}
    bad = {k: v for k, v in waivers.items() if v not in registered}
    if bad:
        print(f"ERROR: --waive cites waiver ids not registered in the "
              f"manifest: {bad}. Registered: {sorted(registered)}")
        return 2

    gate = Gate(manifest, waivers)
    exp = manifest["expected_policy_config"]
    rename_map = exp["rename_map"]
    expected_feats = norm_features(exp["input_features"], {})
    expected_out = norm_features(exp["output_features"], {})
    eg = manifest["eval_gate"]["split"]

    # ---- frozen split ----------------------------------------------------
    split_path = Path(args.split)
    split = json.loads(split_path.read_text())
    rows = canonical_split_rows(split)
    rows_sha = sha256_bytes(json.dumps(rows).encode())
    gate.record("eval.split.episodes", "eval", "split_episode_list",
                rows_sha == eg["canonical_rows_sha256"]
                and len(rows) == int(eg["n_episodes"]),
                f"{eg['n_episodes']} episodes, canonical sha256 "
                f"{eg['canonical_rows_sha256'][:12]}…",
                f"{len(rows)} episodes, canonical sha256 {rows_sha[:12]}…")
    for field, key in (("seed", "seed"), ("per_task", "per_task"),
                       ("disjoint_from_training", "disjoint_from_training"),
                       ("training_index_count", "training_index_count")):
        gate.record(f"eval.split.{key}", "eval", f"split[{key}]",
                    split.get(key) == eg[key], eg[key], split.get(key))
    per_task_counts = {t: len(split["tasks"][t]["episodes"]) for t in split["tasks"]}
    gate.record("eval.split.per_task_counts", "eval", "split_per_task_counts",
                all(v == int(eg["per_task"]) for v in per_task_counts.values())
                and len(per_task_counts) == 5,
                f"5 tasks x {eg['per_task']}", per_task_counts)

    # ---- checkpoints: identical declared inputs, shape-exact -------------
    declared: dict[str, dict] = {}
    ckpt_report: dict[str, dict] = {}
    for name, path in sorted(ckpts.items()):
        p = Path(path)
        cfg_path = p / "config.json"
        entry = {"path": str(p)}
        if not cfg_path.exists():
            gate.record(f"eval.ckpt.{name}.config", name, "config.json",
                        False, "present", f"MISSING: {cfg_path}")
            ckpt_report[name] = entry
            continue
        entry["config_sha256"] = sha256_file(cfg_path)
        cfg = json.loads(cfg_path.read_text())
        feats = norm_features(features_of_config(cfg), rename_map)
        declared[name] = feats
        gate.record(f"eval.ckpt.{name}.input_features", name,
                    "config.input_features",
                    exact_features_match(feats, expected_feats),
                    expected_feats, feats,
                    "D9: a checkpoint declaring different inputs than the "
                    "registered design must never be silently evaluated "
                    "against the other arms")
        out_feats = norm_features(
            {k: (v["shape"] if isinstance(v, dict) else v)
             for k, v in cfg.get("output_features", {}).items()}, {})
        gate.record(f"eval.ckpt.{name}.output_features", name,
                    "config.output_features",
                    exact_features_match(out_feats, expected_out), expected_out,
                    out_feats)
        pre = p / "policy_preprocessor.json"
        gate.record(f"eval.ckpt.{name}.preprocessor", name,
                    "policy_preprocessor.json",
                    pre.exists(), "packaged preprocessor present",
                    "present" if pre.exists() else f"MISSING: {pre}")
        # G-12: existence is not admission - the packaged processors
        # must construct and produce every image feature the
        # checkpoint's OWN config declares, in the config's own
        # naming. (The base snapshot declares the dataset-native
        # front/top/wrist and its processors are the identity for
        # them; trained checkpoints declare camera1/2/3, so their
        # packaged pipeline must perform the rename - the defect-A5
        # discrimination.)
        raw_feats = features_of_config(cfg)
        declared_images = sorted(
            k for k in raw_feats if k.startswith("observation.images."))
        proof_ok, proof_detail = standard_load_proof(
            p, declared_images)
        gate.record(f"eval.ckpt.{name}.standard_load", name,
                    "standard_load_proof",
                    proof_ok,
                    "packaged processors construct (CPU, fresh process) "
                    "and run a declared-keyed batch with all declared "
                    "image features produced",
                    proof_detail,
                    "G-12: an unpackaged or internally inconsistent "
                    "checkpoint (defect A5 class) must fail admission "
                    "here, not at eval load time")
        ckpt_report[name] = entry

    # ---- active-set hygiene (G-6) ---------------------------------------
    # Stale records from superseded checkpoint variants (e.g. the
    # two-view fl_r30 records) must not sit in the active records
    # directory: downstream table builders read the active set.
    records_dir = split_path.parent.parent / "records"
    if records_dir.is_dir():
        stale = stale_active_records(records_dir, sorted(ckpts))
        gate.record("eval.active_records_hygiene", "eval",
                    "active_records_dir",
                    not stale,
                    "every active record belongs to a gate-covered "
                    "checkpoint",
                    f"stale records: {stale}" if stale else
                    "all active records covered",
                    "G-6: stale two-view records sat in the active set "
                    "after the 3-camera repair (gatefix re-audit "
                    "residual)")

    names = sorted(declared)
    if len(names) >= 2:
        first = declared[names[0]]
        differing = [n for n in names[1:] if declared[n] != first]
        gate.record("eval.identical_input_features", "eval",
                    "identical_declared_input_features_across_set",
                    not differing,
                    f"all checkpoints identical to {names[0]}: {first}",
                    f"differing checkpoints: {differing}" if differing
                    else f"all {len(names)} checkpoints identical")

    overrides = [f"--waive={kv}" for kv in args.waive]
    return cec.print_and_write_report(gate, Path(args.out), {
        "gate": "P1-eval-gate",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_version": manifest.get("manifest_version"),
        "tree_hash": cec.tree_hash(cec.REPO),
        "invocation": sys.argv,
        "overrides": overrides,
        "split": str(split_path),
        "split_sha256": sha256_file(split_path),
        "checkpoints": ckpt_report,
    })


if __name__ == "__main__":
    raise SystemExit(main())
