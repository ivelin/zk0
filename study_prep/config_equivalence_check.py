#!/usr/bin/env python3
"""config_equivalence_check.py — Study 001 P1 config-equivalence gate.

Implements Prevention §P1 of the Study 001 config postmortem
(2026-11-22): before any comparative training run launches, dump the
RESOLVED configuration from the actual constructed objects — client
policy, server template model, and the first saved checkpoint's
config.json — and diff every field against the human-approved
registered design manifest (fl_rerun_study_manifest.json). A
difference with a registered waiver is DIFF-WAIVED; a difference
without one is DIFF-FAIL and the gate exits non-zero.

v2 (2026-11-22, gate-fix build closing audit findings F1–F11):
- every expected value is read from the manifest — no hardcoded
  literals for rounds / local steps / LR / census (F4);
- the full budget (steps x batch x views) is computed per arm from
  manifest inputs and compared to the registered budget (F4);
- optimizer + scheduler semantics are compared: the FL app config's
  scheduler block and the LeRobot arms' preset fields (F5);
- saved-checkpoint declarations are compared SHAPE-EXACT, and
  constructed objects must declare either the registered policy shape
  or the dataset-native shape per camera — right names at the wrong
  resolution now FAIL (F6);
- the gate report records the full invocation and every override
  used; the launcher refuses a PASS produced with configuration
  overrides (F7);
- --waive ids are validated against the manifest's registered
  waivers; an unregistered waiver id is an error (F10);
- --arm-config extends coverage to the LeRobot arms (specialist,
  specialist_exp, centralized, sequential): their resolved configs
  (the effective_config_*.json the training harness itself writes)
  are dumped and diffed against the manifest's arms_machine entries
  (F3);
- the census check includes the trainable tensor NAME-SET hash, not
  just counts (F9, trainable side; a post-training moved census is a
  documented known limit);
- per-task frame counts and output_features are checked (F11).

Usage (from the repo root, study venv):
  venv/bin/python study_prep/config_equivalence_check.py \
      --manifest study_prep/fl_rerun_study_manifest.json \
      --out /path/to/gate_report.json [--construct] \
      [--arm-config specialist=block_stack=/path/effective_config.json ...] \
      [--server-dataset armnet/armnetbench_v01_lerobot_so101 \
       --server-root /path/to/dataset/root] \
      [--checkpoint-config /path/to/checkpoint/config.json] \
      [--waive CHECK_ID=WAIVER_ID ...]

Exit codes: 0 = PASS (waivers allowed), 1 = DIFF-FAIL, 2 = error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_P0_MANIFEST = Path(
    "/home/openclaw/muse-work/zk0-study001-20261119/manifest.json"
)
DEFAULT_CHECKPOINT_CONFIG = (
    REPO
    / "outputs/2026-11-21_14-56-47/models/checkpoint_round_1/config.json"
)
POOLED_PREPROCESSOR_DIR = REPO / "study_assets/pooled_preprocessor"
BASE_SNAPSHOT = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"


# ---------------------------------------------------------------- util


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def tree_hash(repo: Path) -> str:
    """Hash covering the code/config tree the gate report applies to.

    Defined identically in launch_dgx_job.sh study mode:
    sha256( <git HEAD> + "\\n" + <git status --porcelain> ).
    Any change to tracked or untracked files in the repo changes it,
    so a stale PASS cannot cover a changed tree.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    return sha256_bytes(
        (git("rev-parse", "HEAD").strip() + "\n" + git("status", "--porcelain")).encode()
    )


def norm_features(feats: dict, rename_map: dict) -> dict:
    """{name: shape} with dataset camera names renamed via the manifest
    rename map, so client/server/checkpoint declarations are compared in
    the registered (post-rename) namespace."""

    out = {}
    for name, v in feats.items():
        shape = v["shape"] if isinstance(v, dict) else v
        out[rename_map.get(name, name)] = [int(x) for x in shape]
    return dict(sorted(out.items()))


def features_of_config(cfg: dict) -> dict:
    return {
        k: (v["shape"] if isinstance(v, dict) else v)
        for k, v in cfg.get("input_features", {}).items()
    }


def names_match(actual_norm: dict, expected_norm: dict) -> bool:
    """Feature-SET equivalence: same declared inputs after rename, with
    the state shape exact. Kept for the unit tests / v1 semantics."""

    return (
        sorted(actual_norm.keys()) == sorted(expected_norm.keys())
        and actual_norm.get("observation.state") == expected_norm.get("observation.state")
    )


def exact_features_match(actual_norm: dict, expected_norm: dict) -> bool:
    """Shape-exact equality — the rule for artifacts of record (saved
    checkpoint config.json) after F6."""

    return dict(actual_norm) == dict(expected_norm)


def constructed_features_match(actual_norm: dict, expected_norm: dict,
                               native_norm: dict) -> bool:
    """Rule for freshly CONSTRUCTED policies (F6): the declared input
    name set must equal the design's, the state shape must be exact,
    and every visual shape must be either the registered policy shape
    or the dataset-native shape for that camera. A declaration with
    the right names at an unregistered resolution (e.g. the leftover
    240x320 of Incident 1) FAILS."""

    if sorted(actual_norm.keys()) != sorted(expected_norm.keys()):
        return False
    for k, shape in actual_norm.items():
        if k == "observation.state":
            if shape != expected_norm[k]:
                return False
        elif shape not in (expected_norm[k], native_norm.get(k)):
            return False
    return True


def native_feature_shapes(manifest: dict) -> dict:
    """Dataset-native camera shapes in the post-rename namespace,
    channel-first: manifest dataset.cameras are HWC [H, W, C]; policy
    declarations are [C, H, W]."""

    rename = manifest["expected_policy_config"]["rename_map"]
    out = {}
    for ds_name, shape in manifest["dataset"]["cameras"].items():
        h, w, c = (int(x) for x in shape)
        out[rename.get(ds_name, ds_name)] = [c, h, w]
    out["observation.state"] = list(
        manifest["expected_policy_config"]["input_features"]["observation.state"]
    )
    return dict(sorted(out.items()))


def num_eq(a, b) -> bool:
    """Numeric-tolerant equality for config values that TOML/JSON may
    render as int or float (1 vs 1.0); exact for everything else."""

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def compute_budget(step_equivalents: int, batch: int, views: int) -> dict:
    samples = step_equivalents * batch
    return {
        "total_step_equivalents": step_equivalents,
        "total_samples": samples,
        "images_at_3_views": samples * views,
    }


# ---------------------------------------------------------------- checks


class Gate:
    def __init__(self, manifest: dict, waivers: dict):
        self.manifest = manifest
        self.waivers = waivers  # check_id -> waiver_id
        self.checks: list[dict] = []

    def record(self, check_id: str, arm: str, field: str, ok: bool,
               expected, actual, detail: str = ""):
        if ok:
            status = "PASS"
        elif check_id in self.waivers:
            status = "DIFF-WAIVED"
            detail = (detail + f" [waiver {self.waivers[check_id]}]").strip()
        else:
            status = "DIFF-FAIL"
        self.checks.append({
            "id": check_id, "arm": arm, "field": field, "status": status,
            "expected": expected, "actual": actual, "detail": detail,
        })
        return ok

    @property
    def verdict(self) -> str:
        return "FAIL" if any(c["status"] == "DIFF-FAIL" for c in self.checks) else "PASS"


def print_and_write_report(gate: Gate, out_path: Path, extra: dict) -> int:
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": gate.verdict,
        "checks": gate.checks,
        **extra,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for c in gate.checks:
        print(f"{c['status']:<12} {c['arm']:<12} {c['field']:<48} "
              f"expected={c['expected']} actual={c['actual']}"
              + (f" :: {c['detail']}" if c["detail"] else ""))
    print(f"GATE VERDICT: {gate.verdict} "
          f"({sum(1 for c in gate.checks if c['status'] == 'PASS')} PASS, "
          f"{sum(1 for c in gate.checks if c['status'] == 'DIFF-WAIVED')} DIFF-WAIVED, "
          f"{sum(1 for c in gate.checks if c['status'] == 'DIFF-FAIL')} DIFF-FAIL)")
    print(f"report: {out}  tree_hash={report.get('tree_hash')}")
    return 0 if gate.verdict == "PASS" else 1


# ---------------------------------------------------------------- sources


def load_pyproject() -> dict:
    import tomllib

    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def load_dataset_meta(name: str, root: str | None, episodes=None):
    from src.common.utils import load_lerobot_dataset

    ds = load_lerobot_dataset(name, root=root, episodes=episodes)
    return ds.meta


def construct_policy(dataset_meta):
    from src.training.model_utils import get_model

    return get_model(dataset_meta=dataset_meta)


def policy_dump(policy) -> dict:
    cfg = policy.config
    feats = {}
    for k, v in dict(cfg.input_features).items():
        if isinstance(v, dict):
            feats[k] = list(v["shape"])
        elif hasattr(v, "shape"):
            feats[k] = list(v.shape)
        else:
            feats[k] = list(v)
    total = list(policy.parameters())
    trainable = [p for p in total if p.requires_grad]
    trainable_names = sorted(
        k for k, p in policy.named_parameters() if p.requires_grad
    )
    all_names = sorted(k for k, _ in policy.named_parameters())
    return {
        "input_features": feats,
        "chunk_size": int(cfg.chunk_size),
        "total_tensors": len(total),
        "total_params": sum(p.numel() for p in total),
        "trainable_tensors": len(trainable),
        "trainable_params": sum(p.numel() for p in trainable),
        "trainable_name_set_sha256": sha256_bytes(
            json.dumps(trainable_names).encode()
        ),
        "total_name_set_sha256": sha256_bytes(json.dumps(all_names).encode()),
    }


# ---------------------------------------------------------------- LeRobot arms (F3)


def check_lerobot_arm(gate: Gate, manifest: dict, arm: str, task: str,
                      cfg_path: Path, expected_feats: dict,
                      budget_tasks_seen: dict):
    """Diff one LeRobot arm's resolved config (the effective_config_*.json
    the training harness wrote from the constructed TrainPipelineConfig)
    against the manifest's machine-readable arms_machine entry."""

    am = manifest["arms_machine"][arm]
    recipe = manifest["recipe"]
    opt = recipe["optimizer_struct"]
    sched = recipe["scheduler_lerobot"]
    label = f"{arm}.{task}" if task != "-" else arm
    cfg = json.loads(Path(cfg_path).read_text())
    pol = cfg["policy"]

    gate.record(f"arms.{label}.scope", arm, "train_expert_only/freeze_vision_encoder",
                pol.get("train_expert_only") == am["train_expert_only"]
                and pol.get("freeze_vision_encoder") == am["freeze_vision_encoder"],
                f"expert_only={am['train_expert_only']} freeze={am['freeze_vision_encoder']} "
                f"({am['scope']}; census {am['census']['trainable_tensors']} tensors / "
                f"{am['census']['trainable_params']} params, name-set "
                f"{am['census']['trainable_name_set_sha256'][:12]}…)",
                f"expert_only={pol.get('train_expert_only')} freeze={pol.get('freeze_vision_encoder')}")
    in_feats = norm_features(
        {k: v["shape"] for k, v in pol["input_features"].items()}, {})
    gate.record(f"arms.{label}.input_features", arm, "policy.input_features",
                exact_features_match(in_feats, expected_feats),
                expected_feats, in_feats)
    out_feats = norm_features(
        {k: v["shape"] for k, v in pol["output_features"].items()}, {})
    exp_out = norm_features(manifest["expected_policy_config"]["output_features"], {})
    gate.record(f"arms.{label}.output_features", arm, "policy.output_features",
                exact_features_match(out_feats, exp_out), exp_out, out_feats)
    gate.record(f"arms.{label}.chunk_size", arm, "policy.chunk_size",
                int(pol["chunk_size"]) == int(manifest["expected_policy_config"]["chunk_size"]),
                manifest["expected_policy_config"]["chunk_size"], pol["chunk_size"])
    gate.record(f"arms.{label}.batch_size", arm, "batch_size",
                int(cfg["batch_size"]) == int(am["batch_size"]),
                am["batch_size"], cfg["batch_size"])
    steps_expected = am.get("steps_per_task", am.get("steps"))
    gate.record(f"arms.{label}.steps", arm, "steps",
                int(cfg["steps"]) == int(steps_expected),
                steps_expected, cfg["steps"])
    gate.record(f"arms.{label}.seed", arm, "seed",
                int(cfg["seed"]) == int(am["seed"]), am["seed"], cfg["seed"])
    gate.record(f"arms.{label}.preset", arm, "use_policy_training_preset",
                cfg.get("use_policy_training_preset") == am["use_policy_training_preset"],
                am["use_policy_training_preset"], cfg.get("use_policy_training_preset"))
    gate.record(f"arms.{label}.rename_map", arm, "rename_map",
                cfg.get("rename_map") == manifest["expected_policy_config"]["rename_map"],
                manifest["expected_policy_config"]["rename_map"], cfg.get("rename_map"))
    eps = cfg["dataset"]["episodes"]
    eps_sha = sha256_bytes(json.dumps(eps).encode())
    sha_key = task if task != "-" else "union"
    gate.record(f"arms.{label}.episodes", arm, "dataset.episodes",
                eps_sha == am["episodes_sha256"][sha_key]
                and len(eps) == int(am["episode_counts"][sha_key]),
                f"{am['episode_counts'][sha_key]} episodes sha256 {am['episodes_sha256'][sha_key][:12]}…",
                f"{len(eps)} episodes sha256 {eps_sha[:12]}…")
    opt_actual = {
        "lr": pol.get("optimizer_lr"), "betas": pol.get("optimizer_betas"),
        "eps": pol.get("optimizer_eps"),
        "weight_decay": pol.get("optimizer_weight_decay"),
        "grad_clip_norm": pol.get("optimizer_grad_clip_norm"),
    }
    opt_expected = {k: opt[k] for k in
                    ("lr", "betas", "eps", "weight_decay", "grad_clip_norm")}
    gate.record(f"arms.{label}.optimizer", arm, "optimizer_preset(AdamW)",
                all(num_eq(opt_actual[k], opt_expected[k]) for k in opt_expected),
                opt_expected, opt_actual)
    sched_actual = {
        "warmup": pol.get("scheduler_warmup_steps"),
        "decay_steps": pol.get("scheduler_decay_steps"),
        "decay_lr": pol.get("scheduler_decay_lr"),
    }
    sched_expected = {
        "warmup": sched["warmup_steps"],
        "decay_steps": am["scheduler_decay_steps"],
        "decay_lr": sched["decay_lr"],
    }
    gate.record(f"arms.{label}.scheduler", arm, "scheduler_preset(cosine)",
                all(num_eq(sched_actual[k], sched_expected[k]) for k in sched_expected),
                sched_expected, sched_actual,
                "decay_steps == arm steps per manifest rule; the cross-arm "
                "horizon difference is waiver W-LR-horizon")
    gate.record(f"arms.{label}.push_to_hub", arm, "save_checkpoint_to_hub",
                cfg.get("save_checkpoint_to_hub") is False,
                False, cfg.get("save_checkpoint_to_hub"))

    if "steps_per_task" in am:
        budget_tasks_seen.setdefault(arm, {})[task] = int(cfg["steps"])
    else:
        budget = compute_budget(int(cfg["steps"]), int(am["batch_size"]),
                                int(recipe["budget_inputs"]["views"]))
        exp_b = recipe["budget"]
        gate.record(f"arms.{arm}.budget", arm, "budget(steps x batch x views)",
                    budget["total_step_equivalents"] == exp_b["total_step_equivalents"]
                    and budget["total_samples"] == exp_b["total_samples"]
                    and budget["images_at_3_views"] == exp_b["images_at_3_views"],
                    exp_b, budget)


def finish_per_task_budgets(gate: Gate, manifest: dict, budget_tasks_seen: dict):
    recipe = manifest["recipe"]
    exp_b = recipe["budget"]
    for arm, tasks in budget_tasks_seen.items():
        am = manifest["arms_machine"][arm]
        if set(tasks) != set(am["tasks"]):
            continue  # partial set: budget recorded only for a complete arm
        budget = compute_budget(sum(tasks.values()), int(am["batch_size"]),
                                int(recipe["budget_inputs"]["views"]))
        gate.record(f"arms.{arm}.budget", arm, "budget(steps x batch x views)",
                    budget["total_step_equivalents"] == exp_b["total_step_equivalents"]
                    and budget["total_samples"] == exp_b["total_samples"]
                    and budget["images_at_3_views"] == exp_b["images_at_3_views"],
                    exp_b, budget,
                    f"sum over {len(tasks)} task configs x {am['steps_per_task']} steps")


# ---------------------------------------------------------------- FL optimizer-state probe (F5)


def fl_optimizer_reset_probe() -> tuple[bool, str]:
    """W-D11-ii: FL AdamW state resets per round. Static code probe of
    the substrate: the optimizer is constructed fresh inside the
    per-fit training setup (make_optimizer_and_scheduler in
    src/training/training_setup.py) and no code path loads optimizer
    state from a previous round. Fail closed if either fact changes."""

    setup = (REPO / "src/training/training_setup.py").read_text()
    fresh = "make_optimizer_and_scheduler" in setup
    loads = []
    for p in (REPO / "src").rglob("*.py"):
        t = p.read_text(errors="ignore")
        if "optimizer.load_state_dict" in t or "optim.load_state_dict" in t:
            loads.append(str(p.relative_to(REPO)))
    return (fresh and not loads), (
        f"fresh-construction={'found' if fresh else 'MISSING'} in "
        f"src/training/training_setup.py; optimizer-state loads: {loads or 'none'}"
    )


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="gate report JSON path")
    ap.add_argument("--p0-manifest", default=str(DEFAULT_P0_MANIFEST))
    ap.add_argument("--checkpoint-config", default=str(DEFAULT_CHECKPOINT_CONFIG),
                    help="first saved checkpoint's config.json to verify")
    ap.add_argument("--server-dataset", default=None,
                    help="override the pyproject server dataset entry "
                         "(the corrected miniature configuration)")
    ap.add_argument("--server-root", default=None,
                    help="on-disk root for the overridden server dataset")
    ap.add_argument("--construct", action="store_true",
                    help="construct client + server template policies and "
                         "dump their resolved configs (CPU; minutes)")
    ap.add_argument("--save-template-config", default=None,
                    help="write the constructed server template's config to "
                         "this path and verify it as the saved checkpoint")
    ap.add_argument("--arm-config", action="append", default=[],
                    help="ARM=TASK=PATH to a LeRobot arm's resolved "
                         "effective_config_*.json (TASK='-' for centralized); "
                         "repeatable (F3)")
    ap.add_argument("--waive", action="append", default=[],
                    help="CHECK_ID=WAIVER_ID for a registered waiver")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    waivers = dict(kv.split("=", 1) for kv in args.waive)
    registered = {w["id"] for w in manifest.get("registered_waivers", [])}
    bad = {k: v for k, v in waivers.items() if v not in registered}
    if bad:
        print(f"ERROR: --waive cites waiver ids not registered in the "
              f"manifest: {bad}. Registered: {sorted(registered)}")
        return 2

    overrides = []
    if args.server_dataset:
        overrides.append(f"--server-dataset={args.server_dataset}")
    if args.server_root:
        overrides.append(f"--server-root={args.server_root}")
    if args.checkpoint_config != str(DEFAULT_CHECKPOINT_CONFIG):
        overrides.append(f"--checkpoint-config={args.checkpoint_config}")
    if args.save_template_config:
        overrides.append(f"--save-template-config={args.save_template_config}")
    overrides += [f"--waive={kv}" for kv in args.waive]

    gate = Gate(manifest, waivers)
    exp = manifest["expected_policy_config"]
    rename_map = exp["rename_map"]
    expected_feats = norm_features(exp["input_features"], {})
    native_feats = native_feature_shapes(manifest)
    recipe = manifest["recipe"]
    fl_am = manifest["arms_machine"]["fl_rerun"]
    fl_app_exp = fl_am["app_config_expected"]
    py = load_pyproject()
    datasets_cfg = py["tool"]["zk0"]["datasets"]
    app_cfg = py["tool"]["flwr"]["app"]["config"]

    # ---- episode lists + frame counts (P0) ------------------------------
    from src.client_app import _load_episode_list

    p0 = json.loads(Path(args.p0_manifest).read_text())
    union: set[int] = set()
    for client in sorted(datasets_cfg["clients"], key=lambda c: c["client_id"]):
        task = p0["fl_node_assignment"][f"node_{client['client_id']}"]["task_id"]
        eps_list = _load_episode_list(client["episodes_file"])
        expected_eps = p0["subsets"][task]["episode_indices"]
        gate.record(
            f"episodes.{task}", "fl", "episode_list",
            list(eps_list) == list(expected_eps),
            f"{len(expected_eps)} episodes, P0 list sha256 "
            f"{sha256_bytes(json.dumps(expected_eps).encode())[:12]}…",
            f"{len(eps_list)} episodes, file sha256 "
            f"{sha256_bytes(json.dumps(eps_list).encode())[:12]}…",
        )
        gate.record(
            f"episodes.{task}.frames", "fl", "episode_frame_count",
            int(p0["subsets"][task]["total_frames"])
            == int(manifest["arms_machine"]["specialist"]["frame_counts"][task]),
            manifest["arms_machine"]["specialist"]["frame_counts"][task],
            p0["subsets"][task]["total_frames"],
        )
        union.update(int(e) for e in eps_list)
    gate.record("episodes.union", "fl", "episode_union_count",
                len(union) == int(manifest["dataset"]["union_episodes"]),
                manifest["dataset"]["union_episodes"], len(union))

    # ---- preprocessor hashes ------------------------------------------
    pre_json = POOLED_PREPROCESSOR_DIR / "policy_preprocessor.json"
    gate.record("preprocessor.json_sha256", "fl", "preprocessor_json",
                sha256_file(pre_json) == manifest["expected_preprocessor"]["policy_preprocessor_json_sha256"],
                manifest["expected_preprocessor"]["policy_preprocessor_json_sha256"],
                sha256_file(pre_json))
    norm_file = POOLED_PREPROCESSOR_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    gate.record("preprocessor.normalizer_sha256", "fl", "normalizer_stats",
                sha256_file(norm_file).startswith(
                    manifest["expected_preprocessor"]["normalizer_safetensors_sha256_prefix"]),
                manifest["expected_preprocessor"]["normalizer_safetensors_sha256_prefix"] + "…",
                sha256_file(norm_file)[:16] + "…")

    # ---- FL resolved app config: recipe + optimizer/schedule semantics --
    for key, exp_val in fl_app_exp.items():
        gate.record(f"recipe.app.{key}", "fl", f"app_config[{key}]",
                    key in app_cfg and num_eq(app_cfg[key], exp_val),
                    exp_val, app_cfg.get(key, "MISSING"))
    push_keys = [k for k in app_cfg if "hf" in k.lower() or "hub" in k.lower()]
    gate.record("recipe.push_to_hub", "fl", "push_to_hub_absent",
                recipe["push_to_hub"] is False and not push_keys,
                "no hf_repo_id / push key in resolved config (P3)",
                f"push keys present: {push_keys}" if push_keys else "none")
    probe_ok, probe_detail = fl_optimizer_reset_probe()
    gate.record("fl.optimizer_state_policy", "fl", "optimizer_state_policy",
                probe_ok, fl_am["optimizer_state_policy"], probe_detail)

    # ---- FL budget (F4): rounds x local steps x clients x batch x views -
    fl_steps = (int(fl_app_exp["num-server-rounds"])
                * int(fl_app_exp["local-epochs"]) * int(fl_am["clients"]))
    fl_budget = compute_budget(fl_steps, int(fl_am["batch_size"]),
                               int(recipe["budget_inputs"]["views"]))
    exp_b = recipe["budget"]
    gate.record("fl.budget", "fl", "budget(steps x batch x views)",
                fl_budget["total_step_equivalents"] == exp_b["total_step_equivalents"]
                and fl_budget["total_samples"] == exp_b["total_samples"]
                and fl_budget["images_at_3_views"] == exp_b["images_at_3_views"],
                exp_b, fl_budget,
                f"{fl_app_exp['num-server-rounds']} rounds x "
                f"{fl_app_exp['local-epochs']} local steps x {fl_am['clients']} clients")

    # ---- client policy: starting-checkpoint declaration ----------------
    snap_cfg = (Path.home() / ".cache/huggingface/hub/models--lerobot--smolvla_base"
                / "snapshots" / BASE_SNAPSHOT / "config.json")
    base_cfg = json.loads(snap_cfg.read_text())
    gate.record("client.input_features", "fl-client", "input_features(base config)",
                norm_features(features_of_config(base_cfg), {}) == expected_feats,
                expected_feats, norm_features(features_of_config(base_cfg), {}))

    # ---- server dataset identity ---------------------------------------
    # Gap G-5 closure (2026-11-22): the substrate no longer builds
    # anything from the server entry - the template derives from the
    # first client's dataset (load_template_dataset_meta), and no
    # server entry is configured by default. A configured entry is
    # eval-only, so the identity rule is: absent, or the study
    # dataset. A leftover non-study entry is an armed eval landmine
    # (and was Incident 1's root cause when the template read it).
    _server_list = datasets_cfg.get("server") or []
    server_entry = args.server_dataset or (
        _server_list[0]["name"] if _server_list else None)
    study_dataset = manifest["dataset"]["repo_id"]
    gate.record("server_dataset.identity", "fl-server", "server_dataset",
                server_entry is None or server_entry == study_dataset,
                f"{study_dataset} (manifest: server_dataset_must_be the study "
                "dataset or none)",
                server_entry if server_entry is not None else "(none configured)",
                "server eval datasets are evaluation-only since the "
                "G-5 fix; a leftover non-study entry here would "
                "evaluate the global model on non-study data if "
                "server eval were enabled")

    # ---- saved checkpoint declaration (SHAPE-EXACT after F6) ------------
    ckpt_cfg_path = Path(args.checkpoint_config)
    _is_template_sim = bool(args.save_template_config) and \
        ckpt_cfg_path == Path(args.save_template_config)
    if ckpt_cfg_path.exists():
        ckpt_cfg = json.loads(ckpt_cfg_path.read_text())
        _actual = norm_features(features_of_config(ckpt_cfg), rename_map)
        # The artifact of record (a real saved checkpoint) is shape-EXACT
        # (F6). The miniature template-save simulation is a construction
        # artifact — a raw template save carries dataset-native shapes —
        # so it is judged by the constructed-object rule.
        _ok = (constructed_features_match(_actual, expected_feats, native_feats)
               if _is_template_sim else
               exact_features_match(_actual, expected_feats))
        gate.record("saved_checkpoint.input_features", "fl-artifact",
                    "saved_checkpoint_config.input_features",
                    _ok,
                    expected_feats, _actual,
                    f"from {ckpt_cfg_path} "
                    f"({'constructed rule: template-save simulation' if _is_template_sim else 'shape-exact: artifact of record'})")
        _out = norm_features(
            {k: (v["shape"] if isinstance(v, dict) else v)
             for k, v in ckpt_cfg.get("output_features", {}).items()}, {})
        _exp_out = norm_features(exp["output_features"], {})
        gate.record("saved_checkpoint.output_features", "fl-artifact",
                    "saved_checkpoint_config.output_features",
                    exact_features_match(_out, _exp_out), _exp_out, _out)
    else:
        gate.record("saved_checkpoint.input_features", "fl-artifact",
                    "saved_checkpoint_config.input_features", False,
                    expected_feats, f"MISSING: {ckpt_cfg_path}")

    # ---- constructed objects (client + server template) -----------------
    if args.construct:
        c0 = sorted(datasets_cfg["clients"], key=lambda c: c["client_id"])[0]
        eps0 = _load_episode_list(c0["episodes_file"])
        client_policy = construct_policy(load_dataset_meta(c0["name"], c0.get("root"), eps0))
        cdump = policy_dump(client_policy)
        gate.record("client.constructed_input_features", "fl-client",
                    "constructed_policy.input_features",
                    constructed_features_match(
                        norm_features(cdump["input_features"], rename_map),
                        expected_feats, native_feats),
                    expected_feats,
                    norm_features(cdump["input_features"], rename_map),
                    "constructed-object rule (F6): names == design; each visual "
                    "shape is the registered policy shape or the dataset-native shape")
        cen = manifest["census"]["expert_only"]
        tot = manifest["census"]["total"]
        gate.record("census.total", "fl-client", "tensor_census_total",
                    cdump["total_tensors"] == tot["tensors"]
                    and cdump["total_params"] == tot["params"]
                    and cdump["total_name_set_sha256"] == tot["name_set_sha256"],
                    f"{tot['tensors']} tensors / {tot['params']} params / name-set "
                    f"{tot['name_set_sha256'][:12]}…",
                    f"{cdump['total_tensors']} tensors / {cdump['total_params']} params / "
                    f"name-set {cdump['total_name_set_sha256'][:12]}…")
        gate.record("census.trainable", "fl-client", "tensor_census_trainable",
                    cdump["trainable_tensors"] == cen["trainable_tensors"]
                    and cdump["trainable_params"] == cen["trainable_params"]
                    and cdump["trainable_name_set_sha256"] == cen["trainable_name_set_sha256"],
                    f"{cen['trainable_tensors']} tensors / {cen['trainable_params']} params / "
                    f"name-set {cen['trainable_name_set_sha256'][:12]}… (expert-only scope)",
                    f"{cdump['trainable_tensors']} tensors / {cdump['trainable_params']} params / "
                    f"name-set {cdump['trainable_name_set_sha256'][:12]}…")
        gate.record("client.chunk_size", "fl-client", "chunk_size",
                    cdump["chunk_size"] == int(exp["chunk_size"]),
                    exp["chunk_size"], cdump["chunk_size"])
        del client_policy

        # The substrate builds its template from the FIRST CLIENT's
        # dataset metadata (load_template_dataset_meta, gap G-5 fix);
        # construct exactly the object the substrate will build.
        template = construct_policy(
            load_dataset_meta(c0["name"], c0.get("root"), eps0))
        tdump = policy_dump(template)
        gate.record(
            "server_template.input_features", "fl-server",
            "template_model.input_features",
            constructed_features_match(
                norm_features(tdump["input_features"], rename_map),
                expected_feats, native_feats),
            expected_feats, norm_features(tdump["input_features"], rename_map),
            "P1 HARD GATE: the server template's declared features become "
            "every saved checkpoint's declared features (Incident 1)")
        if args.save_template_config:
            outp = Path(args.save_template_config)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps({
                "input_features": {
                    k: {"shape": v} for k, v in tdump["input_features"].items()
                },
                "output_features": {"action": {"shape": [6]}},
                "chunk_size": tdump["chunk_size"],
            }, indent=2))
            saved = json.loads(outp.read_text())
            _sactual = norm_features(features_of_config(saved), rename_map)
            gate.record("saved_checkpoint.input_features", "fl-artifact",
                        "template_saved_config.input_features",
                        constructed_features_match(_sactual, expected_feats, native_feats),
                        expected_feats, _sactual,
                        f"template config as save_model_checkpoint would write it -> {outp} "
                        "(constructed rule: a raw template save carries dataset-native "
                        "shapes; the packaged artifact of record is shape-exact-checked "
                        "via saved_checkpoint.input_features)")
        del template

    # ---- LeRobot arms (F3) ----------------------------------------------
    budget_tasks_seen: dict = {}
    for spec in args.arm_config:
        arm, task, path = spec.split("=", 2)
        check_lerobot_arm(gate, manifest, arm, task, Path(path),
                          expected_feats, budget_tasks_seen)
    finish_per_task_budgets(gate, manifest, budget_tasks_seen)

    # ---- report ----------------------------------------------------------
    return print_and_write_report(gate, Path(args.out), {
        "gate": "P1-config-equivalence",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_version": manifest.get("manifest_version"),
        "tree_hash": tree_hash(REPO),
        "invocation": sys.argv,
        "overrides": overrides,
    })


if __name__ == "__main__":
    raise SystemExit(main())
