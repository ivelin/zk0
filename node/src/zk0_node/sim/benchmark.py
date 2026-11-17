"""SIM-1 benchmark runner — sim mode end-to-end (spec §3, Stage 5 shape).

One command runs the SIM-1 benchmark task scene (cube → box, the
certified T1 green-box configuration) on the certified Stack A
harness and emits the spec's artifacts: a **state log** and an
**eval record**. The verdict is computed from the state log and the
gate rows — never from a render or a built-in success flag
(SSOT decision 2).

The policy under test in build slice 1 is the **scripted reference
pilot** (the certified T1 plan): generation mark ORACLE-GUIDED,
ablation rung 0. It is not the shared checkpoint and not a
contract-conforming controller; the eval record states both facts
in plain fields. The checkpoint's own rollout rate remains
unmeasured (the preregistration's honest gap) until the Stage 5
suite machinery lands.

Tier 1 (preregistration §2) = object in target + nothing damaged +
legal actuation, where "nothing damaged" is max single contact
force < 25 N (the PROVISIONAL ceramic-chip low end). Tier 2 is the
grasp-quality gate table, PROVISIONAL per SSOT decision 53.
"""
import os
import shutil
import sys
from pathlib import Path

from ..config import load_config, zk0_home
from ..records import (jsonable, manifest_sha256, runtime_manifest_path,
                       utcnow_iso, write_record)
from . import assets

HARNESS_DIR = Path(__file__).parent / "harness"
DAMAGE_BOUND_N = 25.0  # PROVISIONAL — preregistration §2


def run_sim1(variant: str = "realistic", outdir=None, cfg=None) -> dict:
    cfg = cfg or load_config()
    staged = assets.model_dir(variant)
    if not staged.exists():
        raise FileNotFoundError(
            f"staged model dir missing: {staged} — run dev/bootstrap.sh")
    info = assets.arm_model_info(variant)

    stamp = utcnow_iso().replace(":", "").replace("-", "")
    outdir = Path(outdir) if outdir else (
        zk0_home() / "records" / f"sim1-{variant}-{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Stage a private copy of the model dir: the Pilot writes its
    # scene file into the model dir, and the asset cache stays clean.
    work_model = outdir / "model"
    shutil.copytree(staged, work_model)
    os.environ["ZK0_SIM_MODEL_DIR"] = str(work_model)
    if str(HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(HARNESS_DIR))

    import multitask as MT
    from grasp_gate import evaluate_cycle

    P, log, _snaps, _frames, res = MT.task1({})
    body = "cube_green"
    t_close, t_open = P.events["close_end_t1"], P.events["open_end_t1"]
    gate = evaluate_cycle(P.gate.rows, body, t_close, t_open,
                          MT.GRIP_CLOSE_CUBE, P.gate.mass[body],
                          P.gate.jnt_range, P.gate.force_limit)
    rows = P.gate.rows
    max_single_run = max(r[body]["max_single"] for r in rows)
    window = [r for r in rows if t_close <= r["t"] <= t_open]
    max_single_window = max(r[body]["max_single"] for r in window)

    containment = bool(res["ok"])
    damage_pass = bool(max_single_window < DAMAGE_BOUND_N)
    legal_pass = bool(gate["gates"]["command"]["pass"])
    tier1_pass = containment and damage_pass and legal_pass

    state_log = {
        "log_type": "sim_state_log/0.1",
        "task": "SIM-1",
        "arm_model_variant": variant,
        "events": P.events,
        "t": log["t"], "tcp": log["tcp"], "q": log["q"],
        "bodies": log["bodies"], "quats": log["quats"],
        "gate_rows": rows,
        "ik_failures": [e for e in P.ik_log if not e["ok"]],
    }
    write_record(state_log, outdir / "state_log.json")

    record = {
        "record_type": "eval_record",
        "schema": "zk0-node-eval/0.1",
        "created_utc": utcnow_iso(),
        "node": {"node_id": cfg["node_id"], "package": "zk0 Node",
                 "package_version": "0.1.0",
                 "runtime_manifest_sha256": manifest_sha256(
                     runtime_manifest_path())},
        "mode": "sim",
        "task": {
            "id": "SIM-1",
            "name": "Cube → box (green-box family)",
            "suite": "vendored multitask.task1 — the certified T1 scene "
                     "and plan (SSOT decisions 24–27)",
            "trials": 1,
            "trial_note": "single reference trial in build slice 1; the "
                          "≥20-trial suite machinery is Stage 5 build "
                          "work — no success rate is claimed from n=1",
        },
        "policy": {
            "kind": "scripted_pilot",
            "id": "multitask.task1 plan (certified T1 configuration)",
            "generation_mark": "ORACLE-GUIDED",
            "note": "scripted reference pilot, NOT the shared checkpoint "
                    "(ivelin/zk0-smolvla-fl); the checkpoint's rollout "
                    "rate on this task remains unmeasured",
        },
        "arm_model": {
            "name": info["name"], "version": info["version"],
            "canonical_file": info["canonical_file"],
            "canonical_md5": info.get("canonical_md5"),
            "servo_profile": "variant (the MJCF's position actuators; "
                             "the BAM fitted-servo profile is not engaged "
                             "by this harness — build slice 2)",
        },
        "engine": {"name": "MuJoCo", "version": _mujoco_version()},
        "ablation_rung": 0,
        "rung_note": "the scripted oracle pilot consumes ground-truth "
                     "state; rung 0 is a diagnostic reference run, not a "
                     "telemetry-contract conformance claim (contract §6)",
        "marks": {
            "provenance": "sim-only reference run — no model, no "
                          "contribution; no provenance mark is earned",
            "generation": "ORACLE-GUIDED",
            "envelope": "n/a (trajectory admission is Stage 7 work)",
        },
        "results": {
            "original_scorer": jsonable(res),
            "tier1": {
                "pass": tier1_pass,
                "containment": containment,
                "containment_detail": {
                    "inside_xy": res["inside_xy"],
                    "below_rim": res["below_rim"],
                    "settled_mm": res["settled_mm"],
                    "arm_retreated": res["arm_retreated"],
                    "final_cube": res["final_cube"],
                },
                "damage": {
                    "max_single_contact_N_carry_window": round(
                        float(max_single_window), 2),
                    "max_single_contact_N_whole_run": round(
                        float(max_single_run), 2),
                    "bound_N": DAMAGE_BOUND_N,
                    "bound_status": "PROVISIONAL",
                    "pass": damage_pass,
                },
                "legal_actuation": {
                    "pass": legal_pass,
                    "command_gate": jsonable(gate["gates"]["command"]),
                },
            },
            "tier2": {
                "status": "PROVISIONAL (SSOT decision 53 — not "
                          "human-validated; does not gate anything)",
                "gate_pass": bool(gate["pass"]),
                "failed_gates": [k for k, v in gate["gates"].items()
                                 if not v["pass"]],
                "gates": jsonable(gate["gates"]),
                "window_s": gate["window_s"],
                "n_airborne": gate.get("n_airborne"),
            },
        },
        "state_log": "state_log.json (same directory)",
        "signature": "content hash only in build slice 1; node-key "
                     "signing lands with the Stage 1 identity work",
    }
    write_record(record, outdir / "eval_record.json")
    record["outdir"] = str(outdir)
    return record


def _mujoco_version():
    import mujoco
    return mujoco.__version__
