"""Stage 3 — THE M0 JOIN GATE (spec section 3). Protocol-blocking.

No contribution from this node is accepted — by the local runtime or
by any aggregator — until this gate passes. It is implemented as
code, not policy text: `zk0 fl join` (and later `zk0 contribute`)
call `require_current_attestation()` and are refused without one.

The four checks (spec section 3, real-arm track wording; this build
runs the sim track of decision 4 on the package's sim arm — the
checks are the same, the arm under test is the sim arm):

1. Calibration standardization — the node's residual offsets vs the
   canonical frame are within tolerance, algebraically AND on the
   (sim) arm: the standard probe trajectory is driven through the
   node's own pipeline and commanded vs achieved poses are compared
   from the run log.
2. Action normalization — the declared normalization is the
   protocol's (identity at the boundary, or the shared checkpoint
   statistics — hash-matched), and it round-trips the probe.
3. Calibration fingerprint current — the stored fingerprint matches
   the calibration on disk and the arm model it was taken against.
4. Local eval works — the node runs the Stage 5 quick suite
   (SIM-1, 2 trials, the shared checkpoint as policy) and produces
   a well-formed, signature-valid eval record.

Passing mints a signed join-gate attestation (7-day expiry).
Failing any check refuses with the check named — in the CLI output
and in a signed gate-attempt record, so a refusal is itself
evidence.

FOUNDER OVERRIDE: a node whose identity_kind is "founder" may pass
a refused join with --founder-override "<reason>". The override is
recorded in a signed override record naming every check it bypassed
and the reason verbatim. No other identity can override, and the
override never edits the gate's verdict — it is an admitted,
attributable exception, not a quieter gate.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import calibration as C
from .config import load_config, save_config, zk0_home
from .identity import load_or_create_key
from .records import (manifest_sha256, runtime_manifest_path, utcnow_iso,
                      verify_record, write_signed_record)
from .sim import assets

ATTESTATION_TTL_DAYS = 7


class GateRefused(Exception):
    """Join refused by the M0 gate. Carries the failed check names."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("M0 join gate refused: " + "; ".join(self.reasons))


def attestations_dir() -> Path:
    return zk0_home() / "records" / "attestations"


def _sim_probe(calib: dict) -> dict:
    """Drive the standard probe through the node's pipeline on the
    sim arm and compare commanded vs achieved poses from the log."""
    import mujoco

    if not assets.model_dir("realistic").exists():
        raise FileNotFoundError("sim assets not staged — dev/bootstrap.sh")
    import shutil
    work = zk0_home() / "gate-probe-model"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(assets.model_dir("realistic"), work)
    m = mujoco.MjModel.from_xml_path(str(work / "so101.xml"))
    m.opt.timestep = 0.002
    qadr = [m.jnt(n).qposadr[0] for n in C.JOINT_NAMES]
    d = mujoco.MjData(m)
    achieved = []
    for pose_deg in C.PROBE_POSES_DEG:
        motor_deg = C.pipeline_motor_deg(pose_deg, calib)
        d.ctrl[:] = np.radians(motor_deg)
        for _ in range(750):  # 1.5 s settle per pose
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        achieved.append(np.degrees(d.qpos[qadr]).tolist())
    err = np.abs(np.array(achieved) - np.array(C.PROBE_POSES_DEG))
    return {"max_err_deg": round(float(err.max()), 3),
            "per_pose_max_err_deg": [round(float(e), 3) for e in err.max(axis=1)],
            "achieved_deg": [[round(float(v), 2) for v in row]
                             for row in achieved]}


def run_gate(cfg=None, run_eval: bool = True) -> dict:
    """Run all four checks; write the signed attestation (or the
    signed refusal record). Returns the record."""
    cfg = cfg or load_config()
    calib = C.ensure_calibration(cfg)
    arm = assets.arm_model_info("realistic")
    arm_md5 = arm.get("canonical_md5")
    checks = []

    # -- 1. calibration standardization --------------------------------
    offs = np.array(calib["offsets_deg"], dtype=float)
    max_off = float(np.abs(offs).max()) if offs.size else 0.0
    c1 = {"check": "calibration_standardization",
          "max_residual_offset_deg": round(max_off, 3),
          "tolerance_deg": C.OFFSET_TOLERANCE_DEG}
    if max_off > C.OFFSET_TOLERANCE_DEG:
        worst = C.JOINT_NAMES[int(np.argmax(np.abs(offs)))]
        c1.update(pass_=False,
                  reason=f"residual offset {max_off:.2f} deg on {worst} "
                         f"exceeds the {C.OFFSET_TOLERANCE_DEG} deg "
                         f"standardization tolerance — this node's "
                         f"calibration has not been converted into the "
                         f"protocol's canonical convention")
    else:
        try:
            probe = _sim_probe(calib)
            c1["sim_probe"] = probe
            if probe["max_err_deg"] > C.SIM_POSE_TOLERANCE_DEG:
                c1.update(pass_=False,
                          reason=f"probe on the sim arm: commanded vs "
                                 f"achieved poses diverge by "
                                 f"{probe['max_err_deg']} deg "
                                 f"(tolerance {C.SIM_POSE_TOLERANCE_DEG})")
            else:
                c1.update(pass_=True,
                          reason=f"residuals within tolerance and the "
                                 f"probe reproduces on the sim arm "
                                 f"(max err {probe['max_err_deg']} deg)")
        except Exception as e:  # assets missing etc. — a named failure
            c1.update(pass_=False, reason=f"sim probe could not run: {e}")
    checks.append(c1)

    # -- 2. action normalization ---------------------------------------
    from .sim.policy_eval import shared_stats, shared_stats_hash
    norm = calib.get("action_norm") or {"type": "identity"}
    c2 = {"check": "action_normalization", "declared": norm}
    if norm.get("type") == "identity":
        c2.update(pass_=True,
                  reason="identity at the protocol boundary (canonical "
                         "degrees in, degrees out)")
    elif norm.get("type") == "shared-stats":
        try:
            want = shared_stats_hash()
            if norm.get("stats_sha256") != want:
                c2.update(pass_=False,
                          reason="declared shared-stats hash does not "
                                 "match the shared checkpoint's normalizer "
                                 "statistics — not the protocol's statistics")
            else:
                err = C.norm_roundtrip_max_err_deg(
                    C.PROBE_POSES_DEG, calib, shared_stats=shared_stats())
                c2["roundtrip_max_err_deg"] = round(float(err), 6)
                c2.update(pass_=err <= C.NORM_ROUNDTRIP_TOLERANCE_DEG,
                          reason=f"shared checkpoint statistics, "
                                 f"hash-matched; round-trip max err "
                                 f"{err:.4f} deg")
        except Exception as e:
            c2.update(pass_=False, reason=f"shared-stats check failed: {e}")
    else:
        c2.update(pass_=False,
                  reason=f"declared normalization type "
                         f"{norm.get('type')!r} is not the protocol's — "
                         f"per-arm statistics make this node's actions "
                         f"mean something different from every other "
                         f"node's; the gate refuses them")
    checks.append(c2)

    # -- 3. calibration fingerprint current -----------------------------
    expect = C.calibration_fingerprint(calib, arm_md5) if arm_md5 else None
    c3 = {"check": "calibration_fingerprint_current",
          "generation": calib["generation"],
          "stored": calib.get("fingerprint"),
          "expected": expect}
    if calib["generation"] < 1:
        c3.update(pass_=False,
                  reason="calibration generation is 0 — this node has "
                         "never been calibrated (Stage 2)")
    elif not calib.get("fingerprint"):
        c3.update(pass_=False,
                  reason="no fingerprint stored — refresh the fingerprint "
                         "over the current calibration (Stage 2/3)")
    elif calib["fingerprint"] != expect:
        c3.update(pass_=False,
                  reason="stored fingerprint does not match the "
                         "calibration on disk — the calibration changed "
                         "since it was fingerprinted; re-verify and "
                         "refresh (a silent calibration edit re-opens "
                         "the gate by design)")
    else:
        c3.update(pass_=True,
                  reason="fingerprint matches the calibration on disk "
                         "and the staged arm model")
    checks.append(c3)

    # -- 4. local eval works --------------------------------------------
    c4 = {"check": "local_eval_works"}
    if not run_eval:
        c4.update(pass_=False, reason="eval check not run")
    else:
        try:
            from .sim.policy_eval import run_suite
            rec = run_suite("sim1", trials=2, cfg=cfg)
            v = verify_record(json.loads(
                (Path(rec["outdir"]) / "eval_record.json").read_text()))
            c4["eval_record"] = str(Path(rec["outdir"]) / "eval_record.json")
            c4["eval_verification"] = v
            if v["hash_ok"] and v["signature_ok"]:
                c4.update(pass_=True,
                          reason="Stage 5 quick suite (SIM-1, 2 trials) "
                                 "ran end-to-end and produced a "
                                 "signature-valid eval record")
            else:
                c4.update(pass_=False,
                          reason=f"eval record failed verification: {v}")
        except Exception as e:
            c4.update(pass_=False, reason=f"local eval could not run: {e}")
    checks.append(c4)

    passed = all(c["pass_"] for c in checks)
    now = datetime.now(timezone.utc)
    record = {
        "record_type": "join_gate_attestation" if passed
                       else "join_gate_refusal",
        "schema": "zk0-node-gate/0.1",
        "created_utc": utcnow_iso(),
        "node": {"node_id": cfg["node_id"],
                 "runtime_manifest_sha256": manifest_sha256(
                     runtime_manifest_path())},
        "track": "sim (SSOT decision 4 — sim contributions only; "
                 "real-validated status is earned via the real-arm "
                 "track's own gate)",
        "calibration_fingerprint": calib.get("fingerprint"),
        "calibration_generation": calib["generation"],
        "checks": [{("pass" if k == "pass_" else k): v
                    for k, v in c.items()} for c in checks],
        "failed_checks": [c["check"] for c in checks if not c["pass_"]],
        "pass": passed,
    }
    if passed:
        record["expires_utc"] = (now + timedelta(
            days=ATTESTATION_TTL_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = load_or_create_key()
    out = attestations_dir() / (
        f"gate-{record['created_utc'].replace(':', '').replace('-', '')}.json")
    write_signed_record(record, out, key)
    record["path"] = str(out)
    return record


def current_attestation(cfg=None) -> dict | None:
    """The node's latest passing, unexpired attestation, or None."""
    cfg = cfg or load_config()
    d = attestations_dir()
    if not d.exists():
        return None
    best = None
    for p in sorted(d.glob("gate-*.json")):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if rec.get("record_type") != "join_gate_attestation":
            continue
        if rec.get("node", {}).get("node_id") != cfg["node_id"]:
            continue
        if rec.get("expires_utc", "") <= utcnow_iso():
            continue
        v = verify_record(rec)
        if not (v["hash_ok"] and v["signature_ok"]):
            continue
        best = rec
    return best


def require_current_attestation(cfg=None) -> dict:
    """The protocol-blocking enforcement point (Stage 3). Raises
    GateRefused unless a current attestation exists."""
    cfg = cfg or load_config()
    att = current_attestation(cfg)
    if att is None:
        raise GateRefused(
            ["no current join-gate attestation — run `zk0 gate run` "
             "(Stage 3); contribution stages stay disabled until it "
             "passes"])
    calib = C.ensure_calibration(cfg)
    arm_md5 = assets.arm_model_info("realistic").get("canonical_md5")
    if arm_md5 and att.get("calibration_fingerprint") != \
            C.calibration_fingerprint(calib, arm_md5):
        raise GateRefused(
            ["the calibration changed since the attestation was "
             "issued — the gate re-opens; run `zk0 gate run` again"])
    return att


def record_founder_override(cfg, gate_record: dict, reason: str) -> dict:
    """Record a founder override of a refused gate (see module doc).
    Only identity_kind == 'founder' may override; the record names
    every bypassed check and the reason verbatim."""
    if cfg.get("identity_kind") != "founder":
        raise GateRefused(
            ["founder override attempted by a non-founder identity — "
             "refused; the override is founder-scope only"])
    if not reason or not reason.strip():
        raise GateRefused(["founder override requires a stated reason"])
    record = {
        "record_type": "join_gate_founder_override",
        "schema": "zk0-node-gate/0.1",
        "created_utc": utcnow_iso(),
        "node": {"node_id": cfg["node_id"], "identity_kind": "founder"},
        "bypassed_checks": gate_record.get("failed_checks", []),
        "gate_record_sha256": gate_record.get("content_sha256"),
        "reason_verbatim": reason,
        "note": "an admitted, attributable exception — the gate's "
                "verdict stands in its own record; this override does "
                "not edit it",
    }
    key = load_or_create_key()
    out = attestations_dir() / (
        f"override-{record['created_utc'].replace(':', '').replace('-', '')}.json")
    write_signed_record(record, out, key)
    record["path"] = str(out)
    return record
