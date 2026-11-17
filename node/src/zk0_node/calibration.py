"""Node calibration state and the protocol's canonical convention.

The M0 join gate (spec §3, Stage 3) exists because of a silent trap:
if each arm's calibration and action representation follow its own
local convention, identical canonical actions become different
physical angles on different arms — with no error raised anywhere.
The protocol therefore fixes ONE canonical convention, and joining
means proving your node conforms to it:

- **Canonical action frame:** raw joint positions in degrees, in the
  SO-101 calibrated frame (LeRobot order: shoulder_pan,
  shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper),
  where the calibrated zero is the model's zero.
- **Calibration standardization:** the node's calibration, converted
  into the canonical convention, leaves only small residual offsets
  (the gate's tolerance). A node whose calibration file still holds
  raw per-arm values (e.g. a +12 deg elbow residual that was never
  standardized) is refused — not because its arm is broken, but
  because its actions would mean something different from everyone
  else's.
- **Action normalization at the protocol boundary:** either the
  identity (canonical degrees in, degrees out) or the SHARED
  normalizer statistics of the shared checkpoint (the same stats
  every node normalizes with). Per-arm statistics are exactly what
  the gate refuses.

The calibration fingerprint (backlog C3) is a hash over the
calibration + the arm model it was taken against + its generation;
any change to the calibration re-opens the gate until the
fingerprint is refreshed and the gate re-run.
"""
import hashlib
import json

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper")

# The standard probe trajectory (spec §3 Stage 3, check 2): canonical
# poses in degrees, all inside the SO-101 joint ranges, spread across
# the workspace the benchmark tasks use. Fixed by the protocol —
# every node is probed with the same trajectory.
PROBE_POSES_DEG = [
    [0.0, -30.0, 40.0, 30.0, -50.0, 100.0],
    [-20.0, -50.0, 70.0, 50.0, -60.0, 100.0],
    [20.0, -20.0, 30.0, 20.0, -40.0, 20.0],
    [-35.0, -60.0, 80.0, 60.0, -53.0, 100.0],
    [10.0, -40.0, 58.0, 44.0, -53.0, 50.0],
]

# Gate tolerances (named here, recorded in every attestation).
OFFSET_TOLERANCE_DEG = 1.0      # residual offsets after standardization
SIM_POSE_TOLERANCE_DEG = 2.0    # commanded vs achieved on the sim arm
NORM_ROUNDTRIP_TOLERANCE_DEG = 0.1


def default_calibration() -> dict:
    return {
        "generation": 0,                    # 0 = never calibrated
        "offsets_deg": [0.0] * 6,           # residuals vs canonical frame
        "action_norm": {"type": "identity"},
        "fingerprint": None,
    }


def ensure_calibration(cfg: dict) -> dict:
    """cfg's calibration section, with defaults filled in."""
    calib = dict(default_calibration())
    calib.update(cfg.get("calibration") or {})
    cfg["calibration"] = calib
    return calib


def calibration_fingerprint(calib: dict, arm_model_md5: str) -> str:
    body = {
        "offsets_deg": [round(float(v), 6) for v in calib["offsets_deg"]],
        "action_norm": calib["action_norm"],
        "generation": int(calib["generation"]),
        "arm_model_md5": arm_model_md5,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def refresh_fingerprint(cfg: dict, arm_model_md5: str) -> str:
    """Recompute and store the fingerprint over the CURRENT calibration.

    This is the Stage 2/3 act that says "the calibration file on disk
    is the calibration I stand behind" — the gate then checks the
    stored fingerprint still matches, so a silent edit to the
    calibration re-opens the gate.
    """
    calib = ensure_calibration(cfg)
    calib["fingerprint"] = calibration_fingerprint(calib, arm_model_md5)
    return calib["fingerprint"]


def pipeline_motor_deg(canonical_deg, calib: dict):
    """The node's protocol-boundary pipeline: canonical action (deg)
    -> motor command (deg). Calibration offsets apply here; the
    declared action normalization must already have round-tripped
    (checked separately by the gate), so at the boundary the mapping
    is canonical + residual offsets."""
    offs = calib["offsets_deg"]
    return [float(a) + float(o) for a, o in zip(canonical_deg, offs)]


def norm_roundtrip_max_err_deg(poses_deg, calib: dict, shared_stats=None) -> float:
    """Max |denormalize(normalize(a)) - a| over the probe poses under
    the node's DECLARED action normalization. For the shared-stats
    declaration the stats are the checkpoint's (mean, std); anything
    else declared by the node is evaluated on its own parameters —
    and the gate separately refuses non-protocol declarations."""
    norm = calib.get("action_norm") or {"type": "identity"}
    t = norm.get("type")
    if t == "identity":
        return 0.0
    if t == "shared-stats":
        if shared_stats is None:
            raise ValueError("shared-stats declared but no shared stats "
                             "available to check against")
        mean, std = shared_stats
        err = 0.0
        for a in poses_deg:
            for j, v in enumerate(a):
                rt = (v - mean[j]) / std[j] * std[j] + mean[j]
                err = max(err, abs(rt - v))
        return err
    if t == "min-max":
        lo, hi = norm["min_deg"], norm["max_deg"]
        err = 0.0
        for a in poses_deg:
            for j, v in enumerate(a):
                n = 2.0 * (v - lo[j]) / (hi[j] - lo[j]) - 1.0
                rt = (n + 1.0) / 2.0 * (hi[j] - lo[j]) + lo[j]
                err = max(err, abs(rt - v))
        return err
    raise ValueError(f"unknown action_norm type {t!r}")
