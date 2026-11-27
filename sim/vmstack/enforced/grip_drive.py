"""Shared compliant gripper drive — harness change gripdrive/1.0-20261127.

WHY THIS EXISTS (measured, ep453 v4 solve session, campaign log SESSION
CLOSE 2026-11-27): pilot453_v4 and enforce_replay both drove the gripper
joint KINEMATICALLY (qpos written every substep, qvel zeroed). A kinematic
drive bypasses the actuator entirely, so arm v4's forcerange cap never
entered the loop and the grip could not yield. The real ep453 grasp
completed because the real STS3215 yielded 10.26 deg under load
(calibration-20261127/real_grasp_fk.json: hold plateau state 12.62 deg
vs commanded action median 2.36 deg; ep411: 9.90 deg). Under the rigid
drive a gentle touch already stored ~49.9 N of normal force on one cube
corner (cube weight 0.196 N) and any press past joint ~0.53918 detonated
the cube at |vy| 1.785 m/s, rate-independent; no lift was possible.

THE DRIVE MODEL (this module configures the GRIPPER joint's drive;
since harness P5 — 2026-11-27, actuation-corrections lane — the five
arm joints are driven the same way by their own actuators in both
harnesses: the recordings are the real servo-driven arm's COMMANDED
joint trajectories, and the teleport drive that previously imposed
them was measured to make friction lift mathematically zero and to
leave the arm actuators pulling toward q=0 on every substep
(neon-defect-verification-20261127/VERDICT.md, Defects 2+3). This
module's gripper model is unchanged by P5):
  * the gripper joint is driven by its MuJoCo <position> actuator;
  * the commanded joint angle (the trajectory's grip value, interpolated
    exactly as the old kinematic drive interpolated qpos) is written to
    data.ctrl every substep; qpos/qvel of the gripper dof are NEVER
    written by the harness — the servo tracks, saturates and yields
    under contact load like the real one;
  * at run start the gripper qpos is initialised to the frame-0 command
    (the old drive's frame-0 state) so a run starts at the same state.

CONSTANTS — every one derived, none invented:
  * GRIP_FORCERANGE_NM = 1.9123 N.m — Feetech STS3215 C001 stall torque
    19.5 kg.cm @ 7.4 V (Feetech spec A/0 2020-04-10); C001 is the
    standard SO-101 follower-kit servo per the SO-ARM100 BOM.
    Derivation: corrections-20261127/fact_check_record.md (A1). This is
    also the value arm v4's XML carries; configure() sets it explicitly
    so the drive does not silently depend on the arm version.
  * kp / kv are NOT set here by default: configure() retains the arm
    model's own gainprm/biasprm (arm v4: kp = 18.45701 N.m/rad,
    kv = 2.73117 N.m.s/rad — system-identified so101_new_calib values,
    calibration-20261127 step 4). An explicit kp argument overrides the
    gain (kv preserved) for calibration instruments.
    YIELD-CALIBRATION OUTCOME (2026-11-27, this lane — attempted as
    tasked, blocked by measurement, NOT tuned around): the static
    pinch test at the run-9 witness configuration was run over
    kp = 0.25..72 N.m/rad (grip-drive-20261127/grip_calibration.py,
    grip_calibration.json, diag_pinch.py). NO gain formed the pinch:
    at kp <= 0.5 the servo cannot close the jaw to the cube at all;
    at every kp >= 1 the moving jaw's leading box (b000) meets the
    cube's (+15,+15,+15) mm corner first (grip ~31.1 deg, exactly the
    run-9 touch) and the sustained press cams the cube out of the
    pocket's open -y side (12.5 mm at the pure witness; 75 mm with
    run 9's +0.45 mm walkx shift) — the fingertip face never reaches
    the cube's side face, so there is no pinch whose yield could be
    measured. The block is the contact geometry of the padded-box
    decomposition (driver-notes option (b): contact-model review —
    a corrections-lane item), not the gain. Until that review lands,
    the drive carries the arm model's system-identified gain, which
    has its own derivation; no gain was invented here.

SEAL IMPLICATION: the harness element of the (scene, arm, engine,
harness) tuple changed. Tuples citing pilot453_v4 / enforce_replay
BEFORE this change are superseded for future work; their records stay
as history. Both harnesses import THIS module so a trajectory piloted
under it replays under the identical drive model — a trajectory piloted
under one drive and replayed under another is invalid.

Enforcement is untouched: collision/contact truth, contype discipline,
the declared contact set, the alarm layer and the weld rules are
exactly as before. Only the gripper's actuation model changed.
"""
import mujoco

DRIVE_VERSION = "gripdrive/1.0-20261127"
GRIP_FORCERANGE_NM = 1.9123


def configure(model, kp=None):
    """Point the gripper actuator at the compliant drive parameters.

    Always sets the derived forcerange cap (the fix: the cap must be in
    the loop). With kp=None (the harness default) the arm model's own
    gainprm/biasprm are retained untouched (system-identified gains —
    see the module docstring for why no yield-calibrated gain exists).
    An explicit kp overrides gainprm[0]/biasprm[1] for calibration
    instruments, preserving the model's kv. Returns the actuator id."""
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    if kp is not None:
        kv = -float(model.actuator_biasprm[act][2])
        model.actuator_gainprm[act][0] = float(kp)
        model.actuator_biasprm[act][0] = 0.0
        model.actuator_biasprm[act][1] = -float(kp)
        model.actuator_biasprm[act][2] = -kv
    model.actuator_forcerange[act][0] = -GRIP_FORCERANGE_NM
    model.actuator_forcerange[act][1] = GRIP_FORCERANGE_NM
    model.actuator_forcelimited[act] = 1
    return act


def describe(model, act):
    """Drive parameter record for run records / banners."""
    return {"drive_version": DRIVE_VERSION,
            "gripper_drive": "actuated position servo (compliant)",
            "arm_joints_drive": "actuated position servos (harness P5 full "
                                "actuation, actuation-corrections-20261127)",
            "kp_Nm_per_rad": float(model.actuator_gainprm[act][0]),
            "kv_Nm_s_per_rad": float(-model.actuator_biasprm[act][2]),
            "kp_source": "arm model (system-identified so101_new_calib); "
                         "yield calibration blocked by contact geometry — "
                         "grip-drive-20261127/grip_calibration.json",
            "forcerange_Nm": [float(model.actuator_forcerange[act][0]),
                              float(model.actuator_forcerange[act][1])]}
