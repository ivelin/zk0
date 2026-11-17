"""
SO-101 follower arm in MuJoCo with a choice of STS3215 actuator model.

  stock   the upstream MJCF (TheRobotStudio SO-ARM100): an ideal MuJoCo position
          servo, kp 998 N·m/rad, kv 2.7, ±2.94 N·m, plus Coulomb/viscous friction.
  m1..m6  BAM (github.com/Rhoban/bam) STS3215 7.4 V models: the servo's firmware
          P law (rate-limited target, duty = kp·error_gain·Δq, clipped), a DC motor
          with back-EMF, and M1–M6 gearbox friction updated every step.

Firmware P gain: lerobot's SO-101 follower writes P_Coefficient = 16 (the servo's
default is 32, "lowered to avoid shakiness"), so KP_FIRMWARE defaults to 16.

The arm model is used by reference from software-mfg (vendored upstream, Apache-2.0),
override with SO101_XML=/path/to/so101_new_calib.xml.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
from bam import mujoco as bam_mujoco
from bam.model import load_model

SO101_XML = Path(os.environ.get(
    "SO101_XML",
    Path.home() / "sandbox/punkfab/software-mfg/sim/so101/so101_new_calib.xml"))
BAM_PARAMS = Path(bam_mujoco.__file__).parent / "params" / "feetech_sts3215_7_4V"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = JOINTS[:5]
KP_FIRMWARE = 16          # lerobot so_follower.configure()
VIN = 7.4                 # V, the servo variant BAM identified
TIMESTEP = 0.002          # s

# Named poses (rad), new calibration: zero = mid-range.
POSES = {
    "rest":    dict(shoulder_pan=0.0, shoulder_lift=-1.60, elbow_flex=1.55, wrist_flex=0.7, wrist_roll=0.0),
    "reach":   dict(shoulder_pan=0.0, shoulder_lift=-0.10, elbow_flex=0.10, wrist_flex=0.0, wrist_roll=0.0),
    "mid":     dict(shoulder_pan=0.3, shoulder_lift=-0.60, elbow_flex=0.60, wrist_flex=0.5, wrist_roll=0.0),
}


def bamify(spec, joints):
    """Turn the MJCF position actuators on `joints` (actuator names) into plain torque
    motors, so a BAM controller can drive them. Works on any spec that contains the
    SO-101 (prefixed names are fine: pass them as they appear)."""
    names = set(joints)
    for a in spec.actuators:
        if a.name in names:
            a.set_to_motor()
            a.gear[0] = 1.0
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE


class StsServos:
    """BAM STS3215 servos on the given joints of a compiled model (after bamify).

    Actuator and joint names must match (true for the SO-101 MJCF, with any prefix).
    set(name, q) sets a goal; update() before every mj_step computes the torques from
    the firmware P law and rewrites each joint's friction for this step.
    """

    def __init__(self, model, data, joints, kind="m6", kp=KP_FIRMWARE, vin=VIN,
                 friction_scale=1.0, gain_scale=1.0, armature_scale=1.0,
                 stiff_friction=True):
        bm = load_model(str(BAM_PARAMS / f"{kind}.json"))
        bm.actuator.kp = kp
        bm.actuator.vin = vin
        # Real2sim corrections (fit.py): multipliers on the identified values.
        # friction_scale scales the velocity-independent budget, as Microduck's DR does.
        for name in ("friction_base", "friction_stribeck", "load_friction_base",
                     "load_friction_stribeck", "load_friction_motor",
                     "load_friction_external", "load_friction_motor_stribeck",
                     "load_friction_external_stribeck"):
            if hasattr(bm, name):
                getattr(bm, name).value *= friction_scale
        bm.error_gain_ratio.value *= gain_scale
        bm.armature.value *= armature_scale
        self.ctrl = bam_mujoco.MujocoController(bm, list(joints), model, data)
        if stiff_friction:
            # MuJoCo's soft friction creeps under a held load (robot-actuators/friction/README).
            dofs = [model.jnt_dofadr[model.joint(j).id] for j in joints]
            model.dof_solimp[dofs, :2] = (0.999, 0.9999)

    def set(self, name, q):
        self.ctrl.set_q_target(name, q)

    def reset(self):
        self.ctrl.reset()

    def update(self):
        self.ctrl.update()


class Arm:
    """One SO-101 with a given actuator model. step(q_target) advances TIMESTEP."""

    def __init__(self, kind="m6", payload_kg=0.0, **servo_kw):
        self.kind = kind
        spec = mujoco.MjSpec.from_file(str(SO101_XML))
        spec.option.timestep = TIMESTEP
        if payload_kg > 0:
            self._add_payload(spec, payload_kg)
        if kind != "stock":
            bamify(spec, JOINTS)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.jid = {j: self.m.joint(j).id for j in JOINTS}
        self.qadr = np.array([self.m.jnt_qposadr[self.jid[j]] for j in JOINTS])
        self.dadr = np.array([self.m.jnt_dofadr[self.jid[j]] for j in JOINTS])
        self.aid = np.array([self.m.actuator(j).id for j in JOINTS])
        self.site = self.m.site("gripperframe").id
        # stock stays exactly the upstream model
        self.ctrl = None if kind == "stock" else StsServos(self.m, self.d, JOINTS, kind, **servo_kw)

    @staticmethod
    def _add_payload(spec, kg):
        site = next(s for s in spec.sites if s.name == "gripperframe")
        body = site.parent
        b = body.add_body(name="payload", pos=site.pos)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], mass=kg,
                   contype=0, conaffinity=0, rgba=[0.9, 0.5, 0.1, 1])

    def reset(self, pose: dict):
        mujoco.mj_resetData(self.m, self.d)
        for j, q in pose.items():
            self.d.qpos[self.m.jnt_qposadr[self.jid[j]]] = q
        mujoco.mj_forward(self.m, self.d)
        if self.ctrl is not None:
            self.ctrl.reset()
        self.target = {j: pose.get(j, 0.0) for j in JOINTS}

    def step(self, target: dict | None = None):
        if target:
            self.target.update(target)
        if self.ctrl is None:
            for j, a in zip(JOINTS, self.aid):
                self.d.ctrl[a] = self.target[j]
        else:
            for j in JOINTS:
                self.ctrl.set(j, self.target[j])
            self.ctrl.update()
        mujoco.mj_step(self.m, self.d)

    def run(self, seconds, target=None):
        for _ in range(int(round(seconds / TIMESTEP))):
            self.step(target)
            target = None

    @property
    def q(self):
        return self.d.qpos[self.qadr].copy()

    def tip(self):
        return self.d.site_xpos[self.site].copy()

    def tip_at(self, pose: dict):
        """Gripper-frame position if the joints were exactly at `pose` (kinematics only)."""
        d = mujoco.MjData(self.m)
        for j, q in pose.items():
            d.qpos[self.m.jnt_qposadr[self.jid[j]]] = q
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.site].copy()
