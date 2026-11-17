"""BAM servo engagement (build slice 3): the certified BAM
fitted-servo profile as a selectable sim-harness servo, behind
ZK0_SERVO=bam, alongside the default variant profile.

Provenance — the profile is the certified BAM tier's, verbatim:
  servo law   punkfab/robot-actuators @ fb59c95 `sts3215/arm.py`
              (MIT), vendored at sim/bam/arm.py, md5
              63bfd2fc72fd0222b9c613436647e546 — identical to the
              copy published in zk0bot/so101-realistic v0.2.0
              (servo/bam/) and used by the certified tier.
  parameters  Rhoban/bam @ e9a619d, model STS3215 7.4 V variant "m6"
              (params/feetech_sts3215_7_4V/m6.json, md5
              3209fcb05bfbfddf68ae779fb87df02d), loaded at runtime
              from the installed `bam` package (better-actuator-
              models 1.0.2, installed from the pinned commit; its
              shipped m6.json is byte-identical — doctor check 1
              verifies both md5s).
  integration the certified tier's pattern (its bam_pilot.py, md5
              1bfaabf2f57f93a26a8109bd058214e3 in v0.2.0): subclass
              the harness Pilot, swap ONLY the actuation — targets
              go to StsServos.set() + servos.update() before each
              mj_step; the gate recorder is fed the position
              TARGETS as ctrl; the gate's saturation yardstick under
              BAM is the STS3215 rated stall 2.94 N·m (the MJCF
              motors are unlimited once bamify runs; the servo law
              is the limit).

Scope honesty: engaging the profile here does NOT re-certify BAM.
The certified tier's verdicts (S6b grasp fails under BAM; sag/step
class) stand on their own sealed inputs. What this slice proves is
that the package harness runs the same profile and reproduces its
published behavior class on the package's SIM-1 task — judged from
the run's state log, reported side by side with the variant run.
"""
import importlib.util
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

# The harness is imported the way the package runner imports it:
# top-level modules from the harness dir (MODEL comes from the
# ZK0_SIM_MODEL_DIR env var the runner sets before import).
HARNESS_DIR = Path(__file__).parent / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))
import multitask as MT  # noqa: E402
from grasp_gate import GateRecorder  # noqa: E402,F401  (re-export)

JOINTS = list(MT.JOINTS)
SERVO = os.environ.get("ZK0_SERVO", "variant")
STALL_NM = 2.94          # STS3215 rated stall (certified tier value)
BAM_PINS = {
    "bam_library": "Rhoban/bam @ e9a619d56da5236206f4de6ceec2c1ee1b497b5c "
                   "(installed package: better-actuator-models 1.0.2)",
    "servo_law": "punkfab/robot-actuators @ fb59c95 sts3215/arm.py (MIT)",
    "model": "feetech_sts3215_7_4V/m6",
}
PROFILE_DIR = Path(__file__).parent / "bam"


def _load_punkfab():
    spec = importlib.util.spec_from_file_location(
        "zk0_vendored_punkfab_arm", PROFILE_DIR / "arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bamify(spec, joints):
    return _load_punkfab().bamify(spec, joints)


def make_servos(model, data):
    return _load_punkfab().StsServos(model, data, list(JOINTS), "m6")


def gate_force_limit(pilot, servo=None):
    """The gate's saturation yardstick for a run (certified tier's
    rule): under BAM, the recorder's forcerange read is meaningless
    (bamify unlimits the MJCF motors) — the STS3215 rated stall is
    the limit the servo law itself enforces."""
    if (servo or SERVO) == "bam":
        return STALL_NM
    return pilot.gate.force_limit


class BamPilot(MT.Pilot):
    """MT.Pilot with BAM actuation. __init__/run are the harness's
    own bodies with ONLY the model construction / actuation swapped
    (the certified tier's integration pattern); planning, IK, gate
    instrumentation, and logging are the harness's, untouched."""

    def __init__(self, scene_key, ts=0.002, lookat=(0.24, -0.03, 0.04),
                 bodies=()):
        path = f"{MT.MODEL}/scene_{scene_key}.xml"
        with open(path, "w") as f:
            f.write(MT.SCENES[scene_key])
        spec = mujoco.MjSpec.from_file(path)
        bamify(spec, JOINTS)
        self.m = spec.compile()
        self.m.opt.timestep = ts
        self.ts = ts
        self.site_id = self.m.site("gripperframe").id
        self.qadr = [self.m.jnt(n).qposadr[0] for n in JOINTS]
        self.bodies = list(bodies)
        self.bqadr = {b: self.m.jnt(b + "_free").qposadr[0]
                      for b in self.bodies}
        self.gate = GateRecorder(self.m, self.bodies, self.site_id)
        self.cycles = []
        d0 = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, d0)
        R_home = d0.site_xmat[self.site_id].reshape(3, 3).copy()
        self.R_des = MT.rot_y(np.pi / 2) @ R_home
        self.seed = np.zeros(6); self.seed[5] = MT.GRIP_OPEN
        self.q = np.zeros(6); self.q[5] = MT.GRIP_OPEN
        self.segs = []; self.t = 0.0; self.events = {}
        self.ik_log = []
        MT.aim_camera(self.m, self.m.cam("main").id,
                      self.m.cam_pos[self.m.cam("main").id].copy(),
                      lookat)
        self.cam_home_pos = self.m.cam_pos[self.m.cam("main").id].copy()
        self.cam_home_quat = self.m.cam_quat[self.m.cam("main").id].copy()

    def run(self, render=False):
        m, ts = self.m, self.ts
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, height=480, width=640) if render else None
        n_steps = int(round(self.t / ts))
        sample_every = max(1, int(round(0.05 / ts)))
        gate_every = max(1, int(round(0.01 / ts)))
        frame_every = max(1, int(round((1 / 30) / ts)))
        snap_every = max(1, int(round(0.1 / ts)))
        log = {"t": [], "tcp": [], "q": [], "bodies": {b: [] for b in self.bodies},
               "quats": {b: [] for b in self.bodies}}
        snaps = {}
        frames = []
        # --- BAM actuation setup (replaces position ctrl) ---
        mujoco.mj_forward(m, d)                  # prime qfrc_* state
        servos = make_servos(m, d)
        servos.reset()
        self.servos = servos
        # --- end BAM setup ---
        for i in range(n_steps + 1):
            t = i * ts
            ctrl = MT.target_q(self.segs, t)
            # --- BAM actuation (replaces `d.ctrl[:] = ctrl`) ---
            for j, q in zip(JOINTS, ctrl):
                servos.set(j, float(q))
            servos.update()
            # --- end BAM actuation ---
            if i % gate_every == 0:
                mujoco.mj_forward(m, d)
                self.gate.sample(d, ctrl)
            if i % sample_every == 0:
                mujoco.mj_forward(m, d)
                log["t"].append(t)
                log["tcp"].append(d.site_xpos[self.site_id].copy())
                log["q"].append(d.qpos[:6].copy())
                for b in self.bodies:
                    a = self.bqadr[b]
                    log["bodies"][b].append(d.qpos[a:a + 3].copy())
                    log["quats"][b].append(d.qpos[a + 3:a + 7].copy())
            if i % snap_every == 0:
                snaps[round(t, 2)] = d.qpos.copy()
            if render and i % frame_every == 0:
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera="main")
                frames.append(renderer.render().copy())
            if i < n_steps:
                mujoco.mj_step(m, d)
        for b in self.bodies:
            log["bodies"][b] = np.array(log["bodies"][b])
            log["quats"][b] = np.array(log["quats"][b])
        log["t"] = np.array(log["t"]); log["tcp"] = np.array(log["tcp"])
        log["q"] = np.array(log["q"])
        return log, snaps, frames


def pick_pilot(servo=None):
    return BamPilot if (servo or SERVO) == "bam" else MT.Pilot


def task1_bam(cfg=None):
    """MT.task1 with the BAM pilot: identical scene, plan, and
    scoring (the harness's own functions), only the actuation
    differs — that isolation is the point of the comparison."""
    cfg = cfg or {}
    P = BamPilot("task1", bodies=["cube_green"],
                 lookat=(0.26, -0.06, 0.03))
    dx = cfg.get("dx", -0.01)
    MT.pick_place(P, (0.26, 0.02), 0.02, dx, MT.GRIP_CLOSE_CUBE,
                  cfg.get("carry_z", 0.135), (0.27, -0.14),
                  cfg.get("place_tcp_z", 0.028), tag="t1",
                  body="cube_green")
    P.home(); P.dwell(1.2)
    log, snaps, frames = P.run(render=cfg.get("render", False))
    pos = log["bodies"]["cube_green"][-1]
    res = dict(task="task1", dx=dx,
               final_cube=np.round(pos, 4).tolist(),
               inside_xy=bool(abs(pos[0] - 0.27) <= 0.04
                              and abs(pos[1] + 0.14) <= 0.04),
               below_rim=bool(0.015 <= pos[2] <= 0.06),
               settled_mm=round(MT.settled(log, "cube_green") * 1000, 2),
               arm_retreated=MT.arm_retreated(log))
    res["ok"] = bool(res["inside_xy"] and res["below_rim"]
                     and res["settled_mm"] < 2.0 and res["arm_retreated"])
    return P, log, snaps, frames, res


# ------------------------------------------------------------------
# Servo sanity probes — the certified tier's probe methods, ported
# verbatim in method (probe_servo.py of the BAM tier): step response
# at the elbow; sustained-load sag at punkfab's full-reach pose.
# ------------------------------------------------------------------
_TS = 0.002
_MID = dict(shoulder_pan=0.0, shoulder_lift=-0.60, elbow_flex=0.60,
            wrist_flex=0.5, wrist_roll=0.0, gripper=1.0)
_REACH = dict(shoulder_pan=0.0, shoulder_lift=-0.10, elbow_flex=0.10,
              wrist_flex=0.0, wrist_roll=0.0, gripper=1.0)


def _probe_build(xml, servo, payload_kg=0.0):
    spec = mujoco.MjSpec.from_file(xml)
    spec.option.timestep = _TS
    if payload_kg > 0:
        site = next(s for s in spec.sites if s.name == "gripperframe")
        b = site.parent.add_body(name="payload", pos=site.pos)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0],
                   mass=payload_kg, contype=0, conaffinity=0,
                   rgba=[0.9, 0.5, 0.1, 1])
    if servo == "bam":
        bamify(spec, JOINTS)
    m = spec.compile()
    d = mujoco.MjData(m)
    servos = None
    if servo == "bam":
        mujoco.mj_forward(m, d)
        servos = make_servos(m, d)
        servos.reset()
    return m, d, servos


class _Driver:
    def __init__(self, m, d, servos, servo):
        self.m, self.d, self.servos, self.servo = m, d, servos, servo
        self.qadr = {j: int(m.jnt_qposadr[m.joint(j).id]) for j in JOINTS}
        self.aid = {j: int(m.actuator(j).id) for j in JOINTS}
        self.target = {}

    def set_pose(self, pose):
        for j, q in pose.items():
            self.d.qpos[self.qadr[j]] = q
        self.target = dict(pose)
        mujoco.mj_forward(self.m, self.d)
        if self.servos:
            self.servos.reset()

    def step(self, target=None):
        if target:
            self.target.update(target)
        if self.servo == "bam":
            for j in JOINTS:
                self.servos.set(j, float(self.target[j]))
            self.servos.update()
        else:
            for j in JOINTS:
                self.d.ctrl[self.aid[j]] = float(self.target[j])
        mujoco.mj_step(self.m, self.d)

    def run(self, seconds, target=None):
        for _ in range(int(round(seconds / _TS))):
            self.step(target)
            target = None

    def q(self, j):
        return float(self.d.qpos[self.qadr[j]])


def _step_probe(xml, servo):
    m, d, servos = _probe_build(xml, servo)
    drv = _Driver(m, d, servos, servo)
    drv.set_pose(_MID)
    drv.run(1.5)
    q0 = drv.q("elbow_flex")
    tgt = _MID["elbow_flex"] + 0.5
    aid = drv.aid["elbow_flex"]
    ts_log, q_log, tau_log = [], [], []
    n = int(2.5 / _TS)
    for i in range(n):
        drv.step({"elbow_flex": tgt})
        ts_log.append((i + 1) * _TS)
        q_log.append(drv.q("elbow_flex"))
        tau_log.append(float(d.actuator_force[aid]))
    q = np.array(q_log)
    final = float(np.mean(q[int(-0.25 / _TS):]))
    err = tgt - final
    lo, hi = q0 + 0.1 * 0.5, q0 + 0.9 * 0.5
    t10 = next((t for t, v in zip(ts_log, q) if v >= lo), None)
    t90 = next((t for t, v in zip(ts_log, q) if v >= hi), None)
    return dict(servo=servo, joint="elbow_flex", step_rad=0.5,
                q_start=round(q0, 4), target=round(tgt, 4),
                final_q=round(final, 4),
                steady_state_error_deg=round(float(np.degrees(err)), 3),
                rise_time_10_90_s=(round(t90 - t10, 3) if t10 is not None
                                   and t90 is not None else None),
                overshoot_deg=round(float(np.degrees(
                    max(0.0, q.max() - tgt))), 3),
                peak_abs_torque_nm=round(float(np.max(np.abs(tau_log))), 3))


def _sag_probe(xml, servo, payload_kg):
    m, d, servos = _probe_build(xml, servo, payload_kg)
    drv = _Driver(m, d, servos, servo)
    site = m.site("gripperframe").id
    dk = mujoco.MjData(m)
    for j, q in _REACH.items():
        dk.qpos[drv.qadr[j]] = q
    mujoco.mj_kinematics(m, dk)
    tip_kin = dk.site_xpos[site].copy()
    drv.set_pose(_REACH)
    drv.run(4.0)
    zs, ps = [], []
    for _ in range(int(0.5 / _TS)):
        drv.step()
        zs.append(float(d.site_xpos[site][2]))
        ps.append(d.site_xpos[site].copy())
    tip = np.mean(ps, axis=0)
    return dict(servo=servo, payload_g=round(payload_kg * 1000),
                tip_kinematic=np.round(tip_kin, 4).tolist(),
                tip_settled=np.round(tip, 4).tolist(),
                sag_z_mm=round(float((tip_kin[2] - np.mean(zs)) * 1000), 2),
                sag_total_mm=round(float(np.linalg.norm(tip_kin - tip) * 1000), 2),
                elbow_err_deg=round(float(np.degrees(
                    _REACH["elbow_flex"] - drv.q("elbow_flex"))), 3),
                shoulder_err_deg=round(float(np.degrees(
                    _REACH["shoulder_lift"] - drv.q("shoulder_lift"))), 3))


def servo_probe(work_dir) -> dict:
    """Both profiles, both probes, on the staged arm model — the
    side-by-side the slice-3 report quotes next to the certified
    tier's published numbers (different harness, same profile and
    method: consistency evidence, not re-certification)."""
    xml = str(Path(work_dir) / "so101_realistic.xml")
    out = {"xml": "so101_realistic.xml (staged asset)", "ts": _TS,
           "step": [], "sag": []}
    for servo in ("variant", "bam"):
        out["step"].append(_step_probe(xml, servo))
        for kg in (0.0, 0.1, 0.2):
            out["sag"].append(_sag_probe(xml, servo, kg))
    return out
