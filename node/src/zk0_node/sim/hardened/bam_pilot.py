"""BAM servo tier for the zk0 desc-scenes harness (2026-11-16).

Servo modes (env ZK0_SERVO):
  variant  so101-realistic v0.1.0 as shipped: Menagerie position actuators
           (kp 998.22 / kv 2.731, forcerange +-2.94) on the de-tuned variant.
           This is the control arm of every A/B below.
  bam      the SAME variant model with its position actuators converted to
           torque motors (punkfab/robot-actuators `bamify`) and driven per
           physics step by BAM's fitted STS3215 model (Rhoban/bam M6,
           LeRobot firmware P=16, vin 7.4 V) via punkfab's `StsServos`:
           firmware P law with rate-limited target + command delay, DC
           motor with back-EMF, and M6 load-dependent gearbox friction
           rewritten into dof_frictionloss / dof_damping every step.

Everything else — pilot segments, IK, gate recorder, scoring — is the
harness's own code, untouched. In BAM mode the gate recorder is fed the
pilot's position TARGETS as its `ctrl` argument (the gate's command
semantics are about commanded positions; the recorder's force channel
reads d.actuator_force, which under BAM is the applied servo torque).

Sources (recorded in the report):
  Rhoban/bam @ e9a619d56da5236206f4de6ceec2c1ee1b497b5c (Apache-2.0),
  punkfab/robot-actuators @ fb59c959045b73d7e49b203d3a98ce53472511f3 (MIT).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
# [slice-4 vendoring patch — see PROVENANCE.md] paths were absolute Spark workdir constants;
# HERE (defined above) is this vendored directory, and the
# punkfab servo law is the package's own vendored copy.
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "bam")))

import harness
from harness import Pilot, target_q, HOME_Q, JOINTS, MODEL

SERVO = os.environ.get("ZK0_SERVO", "variant")
STALL_NM = 2.94          # STS3215 rated stall; the gate's saturation yardstick

_bam = None
def _load_bam():
    global _bam
    if _bam is None:
        from arm import bamify, StsServos   # punkfab/robot-actuators sts3215/arm.py
        _bam = (bamify, StsServos)
    return _bam


def load_model_bam(xml, tag="scene"):
    """harness.load_model replacement: same write-to-MODEL behaviour, then
    MjSpec -> bamify -> compile."""
    bamify, _ = _load_bam()
    path = f"{MODEL}/desc_{tag}.xml"
    with open(path, "w") as f:
        f.write(xml)
    spec = mujoco.MjSpec.from_file(path)
    bamify(spec, list(JOINTS))
    return spec.compile()


def install_arm(fname):
    """Copy model/<fname> over model/so101.xml (the file scene_xml includes).
    Only ever touches this work dir's own harness copy."""
    import shutil
    src = os.path.join(MODEL, fname)
    dst = os.path.join(MODEL, "so101.xml")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)


class BamPilot(Pilot):
    """Pilot with the run loop's actuation replaced by the BAM servo law.
    The loop body is harness.Pilot.run copied verbatim except the marked
    actuation block: segment targets go to StsServos (set + update before
    every mj_step) instead of d.ctrl."""

    def run(self, record_cams=None, render=True):
        _, StsServos = _load_bam()
        m, ts = self.m, self.ts
        d = mujoco.MjData(m)
        rend = {}
        if render:
            for cname, h, w in (("main", 480, 640), ("close", 480, 640),
                                (record_cams[0], 240, 320), (record_cams[1], 240, 320)):
                rend[cname] = mujoco.Renderer(m, height=h, width=w)
        d.qpos[:6] = self.q0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)                      # prime qfrc_* for the controller
        servos = StsServos(m, d, list(JOINTS), "m6")  # kp=16, vin=7.4 defaults
        servos.reset()
        self.servos = servos
        n_steps = int(round(self.t / ts))
        sample_every = max(1, int(round(0.05 / ts)))
        gate_every = max(1, int(round(0.01 / ts)))
        frame_every = max(1, int(round((1 / 30) / ts)))
        log = {"t": [], "tcp": [], "q": [], "bodies": {b: [] for b in self.bodies},
               "quats": {b: [] for b in self.bodies}}
        rec = {"state": [], "action": [], "camA": [], "camB": [], "wide": [], "close": []}
        for i in range(n_steps + 1):
            t = i * ts
            ctrl = target_q(self.segs, t)
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
            if render and i % frame_every == 0:
                mujoco.mj_forward(m, d)
                rec["state"].append(d.qpos[:6].copy())
                rec["action"].append(ctrl.copy())
                for key, cname in (("wide", "main"), ("close", "close"),
                                   ("camA", record_cams[0]), ("camB", record_cams[1])):
                    rend[cname].update_scene(d, camera=cname)
                    rec[key].append(rend[cname].render().copy())
            if i < n_steps:
                mujoco.mj_step(m, d)
        for b in self.bodies:
            log["bodies"][b] = np.array(log["bodies"][b])
            log["quats"][b] = np.array(log["quats"][b])
        log["t"] = np.array(log["t"]); log["tcp"] = np.array(log["tcp"]); log["q"] = np.array(log["q"])
        for k in rec:
            rec[k] = np.array(rec[k])
        return log, rec


def settle_probe_bam(xml, bodies_spec, dur=2.5, ts=0.002):
    """harness.settle_probe with the parked arm held by the BAM servo law
    (targets = HOME_Q) instead of position ctrl. Pin logic identical."""
    _, StsServos = _load_bam()
    m = load_model_bam(xml, tag="settle")
    m.opt.timestep = ts
    d = mujoco.MjData(m)
    d.qpos[:6] = HOME_Q
    mujoco.mj_forward(m, d)
    servos = StsServos(m, d, list(JOINTS), "m6")
    servos.reset()
    for j, q in zip(JOINTS, HOME_Q):
        servos.set(j, float(q))
    qadr = {b: m.jnt(b + "_free").qposadr[0] for b in bodies_spec}
    start = {}
    mujoco.mj_forward(m, d)
    for b, a in qadr.items():
        start[b] = (d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy())
    n = int(dur / ts)
    n_pin = int(min(0.7, dur / 3) / ts)
    spawn = {b: d.qpos[a:a + 7].copy() for b, a in qadr.items()}
    vadr = {b: m.jnt(b + "_free").dofadr[0] for b in bodies_spec}
    for i in range(n):
        if i < n_pin:
            for b, a in qadr.items():
                d.qpos[a:a + 7] = spawn[b]
                d.qvel[vadr[b]:vadr[b] + 6] = 0.0
        servos.update()
        mujoco.mj_step(m, d)
    out = {}
    for b, a in qadr.items():
        p1, q1 = d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy()
        p0, q0 = start[b]
        dq = np.zeros(4); mujoco.mju_mulQuat(dq, q1, [-q0[0], q0[1], q0[2], q0[3]])
        tilt = 2 * np.arccos(np.clip(abs(dq[0]), -1, 1))
        out[b] = dict(disp_mm=round(float(np.linalg.norm(p1 - p0)) * 1000, 2),
                      tilt_deg=round(float(np.degrees(tilt)), 2),
                      final_pos=np.round(p1, 4).tolist(),
                      spec_pos=np.round(np.array(bodies_spec[b][0]), 4).tolist())
    return out


def pick_pilot():
    return BamPilot if SERVO == "bam" else Pilot


def pick_settle():
    return settle_probe_bam if SERVO == "bam" else harness.settle_probe


def force_limit(pilot):
    """Gate saturation yardstick: under BAM the motors are unlimited in the
    MJCF (the servo law limits torque), so the recorder's forcerange read
    is meaningless — use the STS3215 rated stall, the same 2.94 N·m the
    stock model's forcerange encodes."""
    if SERVO == "bam":
        return STALL_NM
    return pilot.gate.force_limit
