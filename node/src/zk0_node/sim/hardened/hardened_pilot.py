"""Force-aware closing pilots + contract-section-6 ablation-ladder
degrader + HARDENED realistic controller (zk0 rung-3 rematch,
2026-11-17). The certified ladder_pilot.py with exactly one change:
RealisticController is replaced by HardenedRealisticController
(preregistration: zk0-rung3-rematch-20261117). ObsDegrader,
OracleController, the pilot loop and all logging are unchanged.

The certified zk0-force-aware-20261116/code/fa_pilot.py with exactly
three changes: (1) path constants point at this workdir; (2) an
ObsDegrader (contract section 6 ladder) is added; (3) the REALISTIC
branch of ForceAwareBamPilot.run receives the DEGRADED {q, duty}
sample and logs it (plus the raw duty as duty_raw). Controllers,
plan handling, gate interaction and the oracle branch are
byte-identical to the certified file.

Pre-registration: zk0-ablation-ladder-20261116/preregistration.md
(base experiment: zk0-force-aware-20261116/preregistration.md).

Two controller variants share one run loop (ForceAwareBamPilot, a
BamPilot whose gripper channel is overridden between close-start and
open-start of the certified plan; every other channel follows the plan
exactly, and outside the window the gripper follows the plan too).

SIGNAL CONTRACTS (structural — the verifier can read them off the
classes):
- OracleController.step receives fb = {F, F1, rel}: TRUE contact
  forces from d.contact and the mug's position in the gripper frame.
  This is the sensing upper bound; no real servo sees these.
- RealisticController.step receives fb = {q, duty} ONLY: the gripper's
  present position and the BAM-modeled present load (the gripper
  servo's duty cycle from the BAM actuator state — the quantity the
  real Feetech STS3215 reports as Present Load). The class is built
  with no reference to mujoco data, contacts, or object state.
Ground truth in realistic runs is logged by the runner/recorder for
scoring only; it never enters RealisticController.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

# [slice-4 vendoring patch — see PROVENANCE.md] paths were absolute Spark workdir constants;
# the vendored modules live together in this directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from harness import target_q, JOINTS
import bam_pilot as _bp
from bam_pilot import BamPilot

GRIP_IDX = 5
TICK_S = 0.01            # control tick = gate rate (100 Hz)



class ObsDegrader:
    """Contract §6 ablation-ladder degradation of the REALISTIC
    controller's observation path (preregistration:
    zk0-ablation-ladder-20261116). Fed the raw (q, duty) sample at
    EVERY gate tick from run start; feed() returns the sample the
    controller sees at that tick. Pipeline (sensor -> register ->
    bus), stages cumulative by rung:

      rung>=3: additive Gaussian noise on the RAW sample
               (sigma_q = 0.005 rad, sigma_duty = 0.01 — PROVISIONAL
               placeholders, provenance in the preregistration)
      rung>=1: quantization — q to 4096 counts/rev (Present_Position),
               duty to 0.1% counts (Present_Load sign-magnitude),
               duty clipped to the register's [-1, 1]
      rung>=2: one-tick delay — the controller receives the previous
               tick's formed sample (zero-order hold between ticks is
               the pilot's existing structure)

    Rung 0 is the identity. The scorer/gate never see degraded values.
    """

    Q_STEP = 2.0 * np.pi / 4096.0
    D_STEP = 0.001
    SIG_Q = 0.005
    SIG_DUTY = 0.01

    def __init__(self, rung, seed=20261116):
        self.rung = int(rung)
        self.rng = (np.random.RandomState(seed) if self.rung >= 3
                    else None)
        self._prev = None

    def feed(self, q_raw, duty_raw):
        q, duty = float(q_raw), float(duty_raw)
        if self.rng is not None:
            q += float(self.rng.normal(0.0, self.SIG_Q))
            duty += float(self.rng.normal(0.0, self.SIG_DUTY))
        if self.rung >= 1:
            q = round(q / self.Q_STEP) * self.Q_STEP
            duty = round(duty / self.D_STEP) * self.D_STEP
            duty = min(1.0, max(-1.0, duty))
        formed = (q, duty)
        if self.rung >= 2:
            out = self._prev if self._prev is not None else formed
            self._prev = formed
            return out
        return formed

class OracleController:
    """Close to a true-contact-force band; maintain through the carry.

    Band [8, 14] N summed finger normal force (derivation in the
    pre-registration). Slip = mug CoM displacement in the gripper
    frame vs the reference captured at hold entry."""

    def __init__(self, jnt_range, band_lo=8.0, band_hi=14.0,
                 sum_cap=16.0, single_cap=20.0):
        self.jlo, self.jhi = jnt_range
        self.band_lo, self.band_hi = band_lo, band_hi
        self.sum_cap, self.single_cap = sum_cap, single_cap
        self.cmd = None

    def begin(self, cmd0):
        self.cmd = float(cmd0)
        self.state = "closing"
        self.cmd_hold = None
        self.rel_ref = None
        self._band_ticks = 0
        self._contact_ticks = 0
        self.contact = False
        self.overrides = 0
        self.floor_limited = False
        self.tighten_events = 0
        self.ease_events = 0

    def step(self, fb, t):
        F, F1, rel = fb["F"], fb["F1"], fb["rel"]
        # --- safety override: ease on summed or single-contact spike ---
        if F >= self.sum_cap or F1 >= self.single_cap:
            self.cmd += 0.004
            self.overrides += 1
            self.cmd = min(max(self.cmd, self.jlo), (self.cmd_hold + 0.03)
                           if self.cmd_hold is not None else self.jhi)
            return self.cmd
        if self.state == "closing":
            if F >= self.band_lo:
                self._band_ticks += 1
                if self._band_ticks >= 3:
                    self.state = "holding"
                    self.cmd_hold = self.cmd
                    self.rel_ref = np.array(rel, float)
            else:
                self._band_ticks = 0
                # two-rate close (preregistration AMENDMENT): fast
                # approach until first contact, then the registered
                # 0.08 rad/s squeeze into the band
                if not self.contact:
                    if F >= 2.0:   # amendment 2: graze forces during
                    # blade seating measure <=0.57 N; real wall
                    # engagement rises through 2 N
                        self._contact_ticks += 1
                        if self._contact_ticks >= 2:
                            self.contact = True
                    else:
                        self._contact_ticks = 0
                rate = 0.08 if self.contact else 0.30
                self.cmd -= rate * TICK_S
                if self.cmd <= self.jlo:
                    self.cmd = self.jlo
                    self.floor_limited = True
        else:  # holding
            slip_mm = float(np.linalg.norm(np.array(rel, float)
                                           - self.rel_ref)) * 1000.0
            if F < self.band_lo:
                self.cmd -= 0.004
                self.tighten_events += 1
            elif slip_mm > 3.0 and F < self.band_hi:
                self.cmd -= 0.004                  # slip response
                self.tighten_events += 1
            elif F > self.band_hi:
                self.cmd += 0.002
                self.ease_events += 1
            self.cmd = min(max(self.cmd, self.jlo), self.cmd_hold + 0.03)
            if self.cmd <= self.jlo:
                self.floor_limited = True
        return self.cmd


class HardenedRealisticController:
    """Noise-hardened realistic controller (rung-3 rematch,
    preregistration: zk0-rung3-rematch-20261117). Same signal
    contract as the certified RealisticController - step receives
    the delivered {q, duty} ONLY - and the same calibrated bands,
    two-rate close, tighten/ease rules and clamps. ONLY the decision
    logic changes, per the registered design; every parameter is a
    formula in (SIG_DUTY, SIG_Q), evaluated at the PROVISIONAL
    placeholders:

    (a) Contact: Ds >= THETA_C for N_C consecutive ticks.
        p* = 1 - Phi((THETA_C - G_STAR)/SIG_DUTY) = 1 - Phi(0.3)
           = 0.382 (worst sustained graze G_STAR = 0.087);
        N_C = ceil(ln(ALPHA/L)/ln p*) = ceil(ln(0.01/55)/ln 0.382)
            = 9. (Certified law: N = 2 -> ~8 expected false triggers
        per graze plateau; hardened: <= 0.0095.)
    (b) Band entry: condition Ds >= d_lo + ENTRY_MARGIN with
        ENTRY_MARGIN = 1*SIG_DUTY, accumulated by a LEAKY INTEGRATOR
        (+1 in-condition, -1 out, floor 0); entry at K_B = 5.
        Flicker is absorbed as net progress, never a reset to zero.
    (c) Creep: displacement of the W-tick windowed mean of delivered
        q from a W-tick windowed reference (captured at hold entry),
        threshold THETA_CREEP = 4*SIG_Q = 0.020 rad
        (sigma of the windowed displacement ~ SIG_Q*sqrt(2/W)
        = 0.0022, so the threshold is ~9 sigma). The flag keeps its
        certified role: it sizes the low-duty tighten step (0.008
        vs 0.004) and counts creep events.
    """

    SIG_DUTY = 0.01        # PROVISIONAL placeholder (contract S5)
    SIG_Q = 0.005          # PROVISIONAL placeholder (contract S5)
    THETA_C = 0.09         # calibrated contact threshold (unchanged)
    G_STAR = 0.087         # worst sustained graze (certified record)
    L_EXP = 55             # longest observed graze plateau, ticks
    ALPHA = 0.01           # design false-trigger rate per graze
    N_C = 9                # ceil(ln(ALPHA/L_EXP)/ln(1-Phi(0.3)))
    ENTRY_MARGIN = 1.0 * SIG_DUTY
    K_B = 5
    W = 10
    THETA_CREEP = 4.0 * SIG_Q

    def __init__(self, jnt_range, d_lo, d_hi):
        self.jlo, self.jhi = jnt_range
        self.d_lo, self.d_hi = float(d_lo), float(d_hi)
        self.cmd = None

    def begin(self, cmd0):
        self.cmd = float(cmd0)
        self.state = "closing"
        self.cmd_hold = None
        self.q_ref = None
        self._entry_c = 0
        self._low_ticks = 0
        self._contact_run = 0
        self.contact = False
        self.contact_t = None
        self.entry_t = None
        self.floor_limited = False
        self.tighten_events = 0
        self.ease_events = 0
        self.creep_events = 0
        self._qbuf = []

    def step(self, fb, t):
        q = fb["q"]
        Ds = max(0.0, -fb["duty"])       # squeeze duty (closing = -q dir)
        self._qbuf.append(q)
        if len(self._qbuf) > self.W:
            self._qbuf.pop(0)
        if self.state == "closing":
            # (b) band entry: tightened edge + leaky integrator
            if Ds >= self.d_lo + self.ENTRY_MARGIN:
                self._entry_c += 1
                if self._entry_c >= self.K_B:
                    self.state = "holding"
                    self.cmd_hold = self.cmd
                    # (c) windowed reference at entry
                    self.q_ref = float(np.mean(self._qbuf))
                    self.entry_t = t
                    return self.cmd
            else:
                self._entry_c = max(0, self._entry_c - 1)
                # (a) contact: THETA_C crossing must persist N_C ticks
                if not self.contact:
                    if Ds >= self.THETA_C:
                        self._contact_run += 1
                        if self._contact_run >= self.N_C:
                            self.contact = True
                            self.contact_t = t
                    else:
                        self._contact_run = 0
                rate = 0.08 if self.contact else 0.30
                self.cmd -= rate * TICK_S
                if self.cmd <= self.jlo:
                    self.cmd = self.jlo
                    self.floor_limited = True
        else:  # holding
            # (c) creep: windowed displacement vs windowed reference
            disp = float(np.mean(self._qbuf)) - self.q_ref
            creep = disp <= -self.THETA_CREEP
            if Ds < self.d_lo:
                self._low_ticks += 1
                if self._low_ticks >= 3:
                    step = 0.008 if creep else 0.004
                    if creep:
                        self.creep_events += 1
                    self.cmd -= step
                    self.tighten_events += 1
                    self._low_ticks = 0
            else:
                self._low_ticks = 0
            if Ds > self.d_hi:
                self.cmd += 0.002
                self.ease_events += 1
            self.cmd = min(max(self.cmd, self.jlo), self.cmd_hold + 0.03)
            if self.cmd <= self.jlo:
                self.floor_limited = True
        return self.cmd


class ForceAwareBamPilot(BamPilot):
    """BamPilot.run with the gripper channel overridden inside
    [fa_t0, fa_t1) by fa_controller (None = pure open loop, used by
    the calibration probes). Duty is logged at every gate tick in
    every mode (self.duty_log) — telemetry, never an oracle input.
    Controller trace at gate ticks in the window: self.ctrl_log."""

    fa_controller = None
    fa_t0 = None
    fa_t1 = None
    ladder_rung = 0
    ladder_seed = 20261116

    def _duty(self):
        act = self.servos.ctrl.model.actuator
        dc = getattr(act, "duty_cycle", None)
        if dc is None:
            return 0.0
        return float(np.asarray(dc, dtype=float).reshape(-1)[GRIP_IDX])

    def _oracle_feedback(self, d):
        """TRUE contact aggregates on the mug — oracle variant only."""
        m = self.m
        mug_body = self.gate.body_id["mug"]
        gb = m.geom_bodyid
        F = 0.0
        F1 = 0.0
        f6 = np.zeros(6)
        for ci in range(d.ncon):
            c = d.contact[ci]
            b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
            if mug_body not in (b1, b2):
                continue
            bother = b2 if b1 == mug_body else b1
            if bother in self.gate.robot_bodies:
                mujoco.mj_contactForce(m, d, ci, f6)
                nf = abs(float(f6[0]))
                F += nf
                F1 = max(F1, nf)
        sp = d.site_xpos[self.site_id]
        sR = d.site_xmat[self.site_id].reshape(3, 3)
        qa = self.gate.bqadr["mug"]
        rel = sR.T @ (d.qpos[qa:qa + 3] - sp)
        return dict(F=F, F1=F1, rel=rel)

    def run(self, record_cams=None, render=True):
        _, StsServos = _bp._load_bam()
        m, ts = self.m, self.ts
        d = mujoco.MjData(m)
        rend = {}
        if render:
            for cname, h, w in (("main", 480, 640), ("close", 480, 640),
                                (record_cams[0], 240, 320), (record_cams[1], 240, 320)):
                rend[cname] = mujoco.Renderer(m, height=h, width=w)
        d.qpos[:6] = self.q0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        servos = StsServos(m, d, list(JOINTS), "m6")
        servos.reset()
        self.servos = servos
        n_steps = int(round(self.t / ts))
        sample_every = max(1, int(round(0.05 / ts)))
        gate_every = max(1, int(round(0.01 / ts)))
        frame_every = max(1, int(round((1 / 30) / ts)))
        log = {"t": [], "tcp": [], "q": [], "bodies": {b: [] for b in self.bodies},
               "quats": {b: [] for b in self.bodies}}
        rec = {"state": [], "action": [], "camA": [], "camB": [], "wide": [], "close": []}
        self.duty_log = []
        self.ctrl_log = []
        ctl = self.fa_controller
        deg = ObsDegrader(self.ladder_rung, self.ladder_seed)
        grip_cmd = None
        begun = False
        for i in range(n_steps + 1):
            t = i * ts
            ctrl = target_q(self.segs, t)
            in_window = (ctl is not None and self.fa_t0 is not None
                         and self.fa_t0 - 1e-9 <= t < self.fa_t1 - 1e-9)
            applied = ctrl.copy()
            if in_window:
                if not begun:
                    ctl.begin(ctrl[GRIP_IDX])
                    grip_cmd = ctl.cmd
                    begun = True
                applied[GRIP_IDX] = grip_cmd
            # --- BAM actuation (as BamPilot) ---
            for j, q in zip(JOINTS, applied):
                servos.set(j, float(q))
            servos.update()
            # --- end BAM actuation ---
            if i % gate_every == 0:
                mujoco.mj_forward(m, d)
                self.gate.sample(d, applied)
                duty = self._duty()
                self.duty_log.append((round(t, 3), round(duty, 4)))
                del_q, del_duty = deg.feed(
                    float(d.qpos[self.gate.grip_qadr]), duty)
                if in_window:
                    if isinstance(ctl, OracleController):
                        fb = self._oracle_feedback(d)
                        new_cmd = ctl.step(fb, t)
                        row = self.gate.rows[-1]
                        self.ctrl_log.append(dict(
                            t=round(t, 3), phase=ctl.state,
                            cmd=round(float(applied[GRIP_IDX]), 4),
                            q=round(float(d.qpos[self.gate.grip_qadr]), 4),
                            duty=round(duty, 4),
                            F_N=round(fb["F"], 2), F1_N=round(fb["F1"], 2),
                            F_gate_N=round(row["mug"]["fixed"]
                                           + row["mug"]["moving"], 2)))
                    else:  # RealisticController: {q, duty} ONLY —
                    # the DELIVERED (ladder-degraded) sample; the raw
                    # duty is scorer-side telemetry (duty_log) and is
                    # logged here as duty_raw for comparison only
                        fb = dict(q=del_q, duty=del_duty)
                        new_cmd = ctl.step(fb, t)
                        row = self.gate.rows[-1]
                        self.ctrl_log.append(dict(
                            t=round(t, 3), phase=ctl.state,
                            cmd=round(float(applied[GRIP_IDX]), 4),
                            q=round(fb["q"], 4), duty=round(fb["duty"], 4),
                            duty_raw=round(duty, 4),
                            F_gate_N=round(row["mug"]["fixed"]
                                           + row["mug"]["moving"], 2)))
                    grip_cmd = new_cmd
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
                rec["action"].append(applied.copy())
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
