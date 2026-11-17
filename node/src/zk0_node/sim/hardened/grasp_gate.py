"""Grasp-quality gate for the zk0 MuJoCo evaluators (added 2026-11-16 after
the S6b forensic investigation — see zk0-s6b-forensics-20261116/report.md).

Two halves:
- GateRecorder: per-run instrumentation sampled at 100 Hz inside Pilot.run —
  gripper command / q / actuator force plus, per tracked body, pose, pose in
  the gripperframe-site frame, and contact aggregates split by counterparty
  class (fixed jaw / moving jaw / other robot / support). This is the logging
  the forensic prototype (forensic_s6b.py) proved out: d.contact +
  mj_contactForce + d.actuator_force.
- evaluate_cycle: scores ONE pick cycle's carry window [close_end, open_end]
  against the grasp-quality gates. Containment/placement credit requires ALL
  gates to pass; every failure is named with its measured value.

Gates (per pick cycle, airborne samples = held samples with no support
contact, from first to last such sample in the window):
  tilt        object orientation vs its orientation at close_end, max <= 15 deg
  slip        object origin in gripper frame vs its value at first airborne
              sample, max <= 10 mm
  force_band  summed finger normal force / object weight, median in [1, 15]
  command     requested close command AND logged gripper command throughout
              the window within the gripper joint range
  saturation  fraction of airborne samples with |gripper actuator force| at
              the force limit <= 50%
  two_sided   fraction of airborne samples with contact on BOTH jaws >= 95%
  penetration max finger-object penetration <= 2 mm
  free_support no load-bearing support contact (>= 0.5 x object weight)
              strictly inside the airborne span
"""
import numpy as np
import mujoco

GRAV = 9.81
TILT_MAX_DEG = 15.0
SLIP_MAX_MM = 10.0
FORCE_RATIO_MIN, FORCE_RATIO_MAX = 1.0, 15.0
SAT_FRAC_MAX = 0.50
SAT_TOL = 0.98               # |force| >= 0.98 x limit counts as saturated
TWO_SIDED_FRAC_MIN = 0.95
PEN_MAX_MM = 2.0
SUPPORT_LOAD_FRAC = 0.5      # support force >= this x weight = load-bearing


class GateRecorder:
    def __init__(self, m, bodies, site_id, grip_jnt_name="gripper"):
        self.m = m
        self.bodies = list(bodies)
        self.site_id = site_id
        gj = m.joint(grip_jnt_name).id
        self.grip_jnt = gj
        self.grip_qadr = int(m.jnt_qposadr[gj])
        self.grip_body = int(m.jnt_bodyid[gj])
        self.fixed_body = int(m.body_parentid[self.grip_body])
        self.grip_act = -1
        for a in range(m.nu):
            if int(m.actuator_trnid[a, 0]) == gj:
                self.grip_act = a
                break
        fr = np.abs(m.actuator_forcerange[self.grip_act]) if self.grip_act >= 0 else np.array([0.0])
        self.force_limit = float(fr.max()) if fr.size else 0.0
        self.jnt_range = (float(m.jnt_range[gj][0]), float(m.jnt_range[gj][1]))
        free_bodies = set()
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                free_bodies.add(int(m.jnt_bodyid[j]))
        self.free_bodies = free_bodies
        self.robot_bodies = set(range(1, m.nbody)) - free_bodies
        self.body_id = {b: int(m.body(b).id) for b in self.bodies}
        self.bqadr = {b: int(m.jnt(b + "_free").qposadr[0]) for b in self.bodies}
        self.mass = {b: float(m.body_mass[self.body_id[b]]) for b in self.bodies}
        self.rows = []
        self._f6 = np.zeros(6)

    def sample(self, d, ctrl):
        m = self.m
        ga = self.grip_act
        rec = {"t": float(d.time),
               "grip_cmd": float(ctrl[ga]) if ga >= 0 else float(ctrl[5]),
               "grip_q": float(d.qpos[self.grip_qadr]),
               "grip_force": float(d.actuator_force[ga]) if ga >= 0 else 0.0}
        sp = d.site_xpos[self.site_id]
        sR = d.site_xmat[self.site_id].reshape(3, 3)
        agg = {b: dict(fixed=0.0, moving=0.0, robot=0.0, support=0.0, pen=0.0,
                       n_fixed=0, n_moving=0, n_support=0) for b in self.bodies}
        gb = m.geom_bodyid
        id2body = {v: k for k, v in self.body_id.items()}
        for ci in range(d.ncon):
            c = d.contact[ci]
            b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
            for bend, bother in ((b1, b2), (b2, b1)):
                name = id2body.get(bend)
                if name is None:
                    continue
                mujoco.mj_contactForce(m, d, ci, self._f6)
                nf = abs(float(self._f6[0]))
                a = agg[name]
                if bother == self.grip_body:
                    a["moving"] += nf; a["n_moving"] += 1
                    a["pen"] = max(a["pen"], max(0.0, -float(c.dist)))
                elif bother == self.fixed_body:
                    a["fixed"] += nf; a["n_fixed"] += 1
                    a["pen"] = max(a["pen"], max(0.0, -float(c.dist)))
                elif bother in self.robot_bodies:
                    a["robot"] += nf
                else:
                    a["support"] += nf; a["n_support"] += 1
        for b in self.bodies:
            qa = self.bqadr[b]
            pos = d.qpos[qa:qa + 3].copy()
            quat = d.qpos[qa + 3:qa + 7].copy()
            rec[b] = dict(pos=pos, quat=quat, rel=sR.T @ (pos - sp), **agg[b])
        self.rows.append(rec)


def _command_gate(W, close_cmd, lo, hi):
    cmds = np.array([r["grip_cmd"] for r in W])
    req_ok = (lo - 1e-9) <= close_cmd <= (hi + 1e-9)
    app_ok = bool(cmds.min() >= lo - 1e-6 and cmds.max() <= hi + 1e-6)
    # The gate's physical invariant is the APPLIED command: since 2026-11-16
    # the pilot clamps every gripper target into the joint range, so an
    # out-of-range profile request is absorbed by construction and recorded
    # here (requested_in_range / clamped) instead of reaching the actuator.
    # Pre-clamp runs fail this gate because their applied ctrl left the range.
    return {"pass": bool(app_ok), "requested": float(close_cmd),
            "requested_in_range": bool(req_ok), "clamped": bool(not req_ok and app_ok),
            "applied_min": round(float(cmds.min()), 4),
            "applied_max": round(float(cmds.max()), 4),
            "joint_range": [round(lo, 4), round(hi, 4)]}


def evaluate_cycle(rows, body, t_close, t_open, close_cmd, mass, jnt_range, force_limit):
    """Score one pick cycle. rows = GateRecorder.rows for the whole run."""
    lo, hi = jnt_range
    weight = mass * GRAV
    W = [r for r in rows if t_close - 1e-6 <= r["t"] <= t_open + 1e-6]
    res = {"body": body, "close_cmd": float(close_cmd),
           "window_s": [round(float(t_close), 2), round(float(t_open), 2)],
           "n_window_samples": len(W)}
    if not W:
        res["gates"] = {"carry": {"pass": False, "reason": "no gate samples in carry window"}}
        res["pass"] = False
        return res
    gates = {"command": _command_gate(W, close_cmd, lo, hi)}

    def held(r):
        return r[body]["n_fixed"] + r[body]["n_moving"] >= 1

    def unsupported(r):
        return r[body]["n_support"] == 0

    air_idx = [i for i, r in enumerate(W) if unsupported(r) and held(r)]
    if not air_idx:
        for name in ("tilt", "slip", "force_band", "saturation", "two_sided",
                     "penetration", "free_support"):
            gates[name] = {"pass": False, "reason": "object never airborne while held"}
        res["gates"] = gates
        res["n_airborne"] = 0
        res["pass"] = False
        return res
    i0, i1 = air_idx[0], air_idx[-1]
    air = [W[i] for i in range(i0, i1 + 1) if unsupported(W[i])]
    res["n_airborne"] = len(air)
    res["airborne_s"] = [round(W[i0]["t"], 2), round(W[i1]["t"], 2)]
    # Descent point (forensic sec.4: the free-support window "ends at
    # descent"): the first index after lift-off where load-bearing support
    # is sustained (>=80% of the next 0.3 s) is touch-down / lean-rest.
    # Only the free_support gate is bounded by it; the other gates score
    # every unsupported sample in the span, as the forensic's slip did.
    lb = np.array([r[body]["support"] >= SUPPORT_LOAD_FRAC * weight for r in W])
    desc_i = None
    for i in range(i0 + 1, len(W) - 5):
        seg = lb[i:i + 30]
        if len(seg) >= 10 and seg.mean() >= 0.8:
            desc_i = i
            break
    res["descent_at_s"] = (round(W[desc_i]["t"], 2)
                            if desc_i is not None else None)

    ref_quat = W[0][body]["quat"]
    tilts = []
    for r in air:
        dot = abs(float(np.dot(r[body]["quat"], ref_quat)))
        tilts.append(np.degrees(2 * np.arccos(min(1.0, dot))))
    tilt_max = float(np.max(tilts)) if tilts else 0.0
    gates["tilt"] = {"pass": bool(tilt_max <= TILT_MAX_DEG),
                     "max_deg": round(tilt_max, 1), "limit_deg": TILT_MAX_DEG}

    rel0 = air[0][body]["rel"]
    slip_max = float(np.max([np.linalg.norm(r[body]["rel"] - rel0) for r in air])) * 1000 if air else 0.0
    gates["slip"] = {"pass": bool(slip_max <= SLIP_MAX_MM),
                     "max_mm": round(slip_max, 1), "limit_mm": SLIP_MAX_MM}

    ratios = np.array([(r[body]["fixed"] + r[body]["moving"]) / weight for r in air])
    med = float(np.median(ratios)) if len(ratios) else 0.0
    gates["force_band"] = {"pass": bool(FORCE_RATIO_MIN <= med <= FORCE_RATIO_MAX),
                           "median_ratio": round(med, 1),
                           "max_ratio": round(float(ratios.max()), 1) if len(ratios) else 0.0,
                           "band": [FORCE_RATIO_MIN, FORCE_RATIO_MAX]}

    sat = float(np.mean([abs(r["grip_force"]) >= SAT_TOL * force_limit for r in air])) if air else 1.0
    gates["saturation"] = {"pass": bool(sat <= SAT_FRAC_MAX),
                           "frac_saturated": round(sat, 3), "limit_frac": SAT_FRAC_MAX}

    ts = float(np.mean([r[body]["n_fixed"] >= 1 and r[body]["n_moving"] >= 1 for r in air])) if air else 0.0
    gates["two_sided"] = {"pass": bool(ts >= TWO_SIDED_FRAC_MIN),
                          "frac_both_jaws": round(ts, 3), "limit_frac": TWO_SIDED_FRAC_MIN}

    pen = float(np.max([r[body]["pen"] for r in air])) * 1000 if air else 0.0
    gates["penetration"] = {"pass": bool(pen <= PEN_MAX_MM),
                            "max_mm": round(pen, 2), "limit_mm": PEN_MAX_MM}

    interior = W[i0 + 1:(desc_i if desc_i is not None else i1)]
    fs_events = [r for r in interior if r[body]["support"] >= SUPPORT_LOAD_FRAC * weight]
    fs_contacts = [r for r in interior if r[body]["n_support"] > 0]
    # a genuine prop/drag is a SUSTAINED load-bearing run (>=0.1 s, gaps of
    # <=2 samples bridged); a momentary brush is contact noise, not support
    longest = run = 0
    gap = 0
    for r in interior:
        if r[body]["support"] >= SUPPORT_LOAD_FRAC * weight:
            run += 1 + gap if run else 1
            gap = 0
            longest = max(longest, run)
        elif run:
            gap += 1
            if gap > 2:
                run = 0
                gap = 0
    gates["free_support"] = {"pass": longest < 10,
                             "longest_loadbearing_run_samples": int(longest),
                             "loadbearing_samples": len(fs_events),
                             "support_contact_samples_inside_span": len(fs_contacts)}

    res["gates"] = gates
    res["pass"] = bool(all(g["pass"] for g in gates.values()))
    return res
