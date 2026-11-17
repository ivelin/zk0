"""One run of the certified noise-hardened realistic pilot (SIM-2).

Subprocess entry for Stage 7 (trajectory admission): configures one
run of the vendored certified machinery (see sim/hardened/
PROVENANCE.md) — strategy rim|pinch, the HardenedRealisticController
at a declared telemetry rung, a declared friction/mass envelope
draw — and writes result.json (+ episode.npz when recording).

The configuration and the Tier 1 computation mirror the certified
matrix runner (zk0-rung3-rematch-20261117/code/run_rematch.py)
field for field; DetailRecorder below is copied from it verbatim.
What this runner adds: the aim offset and the mass draw are
parameters (admission varies them), and the recorded episode
(state/action/front/top at 30 fps) is saved as an .npz instead of
rendered videos.

Environment set by the caller: ZK0_HOME (asset staging). This
module sets the vendored machinery's own env (ZK0_SERVO=bam,
ZK0_MUG_MU, ZK0_MUG_PKG, ZK0_HARDENED_MODEL_DIR, ZK0_MUG_MASS when
a mass draw is given) before importing it — the machinery reads
them at import time.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

HARDENED_DIR = Path(__file__).parent / "hardened"
GZ = {"rim": 0.077, "pinch": 0.058}
CLOSE_REQ = -0.105
CLOSE_DUR = 2.5
OPEN_DUR = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["rim", "pinch"], required=True)
    ap.add_argument("--aim-dx", type=float, default=0.0)
    ap.add_argument("--mu", type=float, required=True)
    ap.add_argument("--mass", type=float, default=None,
                    help="mug mass draw (kg); omit for the certified "
                         "nominal (params.json, MEASURED 0.118 kg)")
    ap.add_argument("--rung", type=int, choices=[0, 1, 2, 3], default=3)
    ap.add_argument("--record", action="store_true",
                    help="record the 30 fps episode (state/action/"
                         "front/top) to episode.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Asset + model staging (established package patterns): the
    # harness writes compiled scene XML into its model dir, so it
    # gets a private writable copy of the staged arm model.
    from . import assets
    if os.environ.get("ZK0_HARDENED_MODEL_DIR"):
        # Caller-provided shared work model (Stage 7 runs many
        # draws sequentially against one copy).
        model_copy = Path(os.environ["ZK0_HARDENED_MODEL_DIR"])
    else:
        model_copy = out / "model"
        if model_copy.exists():
            shutil.rmtree(model_copy)
        shutil.copytree(assets.model_dir("realistic"), model_copy)
        os.environ["ZK0_HARDENED_MODEL_DIR"] = str(model_copy)

    os.environ["ZK0_SERVO"] = "bam"
    os.environ["ZK0_MUG_MU"] = repr(args.mu)
    os.environ["ZK0_MUG_PKG"] = str(assets.mug_dir())
    if args.mass is not None:
        os.environ["ZK0_MUG_MASS"] = repr(args.mass)
    else:
        os.environ.pop("ZK0_MUG_MASS", None)

    sys.path.insert(0, str(HARDENED_DIR))
    import numpy as np
    import mujoco

    import harness as _harness
    from harness import settled as log_settled, arm_retreated
    import bam_pilot as _bp
    if _bp.SERVO == "bam":
        _harness.load_model = _bp.load_model_bam
    settle_probe = _bp.pick_settle()
    import scenes as S
    import wall_pinch as WP
    from grasp_gate import evaluate_cycle, GateRecorder
    from hardened_pilot import (ForceAwareBamPilot,
                                HardenedRealisticController)

    class DetailRecorder(GateRecorder):
        """GateRecorder + per-contact detail log (mug frame) at
        100 Hz — copied verbatim from the certified run_rematch.py
        (the rim-hook test's instrumentation pattern)."""
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            m = self.m
            self.mug_body = self.body_id["mug"]
            self.mug_qadr = self.bqadr["mug"]
            self.geom_names = [mujoco.mj_id2name(
                m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"
                for i in range(m.ngeom)]
            self.detail = []
            self._f6 = np.zeros(6)

        def sample(self, d, ctrl):
            super().sample(d, ctrl)
            m = self.m
            qa = self.mug_qadr
            mpos = d.qpos[qa:qa + 3].copy()
            mquat = d.qpos[qa + 3:qa + 7].copy()
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, mquat)
            R = R.reshape(3, 3)
            gb = m.geom_bodyid
            cons = []
            for ci in range(d.ncon):
                c = d.contact[ci]
                b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
                if self.mug_body not in (b1, b2):
                    continue
                bother = b2 if b1 == self.mug_body else b1
                gmug = c.geom1 if b1 == self.mug_body else c.geom2
                grob = c.geom2 if b1 == self.mug_body else c.geom1
                if bother == self.grip_body:
                    side = "moving"
                elif bother == self.fixed_body:
                    side = "fixed"
                elif bother in self.robot_bodies:
                    side = "robot"
                else:
                    continue
                mujoco.mj_contactForce(m, d, ci, self._f6)
                nf = abs(float(self._f6[0]))
                tf = float(np.hypot(self._f6[1], self._f6[2]))
                pw = np.array(c.pos, float)
                pm = R.T @ (pw - mpos)
                cons.append(dict(
                    side=side, geom_mug=self.geom_names[int(gmug)],
                    geom_robot=self.geom_names[int(grob)],
                    nf=round(nf, 3), tf=round(tf, 3),
                    pos_mug_mm=[round(float(v) * 1000, 1) for v in pm],
                    dist_mm=round(float(c.dist) * 1000, 3)))
            self.detail.append(dict(t=round(float(d.time), 3),
                                    contacts=cons))

    cal = json.loads((HARDENED_DIR / "calibration.json").read_text())
    gz = GZ[args.strategy]
    t_start = time.time()
    xml = WP.build_xml()
    sp = settle_probe(xml, {"mug": ((WP.MUG_XY[0], WP.MUG_XY[1],
                                     WP.COM_Z), WP.IDENT)}, dur=2.5)
    settled_pos = sp["mug"]["final_pos"]
    P = ForceAwareBamPilot(xml, bodies=["mug"], lookat=(0.26, 0.0, 0.03))
    P.ladder_rung = args.rung
    P.gate = DetailRecorder(P.m, P.bodies, P.site_id)
    mug_xy = (settled_pos[0], settled_pos[1])
    WP.plan_wall_pinch(P, mug_xy, gz, CLOSE_REQ, aim_dx=args.aim_dx)
    t_close = P.events["close_end_mug"]
    t_open = P.events["open_end_mug"]
    jrange = P.gate.jnt_range
    c = cal[args.strategy]
    ctl = HardenedRealisticController(jrange, c["d_lo"], c["d_hi"])
    ctl_meta = dict(variant="realistic",
                    law="HardenedRealisticController",
                    duty_band=[ctl.d_lo, ctl.d_hi],
                    calibration={k: c[k] for k in
                                 ("d_lo", "d_hi", "gz", "mug_mu")},
                    theta_c=ctl.THETA_C, n_contact=ctl.N_C,
                    entry_margin=ctl.ENTRY_MARGIN, k_entry=ctl.K_B,
                    creep_theta=ctl.THETA_CREEP, creep_window=ctl.W,
                    sigma_duty=ctl.SIG_DUTY, sigma_q=ctl.SIG_Q)
    P.fa_controller = ctl
    P.fa_t0 = t_close - CLOSE_DUR
    P.fa_t1 = t_open - OPEN_DUR
    log, rec = P.run(record_cams=("front", "top"),
                     render=bool(args.record))
    wall_s = round(time.time() - t_start, 2)

    contained = S._contained(log, "mug", WP.BIN[0], WP.BIN[1],
                             WP.BIN[2], WP.BIN[3], WP.BIN[4])
    s_mm = log_settled(log, "mug") * 1000
    max_single = max([cc["nf"] for s in P.gate.detail
                      for cc in s["contacts"]], default=0.0)
    cmds = np.array([r["grip_cmd"] for r in P.gate.rows])
    lo_r, hi_r = jrange
    # Tolerance 5e-5 per the certified preregistration AMENDMENT
    # (the model's own GRIP_OPEN constant sits 3.3e-5 above the
    # rounded joint bound; a zero-tolerance check fails every
    # certified baseline too).
    legal = bool(cmds.min() >= lo_r - 5e-5
                 and cmds.max() <= hi_r + 5e-5)
    tier1 = dict(contained=bool(contained),
                 settled_mm=round(float(s_mm), 2),
                 max_single_contact_N=round(float(max_single), 2),
                 damage_free=bool(max_single < 25.0),
                 legal_actuation=legal,
                 pass_=bool(contained and s_mm < 2.0
                            and max_single < 25.0 and legal))
    g = evaluate_cycle(P.gate.rows, "mug", t_close, t_open, CLOSE_REQ,
                       P.gate.mass["mug"], jrange, _bp.force_limit(P))
    tier2 = dict(status="PROVISIONAL (SSOT decision 53 — recorded "
                        "per draw, never gates admission)",
                 gate_pass=bool(g["pass"]),
                 failed_gates=[k for k, v in g["gates"].items()
                               if not v["pass"]])
    result = {
        "runner": "zk0_node.sim.hardened_run (vendored certified "
                  "machinery, sim/hardened/PROVENANCE.md)",
        "strategy": args.strategy, "gz": gz, "close_req": CLOSE_REQ,
        "aim_dx_m": args.aim_dx,
        "servo": _bp.SERVO,
        "mug_mu": WP.MU, "mug_mass_kg": WP.MASS,
        "ladder": dict(rung=args.rung, seed=20261116,
                       pipeline="noise(rung>=3) -> quantize(rung>=1) "
                                "-> delay-1-tick(rung>=2)",
                       sigma_q=0.005 if args.rung >= 3 else None,
                       sigma_duty=0.01 if args.rung >= 3 else None),
        "settle_probe": {k: sp["mug"][k] for k in
                         ("disp_mm", "tilt_deg", "final_pos")},
        "controller": ctl_meta,
        "controller_summary": dict(
            state_final=ctl.state,
            cmd_hold=None if ctl.cmd_hold is None
            else round(ctl.cmd_hold, 4),
            floor_limited=bool(ctl.floor_limited),
            tighten_events=int(ctl.tighten_events),
            ease_events=int(ctl.ease_events),
            creep_events=int(ctl.creep_events),
            contact_t=ctl.contact_t, entry_t=ctl.entry_t),
        "window": {"fa_t0": round(P.fa_t0, 2),
                   "fa_t1": round(P.fa_t1, 2),
                   "close_end": round(t_close, 2),
                   "open_end": round(t_open, 2)},
        "evaluation_original_scorer": dict(
            contained=bool(contained), settled_mm=round(float(s_mm), 2),
            final=np.round(log["bodies"]["mug"][-1], 4).tolist(),
            arm_retreated=arm_retreated(log),
            ok=bool(contained and s_mm < 2.0 and arm_retreated(log))),
        "tier1": tier1,
        "tier2": tier2,
        "gate_force_limit_nm": _bp.force_limit(P),
        "wall_clock_s": wall_s,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1,
                                                default=str))
    if args.record:
        np.savez_compressed(
            out / "episode.npz",
            state_rad=np.asarray(rec["state"]),
            action_rad=np.asarray(rec["action"]),
            cam_front=np.asarray(rec["camA"]),
            cam_top=np.asarray(rec["camB"]))
        result["episode"] = dict(
            frames=int(len(rec["state"])), fps=30,
            file="episode.npz")
        (out / "result.json").write_text(json.dumps(result, indent=1,
                                                    default=str))
    print(json.dumps({"tier1": tier1, "wall_clock_s": wall_s}))
    print("HARDENED_RUN_DONE")


if __name__ == "__main__":
    main()
