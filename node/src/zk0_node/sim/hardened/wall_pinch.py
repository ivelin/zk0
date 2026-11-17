"""Wall-pinch grasp test for the collection YCB mug (2026-11-16).

Strategy under test: one finger INSIDE the mug cavity, one finger OUTSIDE
the wall, clamp the wall itself (~4.4 mm) instead of spanning the body
(O 93.1 mm). Assignment from the FK probe: at the grasp orientation the
FIXED jaw sits on -X (base side), the MOVING jaw on +X. Targeting the
mug's -X wall (opposite the handle, which faces +X) puts the MOVING jaw
inside the cavity and the FIXED jaw outside the wall; closing sweeps the
moving finger outward through the cavity and clamps the wall against the
fixed finger.

Scene, table, bin, mug spawn and asset consumption mirror the S6b
collection retest (retest_s6b.py) exactly, minus the apple.
Modes: probe (descend + slow close, dump contact/force vs grip q),
       run   (full pick-carry-place, gate + original scorer, videos).
"""
import os, json, glob, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import sys
# [slice-4 vendoring patch — see PROVENANCE.md] paths were absolute Spark workdir constants;
# the vendored modules live together in this directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from harness import (Pilot, settle_probe, mesh_body, walls_xml, scene_xml,
                     settled as log_settled, arm_retreated, HOME_Q, GRIP_OPEN)
from grasp_gate import evaluate_cycle
import scenes as S
import harness as _harness
import bam_pilot as _bp
if _bp.SERVO == "bam":
    _harness.load_model = _bp.load_model_bam
Pilot = _bp.pick_pilot()
settle_probe = _bp.pick_settle()

# [slice-4 vendoring patch — see PROVENANCE.md] PKG was an absolute Spark workdir path; the
# package stages the byte-identical asset package itself
# (md5-verified, PROVENANCE.md) and passes it via the env.
PKG = os.environ["ZK0_MUG_PKG"]
PAR = json.load(open(PKG + "/params.json"))
MASS = PAR["mass_kg"]; DI = PAR["diaginertia_kgm2"]; COM_Z = PAR["com_mm"][2] / 1000.0
HULLS = sorted(glob.glob(PKG + "/hull_*.obj"))
IDENT = np.array([1.0, 0, 0, 0])
BIN = (0.30, -0.02, 0.058, 0.058, 0.11)
WALL_R = 0.0444        # wall centreline radius (outer 46.55 / inner ~42.2 mm)
MUG_XY = (0.24, 0.135)
MU = float(os.environ.get("ZK0_MUG_MU", "0.35"))   # mug contact friction (friction sweep)
# [slice-4 vendoring patch — see PROVENANCE.md] admission envelope draw: ZK0_MUG_MASS scales
# the mug mass (diaginertia scaled with it). Default: unset.
if os.environ.get("ZK0_MUG_MASS"):
    _draw_mass = float(os.environ["ZK0_MUG_MASS"])
    DI = [v * _draw_mass / MASS for v in DI]
    MASS = _draw_mass

table = dict(half=(0.48, 0.33), thick=0.035, height=0.90, mu=0.35,
             rgba="0.88 0.88 0.86 1", wall_rgba="0.86 0.84 0.80 1",
             floor_rgba="0.52 0.50 0.46 1", light_pos="0.5 -0.3 1.6",
             light_dir="-0.2 0.1 -1")


def build_xml():
    a1, b1 = mesh_body("mug", PKG + "/visual.obj", (MUG_XY[0], MUG_XY[1], COM_Z), IDENT,
                       MASS, DI, "1 1 1 1", MU, texfile=PKG + "/texture_map.png",
                       hull_files=HULLS, geom_pos=(0.0, 0.0, -COM_Z))
    _disc = (f'<geom name="mug_floor" type="cylinder" size="0.041 0.002" '
             f'pos="0 0 {0.001 - COM_Z:.5f}" friction="{MU} 0.005 0.0001" '
             f'solref="0.002 1" solimp="0.99 0.999 0.001" rgba="0 0 0 0"/>')
    b1 = b1.replace("</body>", _disc + "</body>")
    static = walls_xml("bin", BIN[0], BIN[1], BIN[2], BIN[3], 0.005, BIN[4],
                       "0.62 0.66 0.70 1", 0.35)
    return scene_xml(table, a1, b1, static)


def plan_wall_pinch(P, mug_xy, gz, close_val, aim_dx=0.0, carry_extra=0.125,
                    approach_grip=0.30, close_dur=2.5):
    """Wall-pinch pick-and-place. mug_xy = settled mug centre. The TCP is
    targeted over the -X wall centreline (aim_dx shifts the aim along X,
    the straddle axis, for robustness runs). Place keeps the same grasp
    offset so the mug body centres on the bin."""
    wx = mug_xy[0] - WALL_R + aim_dx
    wy = mug_xy[1]
    px, py = BIN[0] - WALL_R, BIN[1]     # place TCP: same offset from bin centre
    P.cycles.append(dict(body="mug", tag="mug", close_cmd=close_val,
                         close_event="close_end_mug", open_event="open_end_mug"))
    P.goto((wx, wy, gz + 0.115), 1.2, grip=GRIP_OPEN)
    P.goto((wx, wy, gz), 1.2, orient=True, grip=approach_grip)
    P.grip_to(close_val, close_dur, event="close_end_mug")
    P.dwell(0.4)
    P.goto((wx, wy, gz + carry_extra), 2.0, grip=close_val)
    P.goto((px, py, gz + carry_extra), 1.8, grip=close_val)
    P.goto((px, py, gz + 0.002), 2.2, grip=close_val, orient=True)
    P.grip_to(GRIP_OPEN, 1.0, event="open_end_mug")
    P.dwell(0.8)
    P.goto((px, py, gz + carry_extra), 1.5, grip=GRIP_OPEN)
    P.home(); P.dwell(1.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "run"], default="run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gz", type=float, default=0.058)
    ap.add_argument("--close", type=float, default=-0.16)
    ap.add_argument("--aim-dx", type=float, default=0.0, help="aim offset along X, metres")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    xml = build_xml()
    sp = settle_probe(xml, {"mug": ((MUG_XY[0], MUG_XY[1], COM_Z), IDENT)}, dur=2.5)
    settled_pos = sp["mug"]["final_pos"]
    out = {"mode": args.mode, "gz": args.gz, "close_cmd": args.close,
           "aim_dx_m": args.aim_dx,
           "settle_probe": {k: sp["mug"][k] for k in ("disp_mm", "tilt_deg", "final_pos")}}
    P = Pilot(xml, bodies=["mug"], lookat=(0.26, 0.0, 0.03))
    mug_xy = (settled_pos[0], settled_pos[1])

    if args.mode == "probe":
        wx, wy = mug_xy[0] - WALL_R, mug_xy[1]
        P.goto((wx, wy, args.gz + 0.115), 1.2, grip=GRIP_OPEN)
        P.goto((wx, wy, args.gz), 1.2, orient=True, grip=0.30)
        P.grip_to(-0.1745, 6.0, event="close_end_mug")   # slow close to the legal floor
        P.dwell(1.0, event="open_end_mug")
        P.cycles.append(dict(body="mug", tag="mug", close_cmd=-0.1745,
                             close_event="close_end_mug", open_event="open_end_mug"))
        log, rec = P.run(render=False)
        rows = P.gate.rows
        # summarise the close: grip q, actuator force, per-jaw contact force
        series = []
        for r in rows:
            if r["t"] < P.events["close_end_mug"] - 6.5:
                continue
            series.append(dict(t=round(r["t"], 2), grip_q=round(r["grip_q"], 4),
                               grip_force=round(r["grip_force"], 2),
                               fixed_N=round(r["mug"]["fixed"], 2),
                               moving_N=round(r["mug"]["moving"], 2),
                               support_N=round(r["mug"]["support"], 2),
                               mug_z=round(float(r["mug"]["pos"][2]), 4)))
        out["close_series_tail"] = series[-40:]
        # first sample where each jaw's contact force exceeds 0.5 N
        for jaw in ("fixed", "moving"):
            hit = next((s for s in series if s[jaw + "_N"] > 0.5), None)
            out[f"first_{jaw}_contact"] = hit
        out["final_grip_q"] = series[-1]["grip_q"] if series else None
        out["ik_log"] = P.ik_log
    else:
        plan_wall_pinch(P, mug_xy, args.gz, args.close, aim_dx=args.aim_dx)
        log, rec = P.run(record_cams=("front", "top"), render=args.render)
        if args.render:
            import imageio.v2 as imageio
            for key, cam in (("wide", "wide"), ("close", "close")):
                with imageio.get_writer(f"{args.out}/wallpinch_{key}.mp4", fps=30,
                                        codec="libx264", quality=8) as w:
                    for f in rec[cam]:
                        w.append_data(f)
        c = S._contained(log, "mug", BIN[0], BIN[1], BIN[2], BIN[3], BIN[4])
        s_mm = log_settled(log, "mug") * 1000
        out["evaluation_original_scorer"] = dict(
            contained=bool(c), settled_mm=round(s_mm, 2),
            final=np.round(log["bodies"]["mug"][-1], 4).tolist(),
            arm_retreated=arm_retreated(log),
            ok=bool(c and s_mm < 2.0 and arm_retreated(log)))
        cyc = P.cycles[0]
        g = evaluate_cycle(P.gate.rows, "mug", P.events[cyc["close_event"]],
                           P.events[cyc["open_event"]], cyc["close_cmd"],
                           P.gate.mass["mug"], P.gate.jnt_range, _bp.force_limit(P))
        g["original_body_credit"] = bool(c and s_mm < 2.0)
        g["gated_body_credit"] = bool(g["original_body_credit"] and g["pass"])
        out["cycle_gate"] = g
        out["gate_force_limit_nm"] = _bp.force_limit(P)
        out["servo"] = _bp.SERVO
        out["mug_mu"] = MU
        out["ik_log"] = P.ik_log
    json.dump(out, open(f"{args.out}/result.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str), flush=True)
    print("WALLPINCH_DONE")


if __name__ == "__main__":
    main()
