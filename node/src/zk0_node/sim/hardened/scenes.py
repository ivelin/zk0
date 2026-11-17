"""Scene definitions for the description->scene test. Each scene is built
FROM ITS DESCRIPTION (see SUCCESS_DEFS.md for the pre-registered verdicts):
the builder chooses table, backdrop, objects (library first, spec-layer
second, generation only where a class exists nowhere usable), layout and
camera pair, and a ground-truth-guided pilot plan.
"""
import numpy as np
from harness import (walls_xml, tray_xml, scene_xml, pick_place,
                     settled as log_settled, tilt_deg, arm_retreated, quat_from_euler)
import objects as O

# grasp profiles: (grasp_z, dx, close) — from the battery + grasp probes
PROF = {
    "cube":        dict(gz=0.020, dx=-0.01, close=-0.10),
    "mug_ycb":     dict(gz=0.055, dx=0.0, close=-0.5),
    "mug_triposg": dict(gz=0.060, dx=0.015, close=-0.5, carry_dur=2.8),
    "apple_ycb":   dict(gz=0.034, dx=0.0, close=-0.85, close_dur=3.0, approach_grip=0.35),
    "apple_triposg": dict(gz=0.034, dx=0.0, close=-0.5),
    "pear":        dict(gz=0.034, dx=0.0, close=-0.45),
    "banana":      dict(gz=0.017, dx=0.0, close=-0.4),
    "plum":        dict(gz=0.024, dx=0.0, close=-0.6),
    "sponge":      dict(gz=0.012, dx=0.0, close=-0.35),
    "marker":      dict(gz=0.011, dx=0.0, close=-0.6, close_dur=3.0),
    "screwdriver": dict(gz=0.019, dx=0.0, close=-0.6, close_dur=3.0),
    "foam_brick":  dict(gz=0.035, dx=0.0, close=-0.3, approach_grip=0.6),
    "tennis_ball": dict(gz=0.028, dx=0.0, close=-0.85, close_dur=3.0, approach_grip=0.4),
}

def _pp(P, settled, body, prof, place_xy, place_tcp_z, carry_extra=0.125, tag="",
        grasp_off=(0.0, 0.0), **kw):
    """pick_place wrapper: grasp point may be offset from the body origin
    (elongated objects); the place grasp point keeps the same offset so the
    BODY lands on place_xy."""
    sx, sy = settled[body][0], settled[body][1]
    gx, gy = sx + grasp_off[0], sy + grasp_off[1]
    px, py = place_xy[0] + grasp_off[0], place_xy[1] + grasp_off[1]
    pick_place(P, (gx, gy), prof["gz"], prof["dx"], prof["close"],
               prof["gz"] + carry_extra, (px, py), place_tcp_z,
               close_dur=prof.get("close_dur", 2.5), approach_grip=prof.get("approach_grip"),
               carry_dur=prof.get("carry_dur", 1.8), tag=tag, body=body, **kw)

def _contained(log, body, cx, cy, hx, hy, rim):
    p = log["bodies"][body][-1]
    return bool(abs(p[0] - cx) <= hx and abs(p[1] - cy) <= hy and p[2] < rim)

def _scene_dict(**kw):
    d = dict(assets="", bodies_xml="", static="", bodies=[], infos={},
             verify={}, plan=None, evaluate=None, cams=("front", "top"),
             lookat=(0.25, 0.0, 0.04))
    d.update(kw)
    return d

def _finish(d, table):
    d["xml"] = scene_xml(table, d.pop("assets"), d.pop("bodies_xml"), d.pop("static"))
    d["table"] = table
    return d

# ---------------------------------------------------------------- S1
BIN1 = (0.275, -0.135, 0.058, 0.058, 0.11)   # cx, cy, hx, hy, rim height
def build_s1(mug_kind="triposg"):
    table = dict(half=(0.48, 0.33), thick=0.035, height=0.90, mu=0.8,
                 rgba="0.85 0.72 0.52 1", wall_rgba="0.92 0.90 0.84 1",
                 floor_rgba="0.55 0.50 0.44 1", light_pos="-0.4 -0.6 1.6",
                 light_dir="0.25 0.25 -1")
    if mug_kind == "triposg":
        a1, b1, i1 = O.triposg_mug("mug", (0.265, 0.105), 0.8)
        mprof = PROF["mug_triposg"]
    else:
        a1, b1, i1 = O.ycb_object("mug", "mug", (0.265, 0.105), 0.8)
        mprof = PROF["mug_ycb"]
    a2, b2, i2 = O.ycb_object("apple", "apple", (0.225, -0.03), 0.8)
    static = walls_xml("bin", BIN1[0], BIN1[1], BIN1[2], BIN1[3], 0.005, BIN1[4],
                       "0.62 0.66 0.70 1", 0.8)
    def plan(P, settled):
        mp = dict(mprof); mp["carry_dur"] = 3.6
        _pp(P, settled, "mug", mp, (BIN1[0], BIN1[1]), mp["gz"] + 0.002, tag="mug",
            final_dur=2.2)
        _pp(P, settled, "apple", PROF["apple_ycb"], (BIN1[0] + 0.015, BIN1[1] + 0.01),
            PROF["apple_ycb"]["gz"] + 0.010, tag="apple")
        P.home(); P.dwell(1.2)
    def evaluate(log, settled):
        r = {}
        ok = True
        for b in ("mug", "apple"):
            c = _contained(log, b, BIN1[0], BIN1[1], BIN1[2], BIN1[3], BIN1[4])
            s = log_settled(log, b) * 1000
            r[b] = dict(contained=c, settled_mm=round(s, 2),
                        final=np.round(log["bodies"][b][-1], 4).tolist())
            ok &= c and s < 2.0
        r["arm_retreated"] = arm_retreated(log)
        r["ok"] = bool(ok and r["arm_retreated"])
        return r
    return _finish(_scene_dict(
        assets=a1 + a2, bodies_xml=b1 + b2, static=static, bodies=["mug", "apple"],
        infos={"mug": i1, "apple": i2},
        verify={"mug": ((0.265, 0.105, i1["rest_z"]), i1["quat"]),
                "apple": ((0.225, -0.03, i2["rest_z"]), i2["quat"])},
        plan=plan, evaluate=evaluate, cams=("front", "top"),
        lookat=(0.26, -0.03, 0.03)), table)

# ---------------------------------------------------------------- S2
TRAY_R = (0.30, 0.155, 0.055, 0.040)
TRAY_B = (0.30, -0.155, 0.055, 0.040)
def build_s2():
    table = dict(half=(0.43, 0.30), thick=0.03, height=0.74, mu=0.7,
                 rgba="0.55 0.38 0.22 1", wall_rgba="0.80 0.78 0.74 1",
                 light_pos="0.3 -0.5 1.5", light_dir="-0.1 0.2 -1")
    blocks = [("blk_r1", (0.200, 0.080), "0.90 0.15 0.15 1", TRAY_R),
              ("blk_r2", (0.255, 0.045), "0.90 0.15 0.15 1", TRAY_R),
              ("blk_b1", (0.200, -0.080), "0.15 0.30 0.95 1", TRAY_B),
              ("blk_b2", (0.255, -0.045), "0.15 0.30 0.95 1", TRAY_B)]
    bodies_xml, infos, verify = "", {}, {}
    for name, xy, rgba, _ in blocks:
        _, bx, info = O.spec_cube(name, xy, 0.7, rgba)
        bodies_xml += bx; infos[name] = info
        verify[name] = ((xy[0], xy[1], info["rest_z"]), info["quat"])
    static = tray_xml("tray_r", TRAY_R[0], TRAY_R[1], TRAY_R[2], TRAY_R[3], "0.85 0.35 0.35 1", 0.7)
    static += tray_xml("tray_b", TRAY_B[0], TRAY_B[1], TRAY_B[2], TRAY_B[3], "0.35 0.45 0.90 1", 0.7)
    def plan(P, settled):
        seen = {}
        for name, _, _, tray in blocks:
            k = id(tray)
            off = -0.021 if seen.get(k, 0) == 0 else 0.021
            seen[k] = seen.get(k, 0) + 1
            _pp(P, settled, name, PROF["cube"], (tray[0] + off, tray[1]), 0.030,
                carry_extra=0.115, tag=name)
        P.home(); P.dwell(1.2)
    def evaluate(log, settled):
        r, ok = {}, True
        for name, _, _, tray in blocks:
            c = _contained(log, name, tray[0], tray[1], tray[2], tray[3], 0.045)
            s = log_settled(log, name) * 1000
            r[name] = dict(in_own_tray=c, settled_mm=round(s, 2),
                           final=np.round(log["bodies"][name][-1], 4).tolist())
            ok &= c and s < 2.0
        r["arm_retreated"] = arm_retreated(log)
        r["ok"] = bool(ok and r["arm_retreated"])
        return r
    return _finish(_scene_dict(
        bodies_xml=bodies_xml, static=static, bodies=[b[0] for b in blocks],
        infos=infos, verify=verify, plan=plan, evaluate=evaluate,
        cams=("front", "wrist_cam"), lookat=(0.25, 0.0, 0.03)), table)

# ---------------------------------------------------------------- S3
TRAY3 = (0.295, -0.145, 0.060, 0.045)
def build_s3():
    table = dict(half=(0.45, 0.30), thick=0.04, height=0.90, mu=1.0,
                 rgba="0.16 0.16 0.18 1", wall_rgba="0.45 0.47 0.50 1",
                 floor_rgba="0.25 0.25 0.27 1", light_pos="0.5 0.4 1.6",
                 light_dir="-0.2 -0.15 -1")
    a1, b1, i1 = O.ycb_object_with_pose("screwdriver", "screwdriver",
        (0.249, 0.030, 0.0163), (-0.5258, -0.0279, 0.0167, 0.8500), 1.0)
    a2, b2, i2 = O.ycb_object("chef_can", "chef_can", (0.195, 0.155), 1.0)
    a3, b3, i3 = O.ycb_object("tuna_can", "tuna_can", (0.315, 0.095), 1.0)
    a4, b4, i4 = O.ycb_object("sponge_c", "sponge", (0.185, -0.125), 1.0)
    static = tray_xml("tray", TRAY3[0], TRAY3[1], TRAY3[2], TRAY3[3], "0.30 0.32 0.36 1", 1.0)
    bodies = ["screwdriver", "chef_can", "tuna_can", "sponge_c"]
    infos = {"screwdriver": i1, "chef_can": i2, "tuna_can": i3, "sponge_c": i4}
    verify = {n: ((p[0], p[1], infos[n]["rest_z"]), infos[n]["quat"]) for n, p in
              zip(bodies, [(0.249, 0.030), (0.195, 0.155), (0.315, 0.095), (0.185, -0.125)])}
    def plan(P, settled):
        _pp(P, settled, "screwdriver", PROF["screwdriver"], (TRAY3[0], TRAY3[1]),
            PROF["screwdriver"]["gz"] + 0.030, carry_extra=0.13, tag="sd")
        P.home(); P.dwell(1.2)
    def evaluate(log, settled):
        r = {}
        c = _contained(log, "screwdriver", TRAY3[0], TRAY3[1], TRAY3[2], TRAY3[3], 0.065)
        s = log_settled(log, "screwdriver") * 1000
        r["screwdriver"] = dict(in_tray=c, settled_mm=round(s, 2),
                                final=np.round(log["bodies"]["screwdriver"][-1], 4).tolist())
        ok = c and s < 2.0
        for b in ("chef_can", "tuna_can", "sponge_c"):
            d = float(np.linalg.norm(log["bodies"][b][-1] - np.array(settled[b]))) * 1000
            r[b] = dict(displaced_mm=round(d, 2))
            ok &= d < 10.0
        r["arm_retreated"] = arm_retreated(log)
        r["ok"] = bool(ok and r["arm_retreated"])
        return r
    return _finish(_scene_dict(
        assets=a1 + a2 + a3 + a4, bodies_xml=b1 + b2 + b3 + b4, static=static,
        bodies=bodies, infos=infos, verify=verify, plan=plan, evaluate=evaluate,
        cams=("front", "top"), lookat=(0.25, -0.02, 0.03)), table)

# ---------------------------------------------------------------- S4
BASKET = (0.29, -0.115, 0.062, 0.062, 0.13)
S4_ITEMS = [  # (instance, ycb kind or special, spec xy, profile key, basket target offset)
    # grid layout with real clearances (v1 overpacked the +y side: the book's
    # corner overlapped the apple and marker at spawn and the settle probe
    # shoved all three — an authoring error, caught by verification)
    ("apple", "apple", (0.165, 0.055), "apple_ycb", (0.0, 0.0)),
    ("sponge", "sponge", (0.16, -0.15), "sponge", (-0.015, 0.01)),
    ("plum", "plum", (0.225, -0.24), "plum", (0.01, -0.005)),
    ("marker", "marker", (0.213, -0.115), "marker", (-0.005, -0.01)),
    ("foam_brick", "foam_brick", (0.17, -0.02), "foam_brick", (0.01, 0.015)),
    ("tennis_ball", "tennis_ball", (0.26, 0.02), "tennis_ball", (-0.01, -0.015)),
]
def build_s4():
    table = dict(half=(0.50, 0.35), thick=0.03, height=0.75, mu=0.6,
                 rgba="0.88 0.85 0.80 1", wall_rgba="0.90 0.82 0.70 1",
                 floor_rgba="0.50 0.42 0.35 1", light_pos="-0.3 0.6 1.6",
                 light_dir="0.15 -0.25 -1")
    assets, bodies_xml, infos, verify = "", "", {}, {}
    bodies = []
    for inst, kind, xy, _, _ in S4_ITEMS:
        a, b, info = O.ycb_object(inst, kind, xy, 0.6)
        assets += a; bodies_xml += b; infos[inst] = info; bodies.append(inst)
        verify[inst] = ((xy[0], xy[1], info["rest_z"]), info["quat"])
    _, bb, ib = O.spec_book("book", (0.315, 0.225), 0.6, yaw=0.3)
    bodies_xml += bb; infos["book"] = ib; bodies.append("book")
    verify["book"] = ((0.315, 0.225, ib["rest_z"]), ib["quat"])
    static = walls_xml("basket", BASKET[0], BASKET[1], BASKET[2], BASKET[3], 0.005,
                       BASKET[4], "0.70 0.55 0.35 1", 0.6)
    def plan(P, settled):
        for inst, _, _, pkey, off in S4_ITEMS:
            prof = PROF[pkey]
            _pp(P, settled, inst, prof, (BASKET[0] + off[0], BASKET[1] + off[1]),
                prof["gz"] + 0.012, carry_extra=0.15, tag=inst)
        P.home(); P.dwell(1.2)
    def evaluate(log, settled):
        r, ok = {}, True
        for inst, _, _, _, _ in S4_ITEMS:
            c = _contained(log, inst, BASKET[0], BASKET[1], BASKET[2], BASKET[3], BASKET[4])
            s = log_settled(log, inst) * 1000
            r[inst] = dict(contained=c, settled_mm=round(s, 2),
                           final=np.round(log["bodies"][inst][-1], 4).tolist())
            ok &= c and s < 2.0
        d = float(np.linalg.norm(log["bodies"]["book"][-1] - np.array(settled["book"]))) * 1000
        t = tilt_deg(log, "book", infos["book"]["quat"])
        r["book"] = dict(displaced_mm=round(d, 2), tilt_deg=round(t, 2))
        ok &= d < 5.0 and t < 3.0
        r["arm_retreated"] = arm_retreated(log)
        r["ok"] = bool(ok and r["arm_retreated"])
        return r
    return _finish(_scene_dict(
        assets=assets, bodies_xml=bodies_xml, static=static, bodies=bodies,
        infos=infos, verify=verify, plan=plan, evaluate=evaluate,
        cams=("front", "top"), lookat=(0.25, 0.0, 0.03)), table)

# ---------------------------------------------------------------- S5
def build_s5():
    table = dict(half=(0.33, 0.28), thick=0.03, height=0.60, mu=0.7,
                 rgba="0.45 0.30 0.18 1", wall_rgba="0.78 0.80 0.82 1",
                 floor_rgba="0.40 0.38 0.36 1", light_pos="0.2 -0.4 1.4",
                 light_dir="0 0.15 -1")
    cubes = [("cube_a", (0.205, 0.095), "0.90 0.60 0.15 1"),
             ("cube_b", (0.205, -0.005), "0.20 0.75 0.30 1"),
             ("cube_c", (0.212, -0.098), "0.25 0.45 0.90 1")]
    bodies_xml, infos, verify = "", {}, {}
    for name, xy, rgba in cubes:
        _, bx, info = O.spec_cube(name, xy, 0.8, rgba)
        bodies_xml += bx; infos[name] = info
        verify[name] = ((xy[0], xy[1], info["rest_z"]), info["quat"])
    base = (0.272, 0.0)
    def plan(P, settled):
        _pp(P, settled, "cube_a", PROF["cube"], base, 0.024, carry_extra=0.13, tag="a")
        P.home(1.2)   # re-approach from the park basin: the second pick, approached from the tower side, shoved its cube deterministically (approach-configuration dependent)
        bp = dict(PROF["cube"]); bp["dx"] = 0.0   # shortest-radius pick: the battery dx=-0.01 puts the base-side jaw on the cube (deterministic -17 mm shove in three runs)
        _pp(P, settled, "cube_b", bp, base, 0.063, carry_extra=0.15,
            tag="b", settle_dwell=1.2, final_dur=2.5, hover_dwell=0.6)
        P.home(1.2)
        cc = dict(PROF["cube"]); cc["close"] = -0.18
        _pp(P, settled, "cube_c", cc, base, 0.103, carry_extra=0.17,
            tag="c", settle_dwell=0.4, final_dur=3.0, hover_dwell=0.6)
        P.dwell(3.0)   # the judged settle (SUCCESS_DEFS S5)
    def evaluate(log, settled):
        pa = log["bodies"]["cube_a"][-1]; pb = log["bodies"]["cube_b"][-1]; pc = log["bodies"]["cube_c"][-1]
        r = dict(
            a_final=np.round(pa, 4).tolist(), b_final=np.round(pb, 4).tolist(),
            c_final=np.round(pc, 4).tolist(),
            b_over_a_mm=round(float(np.linalg.norm(pb[:2] - pa[:2])) * 1000, 1),
            c_over_b_mm=round(float(np.linalg.norm(pc[:2] - pb[:2])) * 1000, 1),
            tilts_deg={n: round(tilt_deg(log, n, infos[n]["quat"]), 1) for n, _ , _ in
                       [(c[0], None, None) for c in cubes]})
        ok = (r["b_over_a_mm"] < 15 and 0.055 <= pb[2] <= 0.065
              and r["c_over_b_mm"] < 15 and 0.095 <= pc[2] <= 0.105
              and all(v < 10 for v in r["tilts_deg"].values())
              and min(pa[2], pb[2], pc[2]) > 0.015)
        r["ok"] = bool(ok)
        return r
    return _finish(_scene_dict(
        bodies_xml=bodies_xml, bodies=[c[0] for c in cubes], infos=infos,
        verify=verify, plan=plan, evaluate=evaluate,
        cams=("front", "wrist_cam"), lookat=(0.26, 0.0, 0.05)), table)

# ---------------------------------------------------------------- S6 cousins
def build_s6a():
    table = dict(half=(0.48, 0.33), thick=0.035, height=0.90, mu=0.5,
                 rgba="0.75 0.78 0.80 1", wall_rgba="0.82 0.86 0.88 1",
                 floor_rgba="0.50 0.52 0.55 1", light_pos="-0.4 0.6 1.6",
                 light_dir="0.25 -0.25 -1")
    BIN = (0.275, 0.13, 0.058, 0.058, 0.11)
    a1, b1, i1 = O.triposg_mug("mug", (0.25, -0.10), 0.5)
    a2, b2, i2 = O.ycb_object("apple", "apple", (0.205, 0.03), 0.5)
    static = walls_xml("bin", BIN[0], BIN[1], BIN[2], BIN[3], 0.005, BIN[4],
                       "0.62 0.66 0.70 1", 0.5)
    def plan(P, settled):
        _pp(P, settled, "mug", PROF["mug_triposg"], (BIN[0], BIN[1]),
            PROF["mug_triposg"]["gz"] + 0.002, tag="mug", final_dur=2.2)
        _pp(P, settled, "apple", PROF["apple_ycb"], (BIN[0] - 0.015, BIN[1] - 0.01),
            PROF["apple_ycb"]["gz"] + 0.010, tag="apple")
        P.home(); P.dwell(1.2)
    def evaluate(log, settled):
        r, ok = {}, True
        for b in ("mug", "apple"):
            c = _contained(log, b, BIN[0], BIN[1], BIN[2], BIN[3], BIN[4])
            s = log_settled(log, b) * 1000
            r[b] = dict(contained=c, settled_mm=round(s, 2),
                        final=np.round(log["bodies"][b][-1], 4).tolist())
            ok &= c and s < 2.0
        r["arm_retreated"] = arm_retreated(log)
        r["ok"] = bool(ok and r["arm_retreated"])
        return r
    return _finish(_scene_dict(
        assets=a1 + a2, bodies_xml=b1 + b2, static=static, bodies=["mug", "apple"],
        infos={"mug": i1, "apple": i2},
        verify={"mug": ((0.25, -0.10, i1["rest_z"]), i1["quat"]),
                "apple": ((0.205, 0.03, i2["rest_z"]), i2["quat"])},
        plan=plan, evaluate=evaluate, cams=("front", "top"),
        lookat=(0.26, 0.02, 0.03)), table)

def build_s6b():
    table = dict(half=(0.48, 0.33), thick=0.035, height=0.90, mu=0.35,
                 rgba="0.88 0.88 0.86 1", wall_rgba="0.86 0.84 0.80 1",
                 floor_rgba="0.52 0.50 0.46 1", light_pos="0.5 -0.3 1.6",
                 light_dir="-0.2 0.1 -1")
    BIN = (0.30, -0.02, 0.058, 0.058, 0.11)
    # Cousin deltas vs S1: tile surface (mu 0.35), the mug is a blue-glaze
    # instance of the generated mug (the YCB mug conversion failed its own
    # verification — tips at spawn, grasp never engages — so the "different
    # mug" axis degraded to a re-glazed instance; recorded in the report),
    # layout mirrored front, light from the left.
    a1, b1, i1 = O.triposg_mug("mug", (0.24, 0.135), 0.35, rgba="0.25 0.45 0.85 1")
    a2, b2, i2 = O.ycb_object("apple", "apple", (0.215, -0.125), 0.35)
    static = walls_xml("bin", BIN[0], BIN[1], BIN[2], BIN[3], 0.005, BIN[4],
                       "0.62 0.66 0.70 1", 0.35)
    def plan(P, settled):
        # documented friction-class adjustment: slower descents on tile (mu 0.35)
        mp = dict(PROF["mug_triposg"]); mp.update(close=-0.65, approach_grip=0.2, close_dur=3.5)
        _pp(P, settled, "mug", mp, (BIN[0], BIN[1]),
            mp["gz"] + 0.002, tag="mug", descend_dur=1.8, final_dur=2.2)
        _pp(P, settled, "apple", PROF["apple_ycb"], (BIN[0] + 0.015, BIN[1] + 0.01),
            PROF["apple_ycb"]["gz"] + 0.010, tag="apple", descend_dur=1.8)
        P.home(); P.dwell(1.2)
    def evaluate(log, settled):
        r, ok = {}, True
        for b in ("mug", "apple"):
            c = _contained(log, b, BIN[0], BIN[1], BIN[2], BIN[3], BIN[4])
            s = log_settled(log, b) * 1000
            r[b] = dict(contained=c, settled_mm=round(s, 2),
                        final=np.round(log["bodies"][b][-1], 4).tolist())
            ok &= c and s < 2.0
        r["arm_retreated"] = arm_retreated(log)
        r["ok"] = bool(ok and r["arm_retreated"])
        return r
    return _finish(_scene_dict(
        assets=a1 + a2, bodies_xml=b1 + b2, static=static, bodies=["mug", "apple"],
        infos={"mug": i1, "apple": i2},
        verify={"mug": ((0.24, 0.135, i1["rest_z"]), i1["quat"]),
                "apple": ((0.215, -0.125, i2["rest_z"]), i2["quat"])},
        plan=plan, evaluate=evaluate, cams=("front", "top"),
        lookat=(0.26, 0.0, 0.03)), table)

BUILDERS = {"s1": build_s1, "s2": build_s2, "s3": build_s3, "s4": build_s4,
            "s5": build_s5, "s6a": build_s6a, "s6b": build_s6b}
DESCRIPTIONS = {
    "s1": "move the mug and the apple into the storage bin",
    "s2": "sort the red and blue blocks into the matching trays",
    "s3": "pick up the screwdriver lying by the parts tray and place it in the tray",
    "s4": "clear the small items into the basket, leave the book",
    "s5": "stack the three cubes into a tower",
    "s6a": "move the mug and the apple into the storage bin",
    "s6b": "move the mug and the apple into the storage bin",
}
SCENE_TITLES = {
    "s1": "Kitchen counter, morning cleanup (light wood, mu 0.8, front+top cams)",
    "s2": "Wooden desk, sorting (wood, mu 0.7, front+wrist cams)",
    "s3": "Workshop bench, tools (dark mat, mu 1.0, front+top cams)",
    "s4": "Cluttered dining table (tablecloth, mu 0.6, front+top cams)",
    "s5": "Side table, stacking (wood, mu 0.7, front+wrist cams)",
    "s6a": "Cousin A of S1 (laminate mu 0.5, bin other side, light from right)",
    "s6b": "Cousin B of S1 (tile mu 0.35, blue-glaze mug instance, front bin)",
}
