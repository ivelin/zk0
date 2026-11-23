"""Reference (scripted, oracle-guided) controllers for twin-suite v2 Stage 1
solvability validation, Amendment 2 SS A2.2. GENERATION: ORACLE-GUIDED.

REVISION 2 (2026-11-23): one global controller revision round over the
frozen batch (8/20). Global changes only (see refcore2 header for
G1-G5); driver-level global changes here:
  G6 grasp candidates aligned to the object's measured principal axis
     are added in grasp_with_retries (all tasks, after the caller's
     yaws, before the tilts): fixed yaw sets miss rotated objects and
     the pads close past them, nudging them away on every retry
     (eye 1783 class).
  G7 block place target 3 mm lower (0.030 vs 0.033) + one more
     final_servo iteration: lower release height for the settle class
     (block 498) -- a place change, NOT a criterion change.
  G8 ring RE-GRASP-FLAT (see do_ring): after the first lift, set the
     ring back down flat, re-grasp with the symmetric outside-rim grasp
     at tube-centre height, accept only a test-lift with tilt <= 12 deg
     (up to 3 attempts), then seat with the existing staging servo.
  G9 cable settle order: seat with a 2 mm press, HOLD the weld 1.2 s
     while the chain relaxes in the channel, then release and settle
     longer (cable 2125/2130 chain-tension settle-out class).

Usage: run_reference2.py <task> <episode> <out.json>
One scene per process. Writes the per-scene record JSON and prints it.
All grasp targeting is in the PINCH-CENTRE frame (refcore). Joint limits
are the model's own; IK solutions outside them are rejected.
"""
import sys, json, math
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference")
import numpy as np
from refcore2 import Ref, Rz, Ry, Rx, SR, B
import twin_params as P

CLOSE = -0.16


def yaw_candidates():
    return [Rz(math.radians(a)) for a in (0, 45, -45, 90, -90, 180)]


def tilt_candidates():
    out = []
    for a in (20, -20, 40, -40, 60, -60, 90, -90):
        out.append(Rz(0) @ Ry(math.radians(a)))
    for a in (30, -30, 60, -60):
        out.append(Rz(0) @ Rx(math.radians(a)))
    return out



def demo_carry(r, obj_tgt, hold):
    """Follow the recorded demo from the grasp moment toward its place
    moment for this target (torque-feasible carry path), grip held."""
    pf = r.demo_frame_nearest(obj_tgt, fmin=r.frame + 5)
    if pf is not None and pf > r.frame:
        r.follow_demo(pf, grip_override=hold)
        r.log.append({"demo_carry_to_frame": int(pf)})

def carry_to(r, obj_tgt, o_body, Rt, grip, label):
    """Move held object centre to obj_tgt with orientation Rt: raised waypoint
    (pinch +60mm z, same Rt) if solvable, then the place pose itself."""
    sol, pinch_tgt = r.solve_for_object(obj_tgt, Rt, o_body)
    if sol is None:
        return None
    wp = pinch_tgt + [0, 0, 0.06]
    sol_wp = r.solve(wp, Rt)
    if sol_wp is not None:
        r.execute_sol(sol_wp, grip)
    r.execute_sol(sol, grip)
    r.log.append({"carry": label, "obj_target": [round(float(x), 4) for x in obj_tgt]})
    return sol


def grasp_with_retries(r, name, dz=0.0, yaws=(0.0, math.pi / 2, -math.pi / 2, math.pi / 4, -math.pi / 4, math.pi)):
    """Try grasp orientations in order: yaw branches, then tilted branches,
    then the free-orientation branch (the orientation the arm can actually
    achieve at the grasp point). A failed IK leaves the object untouched;
    a failed pinch is re-read (oracle) and retried."""
    cands = []
    # G6: object-axis-aligned candidates (global), tried AFTER the
    # caller's yaw list and BEFORE the tilts: the object's measured
    # x-axis yaw and its perpendicular. Fixed yaw sets miss rotated
    # objects; the pads then close past the object and nudge it on
    # every retry (eye 1783: 13 attempts, all stall = full close).
    # They are not put first: on near-square objects (cubes) an
    # axis-first candidate can be a corner grasp that shoves the
    # object before the caller's proven first candidate runs
    # (measured on block ep453 in dev).
    axis_yaws = []
    try:
        q0 = r.body_quat(name)
        ax = B.qrot(q0, np.array([1.0, 0, 0]))
        ayaw = math.atan2(ax[1], ax[0])
        axis_yaws = [ayaw + math.pi / 2, ayaw]
    except Exception:
        axis_yaws = []
    seen_yaw = set()
    for y in list(yaws) + axis_yaws:
        key = round(math.degrees(y) % 360)
        if key in seen_yaw:
            continue
        seen_yaw.add(key)
        cands.append(Rz(y))
    base_yaw = yaws[0]
    for a in (25, -25, 50, -50, 65, -65):
        cands.append(Rz(base_yaw) @ Ry(math.radians(a)))
    for a in (25, -25, 50, -50, 65, -65):
        cands.append(Rz(base_yaw) @ Rx(math.radians(a)))
    p0 = r.body_pos(name)
    got = r.solve_position_only(p0 + [0, 0, dz])
    if got is not None:
        cands.append(got[2])
    # NOTE: a demo-anchored candidate (nearest open-gripper demo frame as
    # IK seed + orientation) was tried and REMOVED: the twin's pose error
    # vs the recordings is up to ~85 mm (block ep411: demo pinch never
    # within 85 mm of the twin's cube), so the anchor pinched objects
    # off-centre (eye ep1630: loose end-pinch, carton rotated in the
    # carry) and its transit dragged the ring 46 mm. refcore.demo_approach
    # remains for the record.
    for Rt in cands:
        p = r.body_pos(name)
        g = r.grasp(p, dz=dz, close_target=CLOSE, Rt=Rt, obj_body=name)
        if g is None:
            continue
        if {"gripper", "moving_jaw_so101_v1"} <= set(r.contacts(name)):
            return g, p
        # pinch failed: reopen and retreat before retrying
        q6 = [r.data.ctrl[r.act[j]] for j in SR.JOINTS]; q6[5] = 1.20
        r.run_to(q6, 0.6)
        cur = r.pinch_centre()
        r.move(cur + [0, 0, 0.06], g["Rt"], 1.20, label="grasp-retry:retreat")
    return None, r.body_pos(name)



def final_servo(r, name, obj_tgt, Rt, grip, iters=3, tol=0.0025):
    """Oracle servo polish at the place point: measure the held object's
    CURRENT centre error, lift 20 mm, re-solve with a FRESH held offset,
    re-descend. Absorbs in-pads slip/rotation during the carry."""
    obj_tgt = np.array(obj_tgt, float)
    for i in range(iters):
        cur = r.body_pos(name)
        err = obj_tgt - cur
        if np.linalg.norm(err[:2]) <= tol and abs(err[2]) <= 0.004:
            r.log.append({"servo": name, "converged_iter": i})
            return True
        pc = r.pinch_centre()
        # Orientation-HELD lift: an orientation-free (position-only) lift
        # rotates the gripper and drags a placed object sideways (measured
        # on ring ep7: servo diverged to 45 mm). Fall back to position-only
        # only if the oriented lift is infeasible.
        if r.move(pc + [0, 0, 0.02], Rt, grip, label="servo:lift") is None:
            r.move_pos(pc + [0, 0, 0.02], grip, label="servo:lift-pos")
        o_body, _ = r.held_offset_body(name)
        sol, _ = r.solve_for_object(obj_tgt, Rt, o_body)
        if sol is None:
            # fall back to a fresh place scan with the CURRENT offset
            Rt2, sol2, pinch_tgt = r.place_scan(obj_tgt, o_body, f"servo-rescan {name}")
            if sol2 is None:
                r.log.append({"servo": name, "solve_failed_iter": i})
                return False
            r.execute_sol(sol2, grip, secs=1.4)
            continue
        r.execute_sol(sol, grip, secs=1.4)
    err = obj_tgt - r.body_pos(name)
    ok = bool(np.linalg.norm(err[:2]) <= 0.004)
    r.log.append({"servo": name, "final_err_mm": round(float(np.linalg.norm(err)) * 1000, 1), "ok": ok})
    return ok

def do_block(r):
    notes = []
    A0, B0 = r.body_pos("cube_a"), r.body_pos("cube_b")
    # Moving cube = the one CLOSER to the arm base: placing onto the far
    # cube keeps the held-object place point in the reachable shell, while
    # the near-base point is only visited by the (reachable) grasp.
    # (Measured on ep411: pinch at the near-base cube +58mm z is outside
    # the position-only reachable set over a 500-seed search; the same
    # point at stack height is reachable -- see report.)
    if np.linalg.norm(A0[:2]) <= np.linalg.norm(B0[:2]):
        mov, oth, M0, O0 = "cube_a", "cube_b", A0, B0
    else:
        mov, oth, M0, O0 = "cube_b", "cube_a", B0, A0
    notes.append(f"moving={mov} at {np.round(M0, 3)}, target={oth} at {np.round(O0, 3)}")
    g, M0 = grasp_with_retries(r, mov, dz=0.008)
    if g is None:
        return "FAIL", f"grasp of {mov} infeasible in all yaw branches"
    r.lift(g["hold"], 0.07, Rt=g["Rt"])
    M1 = r.body_pos(mov)
    if M1[2] < M0[2] + 0.03:
        return "FAIL", f"lift failed: {mov} z {M1[2]:.3f} contacts={r.contacts(mov)}"
    notes.append(f"pinch-lift ok, {mov} z {M0[2]:.3f}->{M1[2]:.3f}, contacts {r.contacts(mov)}")
    o_body, _ = r.held_offset_body(mov)
    O0 = r.body_pos(oth)
    obj_tgt = O0 + [0, 0, 0.030]  # G7: 3 mm lower release height
    Rt, sol, _ = r.place_scan(obj_tgt, o_body, yaw_candidates() + tilt_candidates(), label="block place")
    if Rt is None:
        return "FAIL", "place unreachable in all orientation branches (yaw + tilt + free)"
    if carry_to(r, obj_tgt, o_body, Rt, g["hold"], "block place") is None:
        return "FAIL", "place carry failed"
    notes.append(f"place pose reached, {mov} at {np.round(r.body_pos(mov), 3)}")
    final_servo(r, mov, obj_tgt, Rt, g["hold"], iters=3)
    r.release(retreat_h=0.09, Rt=Rt)
    r.settle(1.0)
    res = r.evaluate()
    return ("PASS" if res["sim_success"] else "FAIL"), "; ".join(notes)


def do_eye(r):
    notes = []
    BK = r.body_pos("basket")
    g, C0 = grasp_with_retries(r, "carton", dz=0.006, yaws=(math.pi / 2, 0.0, -math.pi / 2, math.pi / 4))
    if g is None:
        return "FAIL", "grasp of carton infeasible in all yaw branches"
    r.lift(g["hold"], 0.08, Rt=g["Rt"])
    C1 = r.body_pos("carton")
    if C1[2] < C0[2] + 0.03:
        return "FAIL", f"lift failed: carton z {C1[2]:.3f}"
    notes.append(f"pinch-lift ok, carton z {C0[2]:.3f}->{C1[2]:.3f}")
    o_body, _ = r.held_offset_body("carton")
    # Place, don't drop: release with the carton centre at z 0.04 (bottom
    # 0.0225, just above the basket floor at 0.003, walls top 0.07). The
    # earlier 0.115 release dropped the light carton 77 mm and it came to
    # rest leaning/perched at z 0.048 (ep1630).
    obj_tgt = np.array([BK[0], BK[1], 0.04])
    Rt, sol, _ = r.place_scan(obj_tgt, o_body, [Rz(math.pi / 2), Rz(0), Rz(-math.pi / 2)] + tilt_candidates(), label="eye place")
    if Rt is None:
        return "FAIL", "basket place unreachable in all branches"
    if carry_to(r, obj_tgt, o_body, Rt, g["hold"], "eye place") is None:
        return "FAIL", "basket carry failed"
    r.release(retreat_h=0.09, Rt=Rt)
    r.settle(1.0)
    res = r.evaluate()
    return ("PASS" if res["sim_success"] else "FAIL"), "; ".join(notes)


def do_ring(r):
    notes = []
    PEG = r.body_pos("peg")
    # Ring grasp: the pinch must sit near the TUBE centre height, but
    # vertical fingers put the pad tips ~23 mm below the pinch -- into the
    # board. A vertical grasp at tip-clearing height pinches the tube's
    # upper arc and the ring squeezes out on lift (measured, ep7).
    # Tilted fingers (55 deg) clear the board with the pinch ~4 mm above
    # the tube centre. Each candidate is verified by a test lift.
    R0 = r.body_pos("ring")
    ring_cands = []
    # FIRST: outside-rim grasp -- gripper fully open (pre_rad=OPEN: face
    # gap ~61 mm > ring outer 45 mm) descending AROUND the ring with
    # vertical fingers, pinch centre over the ring centre at z +0.015:
    # the pads land on the rim EQUATOR at two opposing points, so the
    # ring hangs level instead of tilting ~50 deg in an eccentric
    # inner-rim pinch (measured). Tilted candidates remain as fallback.
    for Rt in (Rz(0.0), Rz(math.pi / 2), Rz(math.pi / 4)):
        ring_cands.append((Rt, None, None))
    for Rt in (Rz(0.0) @ Ry(math.radians(55)), Rz(0.0) @ Ry(math.radians(-55)),
               Rz(math.pi / 2) @ Ry(math.radians(55)), Rz(math.pi / 2) @ Ry(math.radians(-55)),
               Rz(0.0) @ Rx(math.radians(55)), Rz(0.0) @ Rx(math.radians(-55)), Rz(0.0)):
        ring_cands.append((Rt, None, None))
    g = None
    for Rt, seed, seed_frame in ring_cands:
        p = r.body_pos("ring")
        gg = r.grasp(p, dz=0.015, close_target=CLOSE, Rt=Rt, obj_body="ring", seed=seed, seed_frame=seed_frame, pre_rad=1.20)
        if gg is None:
            continue
        held = {"gripper", "moving_jaw_so101_v1"} <= set(r.contacts("ring"))
        if held:
            r.lift(gg["hold"], 0.035, Rt=gg["Rt"])
            held = r.body_pos("ring")[2] > p[2] + 0.02
        if held:
            g, R0 = gg, p
            break
        q6 = [r.data.ctrl[r.act[j]] for j in SR.JOINTS]; q6[5] = 1.20
        r.run_to(q6, 0.6)
        r.move_pos(r.pinch_centre() + [0, 0, 0.05], 1.20, label="ring-grasp:retreat")
    if g is None:
        return "FAIL", "grasp/lift of ring failed in all tilted branches"
    r.lift(g["hold"], 0.05, Rt=g["Rt"])
    R1 = r.body_pos("ring")
    if R1[2] < R0[2] + 0.02:
        return "FAIL", f"lift failed: ring z {R1[2]:.3f} contacts={r.contacts('ring')}"
    notes.append(f"pinch-lift ok, ring z {R0[2]:.3f}->{R1[2]:.3f}, grip stall {g['stall']:.3f} rad")
    # G8: RE-GRASP FLAT (global ring strategy). A one-point rim pinch
    # lets the ring rotate in the pads to 39-50 deg of tilt by staging,
    # and tilt-levelling at staging is IK-infeasible (wrist at limit).
    # Instead: set the ring back down FLAT at its start pose (release
    # onto the board guarantees flat), re-grasp with the symmetric
    # outside-rim grasp at tube-centre height, and accept only a test
    # lift whose measured tilt is <= 12 deg -- up to 3 attempts varying
    # yaw/dz. If no attempt qualifies, proceed with the last grasp
    # (no worse than the frozen controller's single draw).
    import mujoco as _mj2
    _ring_bid = _mj2.mj_name2id(r.model, _mj2.mjtObj.mjOBJ_BODY, "ring")
    def _ring_tilt_deg():
        Rr = r.data.xmat[_ring_bid].reshape(3, 3)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(Rr[2, 2])))))
    flat_z = R0[2]
    rg_ok = False
    for attempt, (yaw_rg, dz_rg) in enumerate(((0.0, 0.006), (math.pi / 2, 0.006), (math.pi / 4, 0.010))):
        rc = r.body_pos("ring")
        down = np.array([R0[0] - rc[0], R0[1] - rc[1], flat_z - rc[2]])
        pc = r.pinch_centre()
        if r.move(pc + down, g["Rt"], g["hold"], label="ring:regrasp-setdown") is None:
            r.move_pos(pc + down, g["hold"], label="ring:regrasp-setdown-pos")
        r.release(retreat_h=0.06, Rt=g["Rt"])
        p = r.body_pos("ring")
        gg = r.grasp(p, dz=dz_rg, close_target=CLOSE, Rt=Rz(yaw_rg), obj_body="ring", pre_rad=1.20)
        if gg is None:
            r.log.append({"ring_regrasp": attempt, "formed": False})
            # ring is flat on the board but ungrasped: fall back to the
            # original tilted-candidate grasp so the run can proceed
            gg2, _p2 = grasp_with_retries(r, "ring", dz=0.015)
            if gg2 is None:
                return "FAIL", "re-grasp-flat: outside-rim grasp and fallback grasp both failed"
            g = gg2
            r.lift(g["hold"], 0.05, Rt=g["Rt"])
            rg_ok = True
            break
        if not ({"gripper", "moving_jaw_so101_v1"} <= set(r.contacts("ring"))):
            r.log.append({"ring_regrasp": attempt, "formed": False})
            q6 = [r.data.ctrl[r.act[j]] for j in SR.JOINTS]; q6[5] = 1.20
            r.run_to(q6, 0.6)
            r.move_pos(r.pinch_centre() + [0, 0, 0.05], 1.20, label="ring:regrasp-retreat")
            continue
        r.lift(gg["hold"], 0.03, Rt=gg["Rt"])
        tilt = _ring_tilt_deg()
        r.log.append({"ring_regrasp": attempt, "tilt_deg": round(float(tilt), 1),
                      "ring_z": round(float(r.body_pos("ring")[2]), 4)})
        g = gg
        rg_ok = True
        if tilt <= 12.0:
            break
    if not rg_ok:
        return "FAIL", "re-grasp-flat failed: no outside-rim grasp formed in 3 attempts"
    if r.body_pos("ring")[2] < flat_z + 0.02:
        r.lift(g["hold"], 0.05, Rt=g["Rt"])
    notes.append(f"re-grasp-flat done, tilt now {_ring_tilt_deg():.1f} deg")
    o_body, _ = r.held_offset_body("ring")
    # two-stage descent: ring centre at peg-top height, then seat at board top + 2mm + half thickness
    top_tgt = np.array([PEG[0], PEG[1], P.BOARD_H + P.PEG_H + P.RING_THICK / 2 + 0.004])
    fin_tgt = np.array([PEG[0], PEG[1], P.BOARD_H + 0.002 + P.RING_THICK / 2])
    Rt, sol, _ = r.place_scan(top_tgt, o_body, yaw_candidates() + [Rz(0) @ Ry(math.radians(a)) for a in (10, -10, 20, -20)], label="ring over-peg")
    if Rt is None:
        return "FAIL", "over-peg pose unreachable"
    if carry_to(r, top_tgt, o_body, Rt, g["hold"], "ring over-peg") is None:
        return "FAIL", "over-peg carry failed"
    # Planar alignment correction AT STAGING HEIGHT first: the ring
    # rotates in its rim pinch during the carry (grasp-time offset stale by
    # centimetres), and if the planar error exceeds the 3 mm hole
    # clearance when the descend starts, the hole edge cams off the peg
    # rim and the ring is deflected sideways (measured on ep7: 30 mm).
    # At staging the ring bottom clears the peg top, so corrections there
    # are free-space translations. Small-step closed-loop pinch servo:
    # full-pose re-solves saturate at ~9 mm residual (measured); steps of
    # <=6 mm from the CURRENT pose/orientation, re-measuring the ring
    # centre after each, converge.
    from refcore2 import ARM5 as _A5
    for _ in range(10):
        rc = r.body_pos("ring")
        err = np.array([PEG[0] - rc[0], PEG[1] - rc[1], 0.0])
        en = float(np.linalg.norm(err))
        r.log.append({"ring_staging_err_mm": round(en * 1000, 1)})
        if en <= 0.002:
            break
        step = err * min(1.0, 0.006 / en)
        pc = r.pinch_centre()
        R_cur = r.data.xmat[r.gbody].reshape(3, 3).copy()
        cur = np.array([r.data.qpos[r.qadr[j]] for j in _A5])
        q, pe, oe, w = r._ik(pc + step, R_cur, cur, iters=400)
        if not w or pe > 0.003:
            break
        r.execute_sol({"q": q}, g["hold"], secs=1.2)
    # Level the ring before descending: the ring hangs TILTED in its rim
    # pinch, and a tilted hole axis catches the peg tip even at 1.9 mm
    # planar error (measured on ep7: descend blocked at peg-top height,
    # xy servo then drags the ring across the peg top). Rotate the
    # gripper about the pinch centre by the measured tilt of the ring's
    # plane normal, twice, re-measuring between.
    import mujoco as _mj
    from refcore2 import ARM5 as _A5c
    for _ in range(6):
        Rr = r.data.xmat[_mj.mj_name2id(r.model, _mj.mjtObj.mjOBJ_BODY, "ring")].reshape(3, 3)
        n = Rr[:, 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))
        r.log.append({"ring_tilt_deg": round(float(tilt), 1)})
        if tilt <= 3.0:
            break
        axis = np.cross(n, [0, 0, 1.0])
        na = float(np.linalg.norm(axis))
        if na < 1e-6:
            break
        axis = axis / na
        # small steps: the one-shot 49.5 deg rotation is IK-infeasible at
        # this pose (wrist limits); 10 deg steps track it.
        ang = math.radians(min(10.0, tilt))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        Rc = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        R_cur2 = r.data.xmat[r.gbody].reshape(3, 3).copy()
        Rt_lvl = Rc @ R_cur2
        pc = r.pinch_centre()
        cur = np.array([r.data.qpos[r.qadr[j]] for j in _A5c])
        q, pe, oe, w = r._ik(pc, Rt_lvl, cur, iters=500)
        if not w or pe > 0.004:
            break
        r.execute_sol({"q": q}, g["hold"], secs=1.6)
    # Re-centre xy after levelling (rotation about the rim contact swings
    # the centre slightly): rerun the small-step staging servo briefly.
    for _ in range(6):
        rc = r.body_pos("ring")
        err = np.array([PEG[0] - rc[0], PEG[1] - rc[1], 0.0])
        en = float(np.linalg.norm(err))
        if en <= 0.002:
            break
        step = err * min(1.0, 0.006 / en)
        pc = r.pinch_centre()
        R_cur3 = r.data.xmat[r.gbody].reshape(3, 3).copy()
        cur = np.array([r.data.qpos[r.qadr[j]] for j in _A5c])
        q, pe, oe, w = r._ik(pc + step, R_cur3, cur, iters=400)
        if not w or pe > 0.003:
            break
        r.execute_sol({"q": q}, g["hold"], secs=1.2)
    # Seat by STEPPED SERVO DESCEND: the single long descend to the seat
    # solution swings the pinch in an arc and destroys the staging
    # alignment (measured on ep7: 1.9 mm at staging -> 16 mm after the
    # descend, ring left on the peg top). Descend <=8 mm per step with
    # the gripper orientation held at the staging orientation, correcting
    # the ring centre's xy error after every step, so the peg enters the
    # hole under active alignment.
    from refcore2 import ARM5 as _A5b
    R_cur = r.data.xmat[r.gbody].reshape(3, 3).copy()
    seated = False
    for _ in range(16):
        rc = r.body_pos("ring")
        if rc[2] <= fin_tgt[2] + 0.002:
            seated = True
            break
        err = np.array([PEG[0] - rc[0], PEG[1] - rc[1], 0.0])
        dz_step = -min(0.008, rc[2] - fin_tgt[2])
        pc = r.pinch_centre()
        cur = np.array([r.data.qpos[r.qadr[j]] for j in _A5b])
        q, pe, oe, w = r._ik(pc + err + [0, 0, dz_step], R_cur, cur, iters=400)
        if not w or pe > 0.003:
            break
        r.execute_sol({"q": q}, g["hold"], secs=1.0)
    rc = r.body_pos("ring")
    r.log.append({"ring_seated": bool(seated), "ring_final_err_mm": round(float(np.linalg.norm(np.array([PEG[0] - rc[0], PEG[1] - rc[1]]))) * 1000, 1), "ring_z": round(float(rc[2]), 4)})
    notes.append(f"seated pose reached, ring at {np.round(r.body_pos('ring'), 3)}, peg xy {np.round(PEG[:2], 3)}")
    r.release(retreat_h=0.09, Rt=Rt)
    r.settle(1.0)
    res = r.evaluate()
    return ("PASS" if res["sim_success"] else "FAIL"), "; ".join(notes)


def do_tool(r):
    notes = []
    T0 = r.body_pos("tool")
    poses = B.load_episode_poses(r.task, r.ep)
    fx = B.load_fixtures(r.task, poses)
    slot = np.array(fx["slot"]["pos"], float); syaw = math.radians(fx["slot"].get("yaw_deg", 0.0))
    q = r.body_quat("tool")
    axis = B.qrot(q, np.array([1.0, 0, 0]))
    tool_yaw = math.atan2(axis[1], axis[0])
    g, T0 = grasp_with_retries(r, "tool", dz=0.004, yaws=(tool_yaw + math.pi / 2, tool_yaw, tool_yaw - math.pi / 2))
    if g is None:
        return "FAIL", "grasp of tool infeasible in all yaw branches"
    r.lift(g["hold"], 0.07, Rt=g["Rt"])
    T1 = r.body_pos("tool")
    if T1[2] < T0[2] + 0.03:
        return "FAIL", f"lift failed: tool z {T1[2]:.3f} contacts={r.contacts('tool')}"
    notes.append(f"pinch-lift ok, tool z {T0[2]:.3f}->{T1[2]:.3f}, grip stall {g['stall']:.3f} rad, slot at {np.round(slot, 3)} yaw {math.degrees(syaw):.0f}")
    o_body, _ = r.held_offset_body("tool")
    # held tool axis must end aligned with slot axis: rotate gripper by (syaw - tool_yaw) about z
    Rt_place = Rz(syaw - tool_yaw) @ g["Rt"]
    obj_tgt = np.array([slot[0], slot[1], 0.048])
    cands = [Rt_place] + [Rz(syaw - tool_yaw + math.radians(a)) @ g["Rt"] for a in (180,)] + tilt_candidates()
    Rt, sol, _ = r.place_scan(obj_tgt, o_body, cands, label="tool place")
    if Rt is None:
        return "FAIL", "slot place unreachable in all branches"
    if carry_to(r, obj_tgt, o_body, Rt, g["hold"], "tool place") is None:
        return "FAIL", "slot carry failed"
    final_servo(r, "tool", obj_tgt, Rt, g["hold"], iters=2)
    r.release(retreat_h=0.09, Rt=Rt)
    r.settle(1.0)
    res = r.evaluate()
    return ("PASS" if res["sim_success"] else "FAIL"), "; ".join(notes)


def do_cable(r):
    """Cable: pad pinch is physically impossible for a 5mm segment (measured
    minimum pad face gap ~8mm at the gripper joint's -0.1745 rad limit, see
    report) -- the reference execution uses the v1.1 kinematic weld device:
    weld gripper<->segment (the scene's own weld equalities, the declared
    v1.1 grasp device), carry with the ACTUATED arm, release in the channel.
    GENERATION: ORACLE-GUIDED, device: V1.1-WELD."""
    import mujoco
    notes = ["executor: v1.1 weld device (pad pinch impossible: min face gap ~8mm > cable 5mm)"]
    CB = r.body_pos("clip_block")
    yaw = 2 * math.atan2(r.body_quat("clip_block")[3], r.body_quat("clip_block")[0])
    chan = np.array([CB[0], CB[1], P.BLOCK_SIZE[2] + 0.006])
    # choose the cable segment nearest the clip block (xy)
    segs = [(float(np.linalg.norm(r.body_pos(f"cable_seg_{i}")[:2] - CB[:2])), i) for i in range(P.CABLE_N)]
    segs.sort()
    dist0, si = segs[0]
    seg = f"cable_seg_{si}"
    S0 = r.body_pos(seg)
    notes.append(f"target segment {seg}, starts {dist0*1000:.0f} mm from clip block at {np.round(S0, 3)}")
    # move pinch centre over the segment, fingers down (yaw aligned to channel)
    Rt = Rz(yaw)
    tgt = S0 + [0, 0, 0.01]
    if r.move(tgt + [0, 0, 0.07], Rt, 0.7, label="cable:above-seg") is None and not r.move_pos(tgt + [0, 0, 0.07], 0.7, label="cable:above-seg-pos"):
        return "FAIL", "approach IK infeasible"
    if r.move(tgt, Rt, 0.7, label="cable:at-seg") is None and not r.move_pos(tgt, 0.7, label="cable:at-seg-pos"):
        return "FAIL", "segment approach IK infeasible"
    # activate this segment's weld slot 0 (v1.1 device)
    m = r.model
    eqid = None
    for eq in range(m.neq):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
        if nm == f"weld_{seg}_0":
            eqid = eq
    if eqid is None:
        return "FAIL", f"no weld slot for {seg}"
    # v1.1 device attach: eq_data carries the CURRENT relative transform
    # (replay.do_attach convention), so activation does not snap the segment.
    gp = r.body_pos("gripper"); gq = r.body_quat("gripper")
    op = r.body_pos(seg); oq = r.body_quat(seg)
    row = r.model.eq_data[eqid]
    row[0:3] = 0.0
    row[3:6] = B.qrot(B.qinv(gq), op - gp)
    row[6:10] = B.qmul(B.qinv(gq), oq)
    mujoco.mj_forward(r.model, r.data)
    r.data.eq_active[eqid] = 1
    r.weld_eq = eqid
    r.settle(0.2)
    # lift and carry: segment centre -> channel centre (weld preserves offset;
    # solve pinch target from measured segment-minus-pinch offset)
    off = r.body_pos(seg) - r.pinch_centre()
    lift_tgt = r.pinch_centre() + [0, 0, 0.05]
    if r.move(lift_tgt, Rt, 0.7, label="cable:lift") is None:
        return "FAIL", "welded lift IK infeasible"
    chan_pinch = chan - off
    if r.move(chan_pinch + [0, 0, 0.05], Rt, 0.7, label="cable:over-channel") is None and not r.move_pos(chan_pinch + [0, 0, 0.05], 0.7, label="cable:over-channel-pos"):
        return "FAIL", "channel approach IK infeasible"
    # G9: seat with a 2 mm press into the channel, then HOLD the weld
    # 1.2 s while the dragged chain relaxes against the seated segment,
    # then release and settle longer. The frozen order (release
    # immediately at exact channel height) let chain tension pull the
    # segment out of the channel volume during settle (2125/2130).
    seat_pinch = chan_pinch + [0, 0, -0.002]
    if r.move(seat_pinch, Rt, 0.7, label="cable:seat") is None and not r.move_pos(seat_pinch, 0.7, label="cable:seat-pos"):
        return "FAIL", "channel seat IK infeasible"
    notes.append(f"segment at {np.round(r.body_pos(seg), 3)}, channel at {np.round(chan, 3)}")
    r.settle(1.2)  # weld still active: chain relaxes around seated segment
    r.data.eq_active[eqid] = 0
    r.settle(0.3)
    cur = r.pinch_centre()
    r.move(cur + [0, 0, 0.08], Rt, 0.7, label="cable:retreat")
    r.settle(1.5)
    res = r.evaluate()
    return ("PASS" if res["sim_success"] else "FAIL"), "; ".join(notes)


TASKS = {"block_stack": (do_block, False), "eye_drops_to_basket": (do_eye, False),
         "ring_insert": (do_ring, False), "tool_insert": (do_tool, False),
         "cable_clip": (do_cable, True)}


def main():
    task, ep, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    fn, weld = TASKS[task]
    r = Ref(task, ep, use_weld_device=weld)
    try:
        verdict, notes = fn(r)
    except Exception as e:
        import traceback
        verdict, notes = "ERROR", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    rec = r.record(verdict, notes)
    rec["executor"] = "v1.1-weld-device" if weld else "padded-arm scripted (pinch-centre)"
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(json.dumps({"task": task, "episode": ep, "verdict": verdict,
                      "sim_success": rec["sim_success"], "notes": notes}, default=str))


if __name__ == "__main__":
    main()
