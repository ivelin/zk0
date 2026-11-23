"""Reference-controller core for twin-suite v2 Stage 1 solvability validation
(Amendment 2 SS A2.2). REVISION 2 (2026-11-23): ONE global controller
revision round over the frozen batch (8/20). All changes are GLOBAL --
one parameter set for every scene, aimed at failure CLASSES, no
per-scene conditionals, no relaxed limits or criteria:
  G1 staged slow close (4.8 s total vs 2.4 s; segments at 30% / 65% /
     100% of close travel, displacement logged at the checkpoints):
     less ejection energy for light objects -- targets the
     grasp-ejection / displacement-cascade class (tool 697/883,
     eye 1783). (A displacement ABORT was tried in dev and removed:
     benign squeeze-centring displaces a 30 mm cube 7-20 mm during
     close, indistinguishable from ejection -- see grasp().)
  G2 hold = stall - 0.04 (was -0.03): firmer carry pinch (tool 852
     carry-loss class).
  G3 slower carry/place execution (execute_sol dist/0.65, cap 5 s) and
     slower grasp descend (2.2 s): less in-pads rotation/slip in transit
     (ring tilt growth, tool carry loss).
  G4 stepped lift (<=35 mm increments, orientation held): branch-
     following lift for near-base grasps whose one-shot lift target is
     IK-infeasible (block 411 lift class).
  G5 slower release open (1.2 s): less release disturbance at place
     (block 498 settle class).
GENERATION: ORACLE-GUIDED -- every controller here
consumes ground-truth scene state by design; it is a scene-solvability
proof executor, never a policy measurement.

Pinch-centre targeting: all grasp/place targets are expressed in the
PINCH-CENTRE frame, derived from the two authored pad faces
(finger_pad_fixed on body `gripper`, finger_pad_moving on body
`moving_jaw_so101_v1`): pinch_centre = midpoint of the two pad geom
centres, evaluated at the CURRENT gripper joint angle (the moving pad
rides the gripper joint, so the frame is grip-angle dependent and is
re-evaluated inside every IK iteration). The gripperframe site is NOT
used for targeting. The site->pinch-centre offset at the grasp
configuration is measured per scene and recorded as an instrument
parameter in the run record.

IK: damped least squares on the 5 arm joints (gripper excluded),
position of pinch centre + orientation of the gripper body, joint
limits CLIPPED to the model's own jnt_range at every iteration and
verified at the solution (honest limits -- a solution outside the
model ranges is rejected, never relaxed). Multi-start branch search:
seeds = current pose, recorded-demo frames (oracle), a deterministic
grid of shoulder/elbow/wrist seeds, and tilted-approach orientation
variants supplied by the caller.
"""
import json, math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np

V1 = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
V2 = Path("/home/openclaw/muse-work/zk0-study001-twin-v2-20261122")
sys.path.insert(0, str(V2 / "harness"))
sys.path.insert(0, str(V1 / "code"))
import mujoco
import sim_runner_padded as SR
import replay as v1replay
import build_scenes as B

PADDED = str(V2 / "scenes-v2/arm/so101_new_calib_padded.xml")
V1ARM = "/home/openclaw/muse-work/zk0-sim-v01-20261115/sim/model/so101_new_calib.xml"
ARM5 = SR.JOINTS[:5]
OPEN_RAD, PRE_RAD, CLOSE_RAD = 1.20, 0.70, 0.05


def ensure_scene(task, ep):
    """v2 template for a v1.1 nominal scene: byte copy, include swapped to
    the padded arm (same procedure as the fingerpad fix's templates)."""
    dst = V2 / "scenes-v2" / f"{task}__ep{ep}.xml"
    if not dst.exists():
        txt = (V1 / "scenes" / f"{task}__ep{ep}.xml").read_text()
        assert V1ARM in txt, f"arm include not found in {task} ep{ep}"
        dst.write_text(txt.replace(V1ARM, PADDED))
    return dst


def Rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Ref:
    def __init__(self, task, ep, use_weld_device=False):
        self.task, self.ep = task, ep
        self.use_weld_device = use_weld_device
        scene = ensure_scene(task, ep)
        self.model = SR.build_actuated_model(scene)
        self.info = SR.prepare_model(self.model)
        self.model.opt.timestep = SR.P.TIMESTEP
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = {j: int(m.joint(j).qposadr[0]) for j in SR.JOINTS}
        self.vadr = {j: int(m.joint(j).dofadr[0]) for j in SR.JOINTS}
        self.act = {j: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j)) for j in SR.JOINTS}
        self.lim = {j: (float(m.jnt_range[int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j))][0]),
                        float(m.jnt_range[int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j))][1])) for j in ARM5}
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.gbody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.pad_fixed = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_pad_fixed")
        self.pad_moving = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_pad_moving")
        gt_path = SR.RECORDS / f"base__{task}__ep{ep}.npz"
        self.gt = np.load(gt_path)["gt"].astype(float) if gt_path.exists() else None
        self.log = []
        self.frame = 0
        self.weld_eq = None
        mujoco.mj_resetData(self.model, self.data)
        # Executor's chosen start configuration (scripted executor: the arm's
        # initial pose is ours to choose; scene objects are untouched). The
        # recorded gt[0] pose is a deep fold whose elbow gravity torque is
        # already at the +-3.35 N m actuator saturation in this model
        # (measured), which strands actuated motion out of the fold on some
        # scenes; HOME is a mid-workspace pose with all static torques
        # <= ~1 N m.
        HOME = np.radians([0.0, -50.0, 40.0, 70.0, 0.0])  # pinch ~(0.218, 0, 0.143), static torques <=0.4 N m
        for i, j in enumerate(ARM5):
            self.data.qpos[self.qadr[j]] = HOME[i]; self.data.ctrl[self.act[j]] = HOME[i]
        self.data.qpos[self.qadr["gripper"]] = OPEN_RAD; self.data.ctrl[self.act["gripper"]] = OPEN_RAD
        mujoco.mj_forward(self.model, self.data)
        self.settle(0.25)

    # ---------- frames ----------
    def pinch_centre(self):
        d = self.data
        return (np.array(d.geom_xpos[self.pad_fixed]) + np.array(d.geom_xpos[self.pad_moving])) / 2.0

    def site_pos(self):
        return np.array(self.data.site_xpos[self.sid])

    def pinch_offset_from_site(self):
        return self.pinch_centre() - self.site_pos()

    def body_pos(self, name):
        return np.array(self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)])

    def body_quat(self, name):
        return np.array(self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)])

    def contacts(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        out = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1]); b2 = int(self.model.geom_bodyid[c.geom2])
            if bid in (b1, b2):
                out.add(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2 if b1 == bid else b1))
        return sorted(str(x) for x in out)

    def grip_q(self):
        return float(self.data.qpos[self.qadr["gripper"]])

    # ---------- IK ----------
    def _ik(self, target, Rt, seed_q, iters=400, w_ori=0.5):
        """Kinematic IK on saved state. Returns (q5, pos_err, ori_err_deg, within_limits)."""
        m, d = self.model, self.data
        saved = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())
        for i, j in enumerate(ARM5):
            lo, hi = self.lim[j]
            d.qpos[self.qadr[j]] = min(max(seed_q[i], lo), hi)
        mujoco.mj_forward(m, d)
        cols = [self.vadr[j] for j in ARM5]
        target = np.array(target, float)
        for _ in range(iters):
            pc = self.pinch_centre()
            perr = target - pc
            R = d.xmat[self.gbody].reshape(3, 3)
            Re = Rt @ R.T
            ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
            if np.linalg.norm(perr) < 0.0015 and ang < 0.02:
                break
            if ang < 1e-6:
                oerr = np.zeros(3)
            else:
                axis = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * np.sin(ang))
                oerr = axis * ang
            err = np.concatenate([perr, oerr * w_ori])
            Jp = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv))
            for gi, w in ((self.pad_fixed, 0.5), (self.pad_moving, 0.5)):
                jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
                mujoco.mj_jacGeom(m, d, jp, jr, gi)
                Jp += w * jp; Jr += w * jr
            J = np.vstack([Jp[:, cols], Jr[:, cols]])
            # Posture anchor: weak rows pulling q toward the seed, so the
            # solver stays in the seed's branch / nullspace neighbourhood
            # (unanchored position-only IK jumps branches between waypoints
            # and the actuated arm cannot track the jumps).
            w_anchor = 0.015 if w_ori == 0.0 else 0.005
            qcur = np.array([d.qpos[self.qadr[j]] for j in ARM5])
            seedv = np.array([min(max(seed_q[i], self.lim[j][0]), self.lim[j][1]) for i, j in enumerate(ARM5)])
            err_a = np.concatenate([err, w_anchor * (seedv - qcur)])
            J_a = np.vstack([J, w_anchor * np.eye(5)])
            dq = np.linalg.solve(J_a.T @ J_a + 0.01 * np.eye(5), J_a.T @ err_a)
            for i, j in enumerate(ARM5):
                lo, hi = self.lim[j]
                d.qpos[self.qadr[j]] = min(max(d.qpos[self.qadr[j]] + np.clip(dq[i], -0.04, 0.04), lo), hi)
            mujoco.mj_forward(m, d)
        q = np.array([d.qpos[self.qadr[j]] for j in ARM5])
        pc = self.pinch_centre()
        perr = float(np.linalg.norm(target - pc))
        R = d.xmat[self.gbody].reshape(3, 3)
        Re = Rt @ R.T
        oerr = float(np.degrees(np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))))
        within = all(self.lim[j][0] - 1e-9 <= q[i] <= self.lim[j][1] + 1e-9 for i, j in enumerate(ARM5))
        d.qpos[:] = saved[0]; d.qvel[:] = saved[1]; d.ctrl[:] = saved[2]
        mujoco.mj_forward(m, d)
        return q, perr, oerr, within

    def torque_ok(self, q, limit=3.0):
        """Gravity-torque feasibility: required static joint torques at q
        (mj_inverse on a scratch MjData) must be under `limit` N m for every
        arm joint (actuator forcerange is +-3.35 N m). Kinematic IK alone
        wanders into configurations the STS3215s cannot hold (measured:
        three actuators saturated at +-3.35 in such a pose)."""
        if not hasattr(self, "_scratch"):
            self._scratch = mujoco.MjData(self.model)
        sc = self._scratch
        sc.qpos[:] = self.data.qpos[:]
        for i, j in enumerate(ARM5):
            sc.qpos[self.qadr[j]] = q[i]
        sc.qvel[:] = 0.0
        sc.qacc[:] = 0.0
        mujoco.mj_forward(self.model, sc)
        # qfrc_bias at rest = pure gravity torque per joint (qfrc_inverse
        # would also count contact forces from kinematically placed fingers)
        return all(abs(float(sc.qfrc_bias[self.vadr[j]])) <= limit for j in ARM5)

    def solve_position_only(self, target):
        """Best position-only solution over the full seed set.
        Returns (q, pos_err, R_at_solution) or None if pos err > 4 mm."""
        target = np.array(target, float)
        cur = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        seeds = [cur]
        if self.gt is not None:
            for f in (0, 60, 120, 190, len(self.gt) - 1):
                if f < len(self.gt):
                    seeds.append(np.radians(self.gt[f][:5]))
        pan0 = math.atan2(target[1], target[0])
        for pan in (pan0, pan0 + 0.5, pan0 - 0.5, pan0 + 1.0, pan0 - 1.0):
            for lift in (-1.7, -1.0, -0.4, 0.2, 0.8):
                for elb in (-1.0, -0.3, 0.4, 1.0, 1.6):
                    for wr in (-1.4, -0.5, 0.4, 1.3):
                        seeds.append(np.array([pan, lift, elb, wr, cur[4]]))
        # Seeded random restarts: near-base targets are reachable only in
        # exotic branches (base pan ~+-85 deg) that the structured grid
        # misses (measured on block ep411: grid best >4 mm, random-restart
        # best 1.5 mm at +50 mm). Fixed seed -> deterministic.
        rng = np.random.RandomState(11)
        lo = np.array([self.lim[j][0] for j in ARM5]); hi = np.array([self.lim[j][1] for j in ARM5])
        seeds += [rng.uniform(lo, hi) for _ in range(24)]
        cand = []
        for s in seeds:
            q, pe, oe, within = self._ik(target, np.eye(3), s, iters=250, w_ori=0.0)
            if within:
                cand.append((pe, q))
        # Prefer near-basin branches: exotic solutions (wrist_roll flipped,
        # base pan ~+-85 deg) solve the point but arrive at orientations
        # whose descend swipes the object (measured on block ep411).
        cand.sort(key=lambda x: x[0] + 0.0015 * float(np.max(np.abs(x[1] - cur))))
        for pe, q in cand:
            if pe > 0.004:
                continue  # score-sorted, not pe-sorted: skip, don't break
            # No torque gate here: this solver serves TRANSIT/approach
            # targets (stage-1 above, continuation steps) that the arm only
            # passes through. Near-base transit poses legitimately exceed
            # 3.0 N m static gravity torque on a joint (deep-fold region);
            # gating them made reachable approach points unsolvable
            # (block ep411 stage-1 regression). Hold/place solvers
            # (solve, solve_link) keep the torque gate.
            return q, pe, self._orientation_at(q)
        return None

    def demo_approach(self, target):
        """Demo-anchored approach: scan the recorded demo frames kinematically
        and return (q5, R) of the frame whose pinch centre is closest to
        `target`. The human demo is an existence proof of a reachable
        configuration near the grasp; use it as IK seed + orientation
        candidate before synthetic branches."""
        if self.gt is None:
            return None
        target = np.array(target, float)
        m, d = self.model, self.data
        saved = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())
        # Only open-gripper approach frames: mid-motion / closed frames sit
        # ON the object and clip it in transit. The recorded gripper column
        # is NOT joint degrees (measured range 0.55..25.9 on block ep411),
        # so the threshold is relative to this episode's own range.
        grips = [f[5] for f in self.gt if len(f) > 5]
        gthr = (min(grips) + 0.6 * (max(grips) - min(grips))) if grips else None
        best = None
        for f in range(len(self.gt)):
            if gthr is not None and len(self.gt[f]) > 5 and self.gt[f][5] < gthr:
                continue
            for i, j in enumerate(ARM5):
                d.qpos[self.qadr[j]] = math.radians(self.gt[f][i])
            d.qpos[self.qadr["gripper"]] = math.radians(self.gt[f][5])
            mujoco.mj_forward(m, d)
            pc = self.pinch_centre()
            dist = float(np.linalg.norm(pc - target))
            if best is None or dist < best[0]:
                best = (dist, np.radians(self.gt[f][:5]).copy(),
                        d.xmat[self.gbody].reshape(3, 3).copy(), f)
        d.qpos[:] = saved[0]; d.qvel[:] = saved[1]; d.ctrl[:] = saved[2]
        mujoco.mj_forward(m, d)
        if best is not None and best[0] <= 0.11:
            return best[1], best[2], best[3]
        return None

    def solve_link(self, target, Rt):
        """Multi-seed full-IK for one grasp-chain link: current pose, demo
        frames, a coarse grid and random restarts; best (pos + orientation)
        solution inside limits and torque-feasible, or None.
        Acceptance: pos <= 6 mm, ori <= 6 deg (pad grasp tolerance; the
        oracle centring correction absorbs the residual)."""
        target = np.array(target, float)
        cur = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        seeds = [cur]
        if self.gt is not None:
            for f in (0, 30, 57, 80, 110, len(self.gt) // 2, len(self.gt) - 1):
                if f < len(self.gt):
                    seeds.append(np.radians(self.gt[f][:5]))
        pan0 = math.atan2(target[1], target[0])
        for pan in (pan0, pan0 + 0.6, pan0 - 0.6):
            for lift in (-1.2, -0.5, 0.2):
                for elb in (-0.3, 0.5, 1.2):
                    for wr in (-1.0, 0.0, 1.0):
                        seeds.append(np.array([pan, lift, elb, wr, cur[4]]))
        rng = np.random.RandomState(7)
        seeds += [cur + rng.uniform(-0.7, 0.7, 5) for _ in range(10)]
        best = None
        for sd in seeds:
            q, pe, oe, within = self._ik(target, Rt, sd, iters=400)
            if not within or not self.torque_ok(q):
                continue
            score = pe + 0.005 * oe
            if best is None or score < best["score"]:
                best = {"q": q, "pos_err": pe, "ori_err_deg": oe, "score": score,
                        "wrist_flex_deg": float(np.degrees(q[3]))}
        if best and best["pos_err"] <= 0.006 and best["ori_err_deg"] <= 6.0:
            return best
        return None

    def _orientation_at(self, q):
        saved = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy())
        for i, j in enumerate(ARM5):
            self.data.qpos[self.qadr[j]] = q[i]
        mujoco.mj_forward(self.model, self.data)
        R = self.data.xmat[self.gbody].reshape(3, 3).copy()
        self.data.qpos[:] = saved[0]; self.data.qvel[:] = saved[1]; self.data.ctrl[:] = saved[2]
        mujoco.mj_forward(self.model, self.data)
        return R

    def solve(self, target, Rt, extra_seeds=()):
        """Two-phase multi-start branch search: position-only screening over
        current/gt/grid/random seeds, then full-IK polish of the best
        branches. Returns best feasible solution dict or None. A solution
        counts only inside the model's joint limits with pos err <= 4 mm."""
        target = np.array(target, float)
        cur = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        seeds = [cur]
        if self.gt is not None:
            for f in (0, 60, 120, 160, 190, len(self.gt) // 2, len(self.gt) - 1):
                if f < len(self.gt):
                    seeds.append(np.radians(self.gt[f][:5]))
        pan0 = math.atan2(target[1], target[0])
        for pan in (pan0, pan0 + 0.5, pan0 - 0.5, pan0 + 1.0, pan0 - 1.0):
            for lift in (-1.7, -1.0, -0.4, 0.2, 0.8):
                for elb in (-1.0, -0.3, 0.4, 1.0, 1.6):
                    for wr in (-1.4, -0.5, 0.4, 1.3):
                        seeds.append(np.array([pan, lift, elb, wr, cur[4]]))
        rng = np.random.RandomState(20261124)
        for _ in range(8):
            s = cur.copy()
            s[1] += rng.uniform(-0.8, 0.8); s[2] += rng.uniform(-1.0, 1.0); s[3] += rng.uniform(-1.2, 1.2)
            seeds.append(s)
        seeds.extend(extra_seeds)
        # Phase A: position-only screening (cheap iterations)
        cand = []
        for s in seeds:
            q, pe, oe, within = self._ik(target, Rt, s, iters=90, w_ori=0.0)
            if within:
                cand.append((pe, q))
        cand.sort(key=lambda x: x[0])
        # Phase B: polish the best few distinct branches with full IK
        best = None
        seen = []
        for pe0, q0 in cand:
            if any(np.max(np.abs(q0 - s)) < 0.15 for s in seen):
                continue
            seen.append(q0)
            q, pe, oe, within = self._ik(target, Rt, q0, iters=350)
            if not within or not self.torque_ok(q):
                continue
            score = pe + 0.005 * oe
            if best is None or score < best["score"]:
                best = {"q": q, "pos_err": pe, "ori_err_deg": oe, "score": score,
                        "wrist_flex_deg": float(np.degrees(q[3])),
                        "q_deg": [round(float(np.degrees(x)), 1) for x in q]}
            if len(seen) >= 4:
                break
        if best and best["pos_err"] <= 0.004:
            return best
        return None

    # ---------- object-centric place ----------
    def held_offset_body(self, obj_name):
        """Object centre in the gripper-body frame at the current (grasp) pose."""
        R = self.data.xmat[self.gbody].reshape(3, 3)
        return R.T @ (self.body_pos(obj_name) - self.pinch_centre()), R.copy()

    def solve_for_object(self, obj_tgt, Rt, o_body):
        pinch_tgt = np.array(obj_tgt, float) - Rt @ o_body
        # Continuity first: IK seeded by the CURRENT pose stays in the
        # current branch -- a branch jump reorients the gripper and swings
        # an eccentrically held object (ring) by centimetres. Multi-start
        # branch search only as fallback.
        cur = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        q, pe, oe, within = self._ik(pinch_tgt, Rt, cur, iters=500)
        if within and pe <= 0.004 and oe <= 6.0 and self.torque_ok(q):
            return {"q": q, "pos_err": pe, "ori_err_deg": oe,
                    "wrist_flex_deg": float(np.degrees(q[3]))}, pinch_tgt
        sol = self.solve(pinch_tgt, Rt)
        return sol, pinch_tgt

    def place_scan(self, obj_tgt, o_body, candidates, label="place"):
        """Try orientation candidates in preference order; return
        (Rt, sol, pinch_tgt) for the first feasible, else (None,None,None)."""
        for Rt in candidates:
            sol, pinch_tgt = self.solve_for_object(obj_tgt, Rt, o_body)
            if sol is not None:
                self.log.append({"place_scan": label, "pinch_target": [round(float(x), 4) for x in pinch_tgt],
                                 "wrist_flex_deg": round(sol["wrist_flex_deg"], 1),
                                 "pos_err_mm": round(sol["pos_err"] * 1000, 2)})
                return Rt, sol, pinch_tgt
        # Free-orientation branch: position-only IK at the place point, take
        # the gripper orientation the arm can actually achieve there, iterate
        # the object offset twice, then full-solve with that orientation.
        Rt = np.eye(3)
        for _ in range(3):
            pinch_tgt = np.array(obj_tgt, float) - Rt @ o_body
            got = self.solve_position_only(pinch_tgt)
            if got is None:
                break
            qpos_sol, pe0, Rt = got
            pinch_tgt2 = np.array(obj_tgt, float) - Rt @ o_body
            q2, pe2, oe2, within2 = self._ik(pinch_tgt2, Rt, qpos_sol, iters=600)
            sol = None
            if within2 and pe2 <= 0.004:
                sol = {"q": q2, "pos_err": pe2, "ori_err_deg": oe2,
                       "wrist_flex_deg": float(np.degrees(q2[3])),
                       "q_deg": [round(float(np.degrees(x)), 1) for x in q2]}
            if sol is None:
                sol, pinch_tgt2 = self.solve_for_object(obj_tgt, Rt, o_body)
            if sol is not None:
                self.log.append({"place_scan": label, "branch": "free-orientation",
                                 "pinch_target": [round(float(x), 4) for x in pinch_tgt2],
                                 "wrist_flex_deg": round(sol["wrist_flex_deg"], 1),
                                 "pos_err_mm": round(sol["pos_err"] * 1000, 2),
                                 "Rt": [[round(float(x), 3) for x in row] for row in Rt]})
                return Rt, sol, pinch_tgt2
        self.log.append({"place_scan": label, "all_candidates_infeasible": True})
        return None, None, None

    def execute_sol(self, sol, grip, secs=None):
        cur = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        dist = float(np.max(np.abs(sol["q"] - cur)))
        if secs is None:
            secs = min(5.0, max(0.8, dist / 0.65))  # G3: slower carry/place
        self.run_to(list(sol["q"]) + [grip], secs)

    # ---------- continuation (path) IK ----------
    def solve_path(self, targets, Rt, step=0.02):
        """Continuation IK: interpolate pinch-centre targets from the current
        pinch position through `targets`, orientation slerped from the
        current gripper orientation to Rt, each step seeded by the previous
        step's solution. Returns the list of joint solutions (one per
        interpolated point, last = final target) or None."""
        import mujoco as _mj
        targets = [np.array(t, float) for t in targets]
        start = self.pinch_centre()
        pts = []
        prev = start
        for t in targets:
            n = max(2, int(np.ceil(np.linalg.norm(t - prev) / step)))
            for i in range(1, n + 1):
                pts.append(prev + (t - prev) * (i / n))
            prev = t
        R0 = self.data.xmat[self.gbody].reshape(3, 3).copy()
        q0 = np.zeros(4); _mj.mju_mat2Quat(q0, R0.flatten())
        q1 = np.zeros(4); _mj.mju_mat2Quat(q1, np.array(Rt, float).flatten())
        if np.dot(q0, q1) < 0:
            q1 = -q1
        seed = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        sols = []
        N = len(pts)
        for i, p in enumerate(pts):
            f = (i + 1) / N
            qs = q0 * (1 - f) + q1 * f
            qs = qs / np.linalg.norm(qs)
            Rm = np.zeros(9); _mj.mju_quat2Mat(Rm, qs)
            Ri = Rm.reshape(3, 3)
            q, pe, oe, within = self._ik(p, Ri, seed, iters=250)
            if not within or pe > 0.006:
                return None
            sols.append(q)
            seed = q
        q, pe, oe, within = self._ik(targets[-1], Rt, seed, iters=400)
        if not within or pe > 0.004:
            return None
        sols.append(q)
        return sols

    def execute_path(self, sols, grip, secs_per=None):
        prev = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        for q in sols:
            dist = float(np.max(np.abs(np.array(q) - prev)))
            secs = min(1.4, max(0.3, dist / 1.1)) if secs_per is None else secs_per
            self.run_to(list(q) + [grip], secs)
            prev = np.array(q)

    def continuation(self, target, step=0.015):
        """Position-only continuation: pinch-centre steps of `step` from the
        current pinch position to target, each IK seeded by the previous
        step. Returns joint solutions or None."""
        target = np.array(target, float)
        start = self.pinch_centre()
        n = max(2, int(np.ceil(np.linalg.norm(target - start) / step)))
        seed = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        sols = []
        for i in range(1, n + 1):
            p = start + (target - start) * (i / n)
            q, pe, oe, within = self._ik(p, np.eye(3), seed, iters=200, w_ori=0.0)
            # No torque gate on transit waypoints (see solve_position_only:
            # near-base transit poses exceed 3.0 N m static torque; the arm
            # passes through, it does not hold).
            if not within or pe > 0.008:
                return None
            sols.append(q)
            seed = q
        return sols

    def move_pos(self, target, grip, label="move_pos"):
        """Move pinch centre to target with orientation free: continuation,
        else a direct joint move to the best position-only solution."""
        sols = self.continuation(target)
        if sols is not None:
            self.execute_path(sols, grip)
            self.log.append({"move": label, "target": [round(float(x), 4) for x in target], "solved": True, "via": "continuation"})
            return True
        got = self.solve_position_only(target)
        if got is not None:
            q, pe, R = got
            cur = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
            secs = min(3.5, max(0.8, float(np.max(np.abs(q - cur))) / 0.9))
            self.run_to(list(q) + [grip], secs)
            self.log.append({"move": label, "target": [round(float(x), 4) for x in target], "solved": True, "via": "direct-pos-only"})
            return True
        self.log.append({"move": label, "target": [round(float(x), 4) for x in target], "solved": False})
        return False

    # ---------- execution ----------
    def run_to(self, q6, seconds):
        d, m = self.data, self.model
        start = np.array([d.ctrl[self.act[j]] for j in SR.JOINTS])
        steps = max(1, int(seconds / SR.P.TIMESTEP))
        tgt = np.array(q6, float)
        for s in range(steps):
            f = (s + 1) / steps
            cc = start * (1 - f) + tgt * f
            for k, j in enumerate(SR.JOINTS):
                d.ctrl[self.act[j]] = cc[k]
            mujoco.mj_step(m, d)
            if not self.use_weld_device:
                assert not np.any(d.eq_active[:m.neq]), "weld activated in pad mode!"

    def settle(self, seconds=1.0):
        steps = int(seconds / SR.P.TIMESTEP)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if not self.use_weld_device:
                assert not np.any(self.data.eq_active[:self.model.neq]), "weld activated in pad mode!"

    def move(self, target, Rt, grip, secs=None, label=""):
        """Continuation IK first (fast, continuity-seeded); on failure, the
        multi-start branch search. Executes the joint move on success."""
        sols = self.solve_path([target], Rt)
        if sols is not None:
            self.execute_path(sols, grip)
            self.log.append({"move": label, "target": [round(float(x), 4) for x in target],
                             "solved": True, "via": "path",
                             "pinch_actual": [round(float(x), 4) for x in self.pinch_centre()]})
            return {"q": sols[-1], "pos_err": 0.0, "ori_err_deg": 0.0,
                    "wrist_flex_deg": float(np.degrees(sols[-1][3]))}
        sol = self.solve(target, Rt)
        if sol is None:
            self.log.append({"move": label, "target": [round(float(x), 4) for x in target], "solved": False})
            return None
        cur = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
        dist = float(np.max(np.abs(sol["q"] - cur)))
        if secs is None:
            secs = min(3.0, max(0.6, dist / 0.9))
        q6 = list(sol["q"]) + [grip]
        self.run_to(q6, secs)
        self.log.append({"move": label, "target": [round(float(x), 4) for x in target],
                         "solved": True, "pos_err_mm": round(sol["pos_err"] * 1000, 2),
                         "ori_err_deg": round(sol["ori_err_deg"], 2),
                         "wrist_flex_deg": round(sol["wrist_flex_deg"], 1),
                         "pinch_actual": [round(float(x), 4) for x in self.pinch_centre()]})
        return sol

    def move_path(self, targets, Rt, grip, label=""):
        """Cartesian-ish path: solve+execute each waypoint seeded by continuity
        (solve() includes the current pose as first seed)."""
        ok = True
        for i, t in enumerate(targets):
            if self.move(t, Rt, grip, label=f"{label}[{i}]") is None:
                ok = False
        return ok

    # ---------- demo-guided motion (oracle motion prior) ----------
    def demo_frame_nearest(self, point, fmin=0, fmax=None):
        """gt frame whose FK pinch centre is nearest `point` (the FK frame is
        offset from the scene frame by a roughly constant per-task vector,
        so the nearest frame is the demo's grasp/place moment for that
        point). Kinematic scan on saved state."""
        if self.gt is None:
            return None
        point = np.array(point, float)
        fmax = len(self.gt) if fmax is None else min(fmax, len(self.gt))
        saved = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy())
        best = (None, 1e9)
        for f in range(fmin, fmax):
            c = SR.action_deg_to_ctrl(self.gt[f])
            for k, j in enumerate(SR.JOINTS):
                self.data.qpos[self.qadr[j]] = c[k]
            mujoco.mj_forward(self.model, self.data)
            dist = float(np.linalg.norm(self.pinch_centre() - point))
            if dist < best[1]:
                best = (f, dist)
        self.data.qpos[:] = saved[0]; self.data.qvel[:] = saved[1]; self.data.ctrl[:] = saved[2]
        mujoco.mj_forward(self.model, self.data)
        return best[0]

    def follow_demo(self, f1, grip_override=None):
        """Actuated replay of the recorded joint trajectory from the current
        frame to f1 (ctrl interpolated between frames, 100 substeps/frame).
        The demo path is torque-feasible by construction (it is what the
        real arm executed); IK only corrects the last few cm afterwards."""
        if self.gt is None or f1 is None or f1 <= self.frame:
            return
        steps_per = int(round((1.0 / 20.0) / SR.P.TIMESTEP))
        for f in range(self.frame, f1):
            c0 = SR.action_deg_to_ctrl(self.gt[f])
            c1 = SR.action_deg_to_ctrl(self.gt[min(f + 1, len(self.gt) - 1)])
            if grip_override is not None:
                c0 = c0.copy(); c1 = c1.copy(); c0[5] = grip_override; c1[5] = grip_override
            for s in range(steps_per):
                a = (s + 1) / steps_per
                cc = c0 * (1 - a) + c1 * a
                for k, j in enumerate(SR.JOINTS):
                    self.data.ctrl[self.act[j]] = cc[k]
                mujoco.mj_step(self.model, self.data)
                if not self.use_weld_device:
                    assert not np.any(self.data.eq_active[:self.model.neq]), "weld activated in pad mode!"
        self.frame = f1

    # ---------- grasp ----------
    def grasp(self, obj_pos, yaw=0.0, dz=0.0, approach_h=0.08, close_target=CLOSE_RAD, label="grasp", Rt=None, obj_body=None, seed=None, seed_frame=None, pre_rad=PRE_RAD):
        """Fingers-down pinch grasp. Pinch axis = gripper-body x rotated by yaw.
        dz: pinch-centre height relative to object centre at grasp."""
        if Rt is None:
            Rt = Rz(yaw)
        self._obj_body = obj_body
        obj = np.array(obj_pos, float)
        tgt = obj.copy(); tgt[2] = obj[2] + dz
        if seed is not None:
            # Demo-direct grasp: replay the recorded demo's own frames up
            # to the approach frame nearest the grasp point (the demo path
            # is collision-free by construction; a linear joint
            # interpolation from HOME clipped and dragged the ring 46 mm,
            # measured on ring ep7), then polish-descend by IK from that
            # seed. Skips stage 1, whose exotic position-only branches near
            # the base arrive at wild wrist-roll orientations and swipe the
            # object during the descend (measured on block ep411).
            if seed_frame is not None and self.gt is not None:
                for f in range(0, seed_frame + 1, 2):
                    self.run_to([math.radians(v) for v in self.gt[f][:5]] + [OPEN_RAD], 0.07)
                self.run_to(list(seed) + [OPEN_RAD], 0.4)
            else:
                self.run_to(list(seed) + [OPEN_RAD], 2.5)
            q, pe, oe, within = self._ik(tgt, Rt, np.array(seed, float), iters=500)
            if within and pe <= 0.006 and oe <= 8.0 and self.torque_ok(q):
                self.execute_sol({"q": q}, pre_rad, secs=2.2)
            else:
                sol = self.solve_link(tgt, Rt)
                if sol is None:
                    return None
                self.execute_sol(sol, pre_rad, secs=2.2)
        if seed is None:
            # Stage 1: get above the object, orientation free (position-only).
            # Raise straight up first so the transit never drags the fingers
            # across fixtures/objects at working height.
            pc0 = self.pinch_centre()
            if pc0[2] < obj[2] + approach_h - 0.01:
                self.move_pos(pc0 + [0, 0, 0.08], OPEN_RAD, label=f"{label}:raise")
            ok = False
            for h in (approach_h, 0.05, 0.03):
                if self.move_pos(obj + [0, 0, h], OPEN_RAD, label=f"{label}:above"):
                    ok = True
                    break
            if not ok:
                return None
            # Stage 2: ALL-OR-NOTHING oriented descend. Compute the full-IK
            # solution chain kinematically first (hover -> rail -> grasp point,
            # each seeded by the previous, torque-checked); execute only if
            # every link solves. A failed chain leaves the object untouched --
            # no position-only bulldozing with uncontrolled orientation.
            # Single direct link to the grasp point (the hover/rail intermediate
            # points proved HARDER to solve than the grasp point itself --
            # wrist-limit region; measured on eye ep1630). Solved kinematically
            # first; executed as one slow joint-space move from the raised
            # above-pose with the gripper pre-closing on arrival.
            seed0 = np.array(seed, float) if seed is not None else np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
            q, pe, oe, within = self._ik(tgt, Rt, seed0, iters=500)
            if within and pe <= 0.006 and oe <= 6.0 and self.torque_ok(q):
                self.execute_sol({"q": q}, pre_rad, secs=2.2)
            else:
                sol = self.solve_link(tgt, Rt)
                if sol is not None:
                    self.execute_sol(sol, pre_rad, secs=2.2)
                else:
                    # Legacy relaxed path: position-gated multi-start solve, else
                    # position-only continuation + orientation polish. The
                    # achieved orientation is whatever the arm can reach (it is
                    # measured and returned); near-base grasps need this branch.
                    sols_p = self.solve_path([tgt], Rt)
                    if sols_p is not None:
                        self.execute_path(sols_p, pre_rad)
                    else:
                        sol2 = self.solve(tgt, Rt)
                        if sol2 is not None:
                            self.execute_sol(sol2, pre_rad, secs=2.4)
                    if sols_p is None and sol2 is None:
                        sols = self.continuation(tgt)
                        if sols is None:
                            got = self.solve_position_only(tgt)
                            if got is None:
                                return None
                            self.run_to(list(got[0]) + [pre_rad], 1.5)
                        else:
                            self.execute_path(sols, pre_rad)
                        cur = np.array([self.data.qpos[self.qadr[j]] for j in ARM5])
                        q2, pe2, oe2, within2 = self._ik(tgt, Rt, cur, iters=400)
                        if within2 and pe2 <= 0.004:
                            self.run_to(list(q2) + [pre_rad], 0.8)
        # Stage 2b: arrival verification. Near the base the static gravity
        # torque approaches the actuator limit and the arm LAGS the ctrl
        # interpolation, ending centimetres short of the IK target
        # (measured on block ep411: pinch 26 mm off at close). Re-solve
        # from the actual pose and re-execute slower, twice at most.
        for _ in range(2):
            if np.linalg.norm(self.pinch_centre() - tgt) <= 0.008:
                break
            q, pe, oe, within = self._ik(tgt, Rt, np.array([self.data.qpos[self.qadr[j]] for j in ARM5]), iters=500)
            if not within or pe > 0.006:
                break
            self.execute_sol({"q": q}, pre_rad, secs=2.6)
        # Stage 2c: oracle centring correction -- if the pinch centre is
        # >3 mm off the object's CURRENT centre in xy, shift onto it.
        obj_body = self._obj_body
        if obj_body is not None:
            for _ in range(2):
                op = self.body_pos(obj_body)
                pc = self.pinch_centre()
                err = np.array([op[0] - pc[0], op[1] - pc[1], 0.0])
                if np.linalg.norm(err) <= 0.003:
                    break
                ct = pc + err + [0, 0, (op[2] + dz) - pc[2]]
                q, pe, oe, within = self._ik(ct, Rt, np.array([self.data.qpos[self.qadr[j]] for j in ARM5]), iters=400)
                if within and pe <= 0.004:
                    self.execute_sol({"q": q}, pre_rad, secs=0.7)
                else:
                    break
        off = self.pinch_offset_from_site()
        # G1: GUARDED TWO-PHASE SLOW CLOSE (global). A fast close flicks
        # light objects out of the pads before friction engages (measured
        # on tool ep683: the 10 mm capsule was ejected 60 mm by a 1.2 s
        # close), and repeated full closes cascade a misaligned object
        # out of the workspace (tool 697/883, eye 1783). Phase 1 closes
        # halfway over 1.8 s; an oracle displacement check then aborts
        # the attempt -- reopen + retreat, object nearly untouched -- if
        # the object moved > 5 mm, so the retry loop gets a fresh draw
        # instead of chasing a displaced object. Phase 2 completes the
        # close over 2.6 s.
        obj_pre = self.body_pos(obj_body) if obj_body is not None else None
        grip_now = float(self.data.ctrl[self.act["gripper"]])
        # Two checkpoints (30% / 65% of the close travel): an abort must
        # fire BEFORE the pads have pushed the object far, so each failed
        # attempt costs <= ~4 mm of displacement and the retry loop keeps
        # working near the object's original pose.
        # NOTE (dev evidence, rev2): a displacement ABORT at these
        # checkpoints was tried and REMOVED -- a benign squeeze-centring
        # pinch of a 30 mm cube displaces it 7-20 mm during close
        # (measured on block ep453, a frozen PASS, where the abort
        # converted every grasp into a failure), indistinguishable in
        # magnitude from ejection. The checkpoints remain as a slower,
        # staged close profile + a displacement LOG for the record.
        for frac, secs in ((0.30, 1.2), (0.65, 1.2), (1.0, 2.4)):
            q6 = [self.data.ctrl[self.act[j]] for j in SR.JOINTS]
            q6[5] = grip_now + frac * (close_target - grip_now)
            self.run_to(q6, secs)
            if obj_pre is not None and frac < 1.0:
                disp = float(np.linalg.norm(self.body_pos(obj_body) - obj_pre))
                self.log.append({"close_checkpoint": label, "at_frac": frac,
                                 "displacement_mm": round(disp * 1000, 1)})
        stall = self.grip_q()
        hold = stall - 0.04  # G2: firmer carry pinch
        q6 = [self.data.ctrl[self.act[j]] for j in SR.JOINTS]; q6[5] = hold
        self.run_to(q6, 0.3)
        rec = {"label": label, "pinch_offset_from_site_m": [round(float(x), 4) for x in off],
               "grip_stall_rad": round(stall, 3), "hold_grip_rad": round(hold, 3)}
        self.log.append({"grasp": rec})
        Rt_achieved = self.data.xmat[self.gbody].reshape(3, 3).copy()
        return {"stall": stall, "hold": hold, "Rt": Rt_achieved, "offset_site": off}

    def lift(self, grip, height=0.07, Rt=None, label="lift"):
        # G4: STEPPED LIFT (global): <=35 mm increments, orientation held,
        # each step seeded by continuity from the last. A one-shot lift
        # target can be IK-infeasible from an exotic near-base grasp pose
        # even when a branch-following stepped path exists (block 411).
        R = Rt if Rt is not None else self.data.xmat[self.gbody].reshape(3, 3).copy()
        cur = self.pinch_centre()
        n = max(1, int(math.ceil(height / 0.035)))
        ok = True
        sol = None
        for i in range(1, n + 1):
            tgt = cur + [0, 0, height * i / n]
            sol = self.move(tgt, R, grip, label=f"{label}:step{i}")
            if sol is None:
                ok = False
                break
        if ok:
            return sol
        # orientation-free fallback: height is what matters here
        tgt = self.pinch_centre()
        tgt = np.array([tgt[0], tgt[1], cur[2] + height])
        ok2 = self.move_pos(tgt, grip, label=f"{label}:pos-only")
        return {"q": None} if ok2 else None

    def release(self, retreat_h=0.10, Rt=None):
        q6 = [self.data.ctrl[self.act[j]] for j in SR.JOINTS]; q6[5] = OPEN_RAD
        self.run_to(q6, 1.2)  # G5: slower release open
        cur = self.pinch_centre()
        R = Rt if Rt is not None else self.data.xmat[self.gbody].reshape(3, 3).copy()
        self.move(cur + [0, 0, retreat_h], R, OPEN_RAD, label="release:retreat")

    def evaluate(self):
        mujoco.mj_forward(self.model, self.data)
        return v1replay.evaluate(self.model, self.data, self.task, self.ep, mujoco)

    def record(self, verdict, phase_notes):
        res = self.evaluate()
        return {"task": self.task, "episode": self.ep, "generation": "ORACLE-GUIDED",
                "sim_success": bool(res.get("sim_success")), "secondary_pass": res.get("secondary_pass"),
                "solver_failure": bool(res.get("solver_failure")), "checks": res.get("checks"),
                "objects_final": res.get("objects_final"), "verdict": verdict,
                "phase_notes": phase_notes, "log": self.log}
