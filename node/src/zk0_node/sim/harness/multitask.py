"""Multi-task MuJoCo benchmark for the SO-101 (Menagerie robotstudio_so101, stock).

Generalizes test_mujoco.py (single-cube grasp test): same DLS IK on the
gripperframe site, same smoothstep joint-space waypoints, same Renderer ->
imageio MP4 pipeline, state logged every 0.05 s. Grasps are contact-and-
friction only: no welds, no equality constraints, no object scripting.

Usage: python multitask.py <task> '<json overrides>'
Tasks: task1, task2, task3 (pen; add "scan":true for a close-parameter scan),
       task4, pen_scan

VENDORED COPY — zk0 Node package (2026-11-17). Identical to the certified
original (md5 87b1bad6cdda448d79deedf6933c6a64 — see PROVENANCE.md) except
for ONE declared patch, marked [zk0-node vendored patch] below: the
hardcoded build-directory paths W/MODEL are taken from the
ZK0_SIM_MODEL_DIR environment variable instead, so the package stages
its own model directory and never writes into the certified work dir.
No task, pilot, IK, or scoring logic changes.
"""
import numpy as np, json, sys, time, os
os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering on the Spark
import mujoco
from grasp_gate import GateRecorder

# [zk0-node vendored patch — see PROVENANCE.md] The certified original
# hardcodes W/MODEL to its build directory. The vendored copy takes the
# staged model directory from ZK0_SIM_MODEL_DIR (set by the package
# runner before import) so the package never writes scene files into
# the certified work directory. No logic changes.
MODEL = os.environ.get("ZK0_SIM_MODEL_DIR")
if not MODEL:
    raise RuntimeError("ZK0_SIM_MODEL_DIR is not set — the zk0 node sim "
                       "runner stages the model directory and sets it")
W = MODEL
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIP_OPEN = 1.74533
GRIP_CLOSE_CUBE = -0.10

# ---------------------------------------------------------------- scenes
def walls_xml(prefix, cx, cy, hx, hy, t, h, rgba):
    """Four static wall geoms forming an open-top box. hx,hy = interior half
    extents, t = wall half-thickness, h = wall height."""
    z = h / 2
    return f"""
    <geom name="{prefix}_n" type="box" pos="{cx} {cy + hy + t} {z}" size="{hx + t} {t} {z}" friction="1 0.005 0.0001" rgba="{rgba}"/>
    <geom name="{prefix}_s" type="box" pos="{cx} {cy - hy - t} {z}" size="{hx + t} {t} {z}" friction="1 0.005 0.0001" rgba="{rgba}"/>
    <geom name="{prefix}_e" type="box" pos="{cx + hx + t} {cy} {z}" size="{t} {hy + t} {z}" friction="1 0.005 0.0001" rgba="{rgba}"/>
    <geom name="{prefix}_w" type="box" pos="{cx - hx - t} {cy} {z}" size="{t} {hy + t} {z}" friction="1 0.005 0.0001" rgba="{rgba}"/>"""

def cube_body(name, x, y, rgba):
    return f"""
    <body name="{name}" pos="{x} {y} 0.0205">
      <freejoint name="{name}_free"/>
      <inertial pos="0 0 0" mass="0.064" diaginertia="1.7067e-5 1.7067e-5 1.7067e-5"/>
      <geom name="{name}_geom" type="box" size="0.02 0.02 0.02" friction="1 0.005 0.0001" rgba="{rgba}"/>
    </body>"""

def scene_xml(inner):
    return f"""<mujoco model="task_scene">
  <include file="so101.xml"/>
  <worldbody>
    <light pos="0.2 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" friction="1 0.005 0.0001" rgba="0.25 0.28 0.33 1"/>
    {inner}
    <camera name="main" pos="0.62 -0.52 0.45"/>
  </worldbody>
</mujoco>"""

SCENES = {
    "task1": scene_xml(
        cube_body("cube_green", 0.26, 0.02, "0.1 0.8 0.2 1")
        + walls_xml("box", 0.27, -0.14, 0.06, 0.06, 0.005, 0.07, "0.75 0.75 0.78 1")),
    "task2": scene_xml(
        cube_body("cube_red", 0.21, 0.07, "0.9 0.15 0.15 1")
        + cube_body("cube_blue", 0.21, -0.07, "0.15 0.3 0.95 1")
        + walls_xml("bucket_red", 0.28, 0.15, 0.05, 0.05, 0.005, 0.10, "0.85 0.35 0.35 1")
        + walls_xml("bucket_blue", 0.28, -0.15, 0.05, 0.05, 0.005, 0.10, "0.35 0.45 0.9 1")),
    "task3": scene_xml("""
    <body name="pen" pos="0.26 0 0.005" quat="0.7071068 -0.7071068 0 0">
      <freejoint name="pen_free"/>
      <inertial pos="0 0 0" mass="0.012" diaginertia="1.7e-5 1.7e-5 1.5e-7"/>
      <geom name="pen_geom" type="capsule" size="0.005 0.06" friction="1 0.005 0.0001" rgba="0.95 0.6 0.1 1"/>
    </body>"""),
    "task4": scene_xml("""
    <body name="mug" pos="0.25 0.05 0.0425">
      <freejoint name="mug_free"/>
      <inertial pos="0 0 0" mass="0.15" diaginertia="1.29e-4 1.29e-4 7.7e-5"/>
      <geom name="mug_geom" type="cylinder" size="0.032 0.0425" friction="1 0.005 0.0001" rgba="0.9 0.9 0.92 1"/>
      <geom name="mug_handle_a" type="capsule" pos="0.048 0 0" size="0.006 0.02" contype="0" conaffinity="0" rgba="0.9 0.9 0.92 1"/>
      <geom name="mug_handle_b" type="capsule" pos="0.04 0 0.024" quat="0.7071068 0 0.7071068 0" size="0.006 0.008" contype="0" conaffinity="0" rgba="0.9 0.9 0.92 1"/>
      <geom name="mug_handle_c" type="capsule" pos="0.04 0 -0.024" quat="0.7071068 0 0.7071068 0" size="0.006 0.008" contype="0" conaffinity="0" rgba="0.9 0.9 0.92 1"/>
    </body>
    <geom name="coaster" type="cylinder" pos="0.27 -0.10 0.002" size="0.05 0.002" friction="1 0.005 0.0001" rgba="0.15 0.15 0.18 1"/>"""),
}

# ---------------------------------------------------------------- IK (from test_mujoco.py)
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def solve_ik(m, site_id, qadr, target_pos, R_des, q_seed, use_orient=True):
    d = mujoco.MjData(m)
    lo = m.jnt_range[:5, 0] + 1e-4
    hi = m.jnt_range[:5, 1] - 1e-4
    q = np.clip(np.array(q_seed[:5], float), lo, hi)
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    ok = False
    for _ in range(800):
        d.qpos[qadr[:5]] = q; d.qpos[qadr[5]] = q_seed[5]
        mujoco.mj_forward(m, d)
        p_cur = d.site_xpos[site_id].copy()
        e_pos = target_pos - p_cur
        if use_orient:
            R_cur = d.site_xmat[site_id].reshape(3, 3)
            Re = R_des @ R_cur.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, Re.flatten())
            if quat[0] < 0: quat = -quat
            angle = 2 * np.arccos(np.clip(quat[0], -1, 1))
            e_rot = quat[1:] / max(np.sin(angle / 2), 1e-9) * angle if angle > 1e-9 else np.zeros(3)
        else:
            e_rot = np.zeros(3)
        if np.linalg.norm(e_pos) < 1e-3 and (not use_orient or np.linalg.norm(e_rot) < 0.02):
            ok = True; break
        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        J = np.vstack([jacp[:, qadr[:5]], jacr[:, qadr[:5]]]) if use_orient else jacp[:, qadr[:5]]
        err = np.concatenate([e_pos, e_rot]) if use_orient else e_pos
        lam = 0.03
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(len(err)), err)
        q = np.clip(q + np.clip(dq, -0.1, 0.1), lo, hi)
    q_full = np.zeros(6); q_full[:5] = q; q_full[5] = q_seed[5]
    return q_full, ok

def target_q(segs, t):
    for (t0, t1, qa, qb) in segs:
        if t <= t1:
            s = 0.0 if t1 == t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
            s = s * s * (3 - 2 * s)
            return qa + (qb - qa) * s
    return segs[-1][3]

def aim_camera(m, cam_id, campos, lookat):
    fwd = np.array(lookat, float) - np.array(campos, float)
    fwd /= np.linalg.norm(fwd)
    xc = np.cross(fwd, np.array([0, 0, 1.0])); xc /= np.linalg.norm(xc)
    zc = -fwd
    yc = np.cross(zc, xc)
    R = np.column_stack([xc, yc, zc])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
    m.cam_pos[cam_id] = campos
    m.cam_quat[cam_id] = q

# ---------------------------------------------------------------- pilot
class Pilot:
    def __init__(self, scene_key, ts=0.002, lookat=(0.24, -0.03, 0.04), bodies=()):
        path = f"{MODEL}/scene_{scene_key}.xml"
        with open(path, "w") as f:
            f.write(SCENES[scene_key])
        self.m = mujoco.MjModel.from_xml_path(path)
        self.m.opt.timestep = ts
        self.ts = ts
        self.site_id = self.m.site("gripperframe").id
        self.qadr = [self.m.jnt(n).qposadr[0] for n in JOINTS]
        self.bodies = list(bodies)
        self.bqadr = {b: self.m.jnt(b + "_free").qposadr[0] for b in self.bodies}
        # grasp-quality gate instrumentation (2026-11-16): contact/actuator
        # logging at 100 Hz during run(); pick cycles register in self.cycles
        self.gate = GateRecorder(self.m, self.bodies, self.site_id)
        self.cycles = []
        d0 = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, d0)
        R_home = d0.site_xmat[self.site_id].reshape(3, 3).copy()
        self.R_des = rot_y(np.pi / 2) @ R_home
        self.seed = np.zeros(6); self.seed[5] = GRIP_OPEN
        self.q = np.zeros(6); self.q[5] = GRIP_OPEN
        self.segs = []; self.t = 0.0; self.events = {}
        self.ik_log = []
        aim_camera(self.m, self.m.cam("main").id, self.m.cam_pos[self.m.cam("main").id].copy(), lookat)
        self.cam_home_pos = self.m.cam_pos[self.m.cam("main").id].copy()
        self.cam_home_quat = self.m.cam_quat[self.m.cam("main").id].copy()

    def _ik(self, pos, orient):
        q, ok = solve_ik(self.m, self.site_id, self.qadr, np.array(pos, float),
                         self.R_des, self.seed, orient)
        if ok:
            self.seed = q.copy()
        return q, ok

    def _clamp_grip(self, v):
        """Clamp a gripper command into the joint's own range (2026-11-16,
        S6b forensics: profiles commanded closes down to -0.85 rad against a
        -0.1745 rad floor; the out-of-range command turned the torque-capped
        position servo into a permanent stall clamp. The pilot must never
        ask for a command the joint cannot legally reach)."""
        lo, hi = self.gate.jnt_range
        return float(min(max(float(v), lo), hi))

    def goto(self, pos, dur, grip=None, orient=False, event=None):
        q_t, ok = self._ik(pos, orient)
        self.ik_log.append(dict(pos=[round(float(p), 3) for p in pos], orient=orient, ok=bool(ok)))
        q_t[5] = self.q[5] if grip is None else self._clamp_grip(grip)
        self.segs.append((self.t, self.t + dur, self.q.copy(), q_t))
        self.q = q_t; self.t += dur
        if event: self.events[event] = self.t
        return ok

    def grip_to(self, val, dur, event=None):
        q_t = self.q.copy(); q_t[5] = self._clamp_grip(val)
        self.segs.append((self.t, self.t + dur, self.q.copy(), q_t))
        self.q = q_t; self.t += dur
        if event: self.events[event] = self.t

    def dwell(self, dur, event=None):
        self.segs.append((self.t, self.t + dur, self.q.copy(), self.q.copy()))
        self.t += dur
        if event: self.events[event] = self.t

    def home(self, dur=1.6):
        q_t = np.zeros(6); q_t[5] = GRIP_OPEN
        self.segs.append((self.t, self.t + dur, self.q.copy(), q_t))
        self.q = q_t; self.t += dur

    def run(self, render=False):
        m, ts = self.m, self.ts
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, height=480, width=640) if render else None
        cam_id = m.cam("main").id
        n_steps = int(round(self.t / ts))
        sample_every = max(1, int(round(0.05 / ts)))
        gate_every = max(1, int(round(0.01 / ts)))
        frame_every = max(1, int(round((1 / 30) / ts)))
        snap_every = max(1, int(round(0.1 / ts)))
        log = {"t": [], "tcp": [], "q": [], "bodies": {b: [] for b in self.bodies},
               "quats": {b: [] for b in self.bodies}}
        snaps = {}
        frames = []
        for i in range(n_steps + 1):
            t = i * ts
            ctrl = target_q(self.segs, t)
            d.ctrl[:] = ctrl
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
        log["t"] = np.array(log["t"]); log["tcp"] = np.array(log["tcp"]); log["q"] = np.array(log["q"])
        return log, snaps, frames

    def still(self, snaps, t_snap, lookat, dist=0.30, out=None):
        """Render a close-up from a saved qpos snapshot with a re-aimed camera."""
        m = self.m
        d = mujoco.MjData(m)
        key = min(snaps.keys(), key=lambda k: abs(k - t_snap))
        d.qpos[:] = snaps[key]
        mujoco.mj_forward(m, d)
        cam_id = m.cam("main").id
        off = np.array([0.62, -0.62, 0.42]); off = off / np.linalg.norm(off) * dist
        aim_camera(m, cam_id, np.array(lookat, float) + off, lookat)
        r = mujoco.Renderer(m, height=480, width=640)
        r.update_scene(d, camera="main")
        img = r.render().copy()
        m.cam_pos[cam_id] = self.cam_home_pos; m.cam_quat[cam_id] = self.cam_home_quat
        if out:
            import imageio.v2 as imageio
            imageio.imwrite(out, img)
        return img

# ---------------------------------------------------------------- eval helpers
def settled(log, body, window=0.5):
    t, pos = log["t"], log["bodies"][body]
    m_ = t >= t[-1] - window
    return float(np.max(np.linalg.norm(pos[m_] - pos[-1], axis=1)))

def arm_retreated(log):
    return bool(log["tcp"][-1][2] > 0.10 and abs(log["q"][-1][5] - GRIP_OPEN) < 0.05)

def save_media(tag, frames, ok):
    import imageio.v2 as imageio
    name = f"{tag}_{'PASS' if ok else 'FAIL'}.mp4"
    if frames:
        imageio.mimsave(f"{W}/{name}", frames, fps=30)
    return name

# ---------------------------------------------------------------- tasks
def pick_place(P, obj_xy, grasp_z, dx, close_val, carry_z, place_xy, place_tcp_z,
               close_dur=2.5, tag="", settle_dwell=0.8, body=None):
    """One full pick-and-place cycle. Object centre ends near place_xy when the
    grasp offset dx is the same at pick and place (object-TCP offset cancels)."""
    if body is not None:
        P.cycles.append(dict(body=body, tag=tag, close_cmd=close_val,
                             close_event=f"close_end_{tag}",
                             open_event=f"open_end_{tag}"))
    ox, oy = obj_xy
    P.goto((ox + dx, oy, grasp_z + 0.115), 1.2, grip=GRIP_OPEN)
    P.goto((ox + dx, oy, grasp_z), 1.2, orient=True)
    P.grip_to(close_val, close_dur, event=f"close_end_{tag}")
    P.dwell(0.4)
    P.goto((ox + dx, oy, carry_z), 2.0, grip=close_val)
    P.goto((place_xy[0] + dx, place_xy[1], carry_z), 1.8, grip=close_val)
    P.goto((place_xy[0] + dx, place_xy[1], place_tcp_z), 1.5, grip=close_val, orient=True)
    P.grip_to(GRIP_OPEN, 1.0, event=f"open_end_{tag}")
    P.dwell(settle_dwell)
    P.goto((place_xy[0] + dx, place_xy[1], carry_z), 1.5, grip=GRIP_OPEN)

def task1(cfg):
    P = Pilot("task1", bodies=["cube_green"], lookat=(0.26, -0.06, 0.03))
    dx = cfg.get("dx", -0.01)
    pick_place(P, (0.26, 0.02), 0.02, dx, GRIP_CLOSE_CUBE, cfg.get("carry_z", 0.135),
               (0.27, -0.14), cfg.get("place_tcp_z", 0.028), tag="t1", body="cube_green")
    P.home(); P.dwell(1.2)
    log, snaps, frames = P.run(render=cfg.get("render", False))
    pos = log["bodies"]["cube_green"][-1]
    res = dict(task="task1", dx=dx,
               final_cube=np.round(pos, 4).tolist(),
               inside_xy=bool(abs(pos[0] - 0.27) <= 0.04 and abs(pos[1] + 0.14) <= 0.04),
               below_rim=bool(0.015 <= pos[2] <= 0.06),
               settled_mm=round(settled(log, "cube_green") * 1000, 2),
               arm_retreated=arm_retreated(log))
    res["ok"] = bool(res["inside_xy"] and res["below_rim"] and res["settled_mm"] < 2.0 and res["arm_retreated"])
    return P, log, snaps, frames, res

def task2(cfg):
    P = Pilot("task2", bodies=["cube_red", "cube_blue"], lookat=(0.25, 0.0, 0.03))
    dx = cfg.get("dx", -0.01)
    carry_z = cfg.get("carry_z", 0.16)
    place_z = cfg.get("place_tcp_z", 0.045)
    pick_place(P, (0.21, 0.07), 0.02, dx, GRIP_CLOSE_CUBE, carry_z, (0.28, 0.15), place_z, tag="red", body="cube_red")
    pick_place(P, (0.21, -0.07), 0.02, dx, GRIP_CLOSE_CUBE, carry_z, (0.28, -0.15), place_z, tag="blue", body="cube_blue")
    P.home(); P.dwell(1.2)
    log, snaps, frames = P.run(render=cfg.get("render", False))
    res = dict(task="task2", dx=dx, carry_z=carry_z, place_tcp_z=place_z)
    ok_all = True
    for body, centre in (("cube_red", (0.28, 0.15)), ("cube_blue", (0.28, -0.15))):
        pos = log["bodies"][body][-1]
        inside = bool(abs(pos[0] - centre[0]) <= 0.03 and abs(pos[1] - centre[1]) <= 0.03)
        below = bool(pos[2] <= 0.10)
        st = settled(log, body)
        res[body] = dict(final=np.round(pos, 4).tolist(), inside=inside, below_rim=below,
                         settled_mm=round(st * 1000, 2))
        ok_all &= inside and below and st < 0.002
    res["arm_retreated"] = arm_retreated(log)
    res["ok"] = bool(ok_all and res["arm_retreated"])
    return P, log, snaps, frames, res

def pen_attempt(cfg, render=False, hold=3.0):
    P = Pilot("task3", bodies=["pen"], lookat=(0.26, 0.0, 0.02))
    dx = cfg.get("dx", -0.01); gz = cfg.get("grasp_z", 0.006)
    cv = cfg.get("close_val", 0.1); cd = cfg.get("close_dur", 3.0)
    px, py = 0.26, 0.0
    P.goto((px + dx, py, 0.10), 1.2, grip=GRIP_OPEN)
    P.goto((px + dx, py, gz), 1.4, orient=True)
    P.grip_to(cv, cd, event="close_end")
    P.dwell(0.3)
    P.goto((px + dx, py, cfg.get("lift_z", 0.12)), 2.2, grip=cv, event="lift_end")
    P.dwell(hold, event="hold_end")
    log, snaps, frames = P.run(render=render)
    t = log["t"]; pen = log["bodies"]["pen"]; tcp = log["tcp"]
    t0 = P.events["lift_end"]
    win = (t >= t0) & (t <= t0 + hold)
    rel = pen - tcp
    drift = float(np.max(np.linalg.norm(rel[win] - rel[win][0], axis=1))) if win.any() else -1
    res = dict(task="task3", dx=dx, grasp_z=gz, close_val=cv, close_dur=cd,
               hold_min_pen_z=round(float(np.min(pen[win, 2])), 4) if win.any() else -1,
               hold_drift_mm=round(drift * 1000, 2),
               final_pen=np.round(pen[-1], 4).tolist())
    res["ok"] = bool(win.any() and res["hold_min_pen_z"] >= 0.05 and drift <= 0.02)
    return P, log, snaps, frames, res

def pen_scan():
    out = []
    for cv in (0.5, 0.3, 0.1, -0.05, -0.10):
        for gz in (0.004, 0.008):
            for dx in (-0.01,):
                _, log, _, _, res = pen_attempt(dict(close_val=cv, grasp_z=gz, dx=dx), hold=1.0)
                out.append(res)
                print(json.dumps(res), flush=True)
    return out

def task4(cfg):
    P = Pilot("task4", bodies=["mug"], lookat=(0.26, -0.02, 0.03))
    dx = cfg.get("dx", -0.01)
    pick_place(P, (0.25, 0.05), cfg.get("grasp_z", 0.0425), dx, cfg.get("close_val", 0.3),
               cfg.get("carry_z", 0.15), (0.27, -0.10), cfg.get("place_tcp_z", 0.0505),
               close_dur=cfg.get("close_dur", 2.5), tag="t4", body="mug")
    P.home(); P.dwell(1.2)
    log, snaps, frames = P.run(render=cfg.get("render", False))
    pos = log["bodies"]["mug"][-1]
    quat = log["quats"]["mug"][-1]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
    axis = R @ np.array([0, 0, 1.0])
    tilt = float(np.degrees(np.arccos(np.clip(axis[2], -1, 1))))
    res = dict(task="task4", dx=dx, close_val=cfg.get("close_val", 0.3),
               final_mug=np.round(pos, 4).tolist(),
               coaster_err_mm=round(float(np.linalg.norm(pos[:2] - np.array([0.27, -0.10]))) * 1000, 1),
               tilt_deg=round(tilt, 1),
               rest_z_err_mm=round(abs(float(pos[2]) - 0.0465) * 1000, 1),
               settled_mm=round(settled(log, "mug") * 1000, 2),
               arm_retreated=arm_retreated(log))
    res["ok"] = bool(res["coaster_err_mm"] <= 30 and tilt <= 15 and res["rest_z_err_mm"] <= 10
                     and res["settled_mm"] < 2.0 and res["arm_retreated"])
    return P, log, snaps, frames, res

TASKS = {"task1": task1, "task2": task2, "task4": task4}

if __name__ == "__main__":
    task = sys.argv[1]
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    t0 = time.time()
    if task == "pen_scan":
        pen_scan()
        print("wall_s", round(time.time() - t0, 1)); sys.exit(0)
    if task == "task3":
        P, log, snaps, frames, res = pen_attempt(cfg, render=cfg.get("render", False),
                                                 hold=cfg.get("hold", 3.0))
    else:
        P, log, snaps, frames, res = TASKS[task](cfg)
    tag = dict(task1="task1_green_box", task2="task2_two_buckets",
               task3="task3_pen", task4="task4_mug")[task]
    res["mp4"] = save_media(tag, frames, res["ok"]) if frames else None
    body0 = {"task1": "cube_green", "task2": "cube_red", "task3": "pen", "task4": "mug"}[task]
    if frames:
        t_close = P.events.get("close_end_" + ("t1" if task == "task1" else "red" if task == "task2" else "t4"),
                               P.events.get("close_end", 3.0))
        P.still(snaps, t_close, log["bodies"][body0][np.argmin(np.abs(log["t"] - t_close))],
                dist=0.26, out=f"{W}/{tag}_closeup_grasp.png")
        P.still(snaps, log["t"][-1], log["bodies"][body0][-1],
                dist=0.34, out=f"{W}/{tag}_closeup_final.png")
        res["stills"] = [f"{tag}_closeup_grasp.png", f"{tag}_closeup_final.png"]
    res["ik_failures"] = [e for e in P.ik_log if not e["ok"]]
    with open(f"{W}/results_{tag}_{cfg.get('tag', 'run')}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res), flush=True)
    print("TASK_OK" if res["ok"] else "TASK_FAIL", flush=True)
    print("wall_s", round(time.time() - t0, 1))
