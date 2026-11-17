"""Description->scene harness for the zk0 Stage 0 test (MuJoCo 4.0.1,
Menagerie robotstudio_so101, stock). Generalizes the multitask battery
pilot: DLS IK on the gripperframe site, smoothstep joint-space segments,
contact-and-friction grasps only (no welds / no object scripting).

Adds: spec-driven scene builder (tables, backdrops, library meshes,
spec-layer primitives), per-object grasp profiles incl. wrist roll,
scene verification (settle probe + envelope + spec-vs-measured), and
30 Hz recording of state/action + 4 cameras (wide, close, node A/B)
for LeRobot training capture.
"""
import os, json, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from grasp_gate import GateRecorder

# [slice-4 vendoring patch — see PROVENANCE.md] paths were absolute Spark workdir constants; now
# package-relative, with the model dir env-overridable (the
# runner points it at a writable copy of the staged arm model).
W = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("ZK0_HARDENED_MODEL_DIR", W + "/model")

def load_model(xml, tag="scene"):
    """MJCF <include> resolves relative to the XML file, so scenes are
    written into the model dir and loaded from path."""
    path = f"{MODEL}/desc_{tag}.xml"
    with open(path, "w") as f:
        f.write(xml)
    return mujoco.MjModel.from_xml_path(path)
YCB = W + "/assets/ycb"
# [slice-4 vendoring patch — see PROVENANCE.md] was an absolute path to another workdir;
# unused on the wall-pinch path.
TRIPO = W + "/assets"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIP_OPEN = 1.74533
# Node idle pose: arm raised and folded clear of the workspace. At q=0 the
# SO-101 lies across the table and disturbs objects spawned beneath it, so
# verification and runs both start (and end) here instead.
PARK = np.array([0.0, -1.05, 1.05, 0.35, 0.0, 0.9])
HOME_Q = PARK.copy(); HOME_Q[5] = GRIP_OPEN
FRIC_TAIL = "0.005 0.0001"   # torsional, rolling (battery-proven values)

# ---------------------------------------------------------------- obj audit
def obj_audit(path):
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                fs.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    V = np.array(vs); F = np.array(fs)
    lo, hi = V.min(0), V.max(0)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    # centroid of the (closed) mesh via tetrahedra
    cen = ((a + b + c) / 4.0 * (np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0)[:, None]).sum(0) / vol
    return dict(lo=lo, hi=hi, dims=hi - lo, vol=abs(vol), cen=cen)

def quat_from_euler(rx, ry, rz):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([rx, ry, rz]), "xyz")
    return q

def quat_axis_angle(axis, ang):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array(axis, float), ang)
    return q

def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out

# ---------------------------------------------------------------- mesh prep
def load_obj_verts(path):
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                fs.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(vs), np.array(fs)

def pca_align_quat(verts):
    """Rotation (as quat) mapping the mesh's principal axes to world axes:
    PC1 (longest) -> +Y, PC2 -> +X, PC3 (thinnest) -> +Z. For elongated
    objects that should lie flat (screwdriver, marker, banana)."""
    c = verts.mean(0)
    X = verts - c
    cov = X.T @ X / len(X)
    w, V = np.linalg.eigh(cov)          # ascending eigenvalues
    R = np.column_stack([V[:, 1], V[:, 2], V[:, 0]])  # x=PC2, y=PC1, z=PC3
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q

def prep_mesh(vert_paths, quat):
    """Combine one or more OBJ files, rotate by quat, and return the placement
    data mesh_body needs: geom offset (=-rotated VOLUMETRIC centroid, so the
    body origin sits at the centre of mass, not the vertex mean), rest_z
    (body-origin height so the mesh bottom touches z=0), rotated bbox dims."""
    Vs, Fs = [], []
    off = 0
    for p in vert_paths:
        v, f = load_obj_verts(p)
        Vs.append(v); Fs.append(f + off); off += len(v)
    V = np.vstack(Vs); F = np.vstack(Fs)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    Vr = V @ R.T
    a, b, c = Vr[F[:, 0]], Vr[F[:, 1]], Vr[F[:, 2]]
    w = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    vol = w.sum()
    cen_r = ((a + b + c) / 4.0 * w[:, None]).sum(0) / vol if abs(vol) > 1e-12 else Vr.mean(0)
    lo, hi = Vr.min(0), Vr.max(0)
    # geom_pos is expressed in the BODY frame, whose rotation maps raw verts
    # v -> R v; the offset must therefore be the UNROTATED centroid
    # (-R^T cen_r). Passing -cen_r here (the rotated centroid) misplaced the
    # mesh inside its body for every non-identity orientation — the hidden
    # cause of the pear/mug/screwdriver settle failures.
    cen_raw = R.T @ cen_r
    return dict(geom_pos=-cen_raw, rest_z=float(cen_r[2] - lo[2]),
                dims=hi - lo, lo=lo, hi=hi, cen_r=cen_r)

def ycb_obj(name):
    return f"{YCB}/{name}/google_16k/textured.obj"

def ycb_tex(name):
    return f"{YCB}/{name}/google_16k/texture_map.png"

def box_inertia(mass, dims):
    return [mass * (dims[1]**2 + dims[2]**2) / 12,
            mass * (dims[0]**2 + dims[2]**2) / 12,
            mass * (dims[0]**2 + dims[1]**2) / 12]

# ---------------------------------------------------------------- scene XML
def walls_xml(prefix, cx, cy, hx, hy, t, h, rgba, mu):
    z = h / 2
    fr = f"{mu} {FRIC_TAIL}"
    return f"""
    <geom name="{prefix}_n" type="box" pos="{cx} {cy + hy + t} {z}" size="{hx + t} {t} {z}" friction="{fr}" rgba="{rgba}"/>
    <geom name="{prefix}_s" type="box" pos="{cx} {cy - hy - t} {z}" size="{hx + t} {t} {z}" friction="{fr}" rgba="{rgba}"/>
    <geom name="{prefix}_e" type="box" pos="{cx + hx + t} {cy} {z}" size="{t} {hy + t} {z}" friction="{fr}" rgba="{rgba}"/>
    <geom name="{prefix}_w" type="box" pos="{cx - hx - t} {cy} {z}" size="{t} {hy + t} {z}" friction="{fr}" rgba="{rgba}"/>"""

def tray_xml(prefix, cx, cy, hx, hy, rgba, mu):
    """Shallow tray: floor plate + 25 mm rim."""
    t, h = 0.004, 0.025
    fr = f"{mu} {FRIC_TAIL}"
    return f"""
    <geom name="{prefix}_floor" type="box" pos="{cx} {cy} 0.003" size="{hx + t} {hy + t} 0.003" friction="{fr}" rgba="{rgba}"/>""" \
        + walls_xml(prefix, cx, cy, hx, hy, t, h, rgba, mu)

def table_xml(t):
    spec = t
    cx, cy = t.get("center", (0.05, 0.0))
    hx, hy = t["half"]
    thick = t.get("thick", 0.03)
    h = t.get("height", 0.74)
    mu = t["mu"]
    fr = f"{mu} {FRIC_TAIL}"
    # Collision slab is 0.40 thick whatever the visual thickness: with a thin
    # (30 mm) box, slim objects (marker, screwdriver) tunnelled into the box
    # interior and came to rest embedded in the table. The visible top is a
    # separate non-colliding veneer of the specified thickness.
    xml = f"""
    <geom name="tabletop" type="box" pos="{cx} {cy} {-0.20}" size="{hx} {hy} {0.20}" friction="{fr}" rgba="{t['rgba']}"/>
    <geom name="tabletop_veneer" type="box" pos="{cx} {cy} {-thick/2 + 0.0008}" size="{hx} {hy} {thick/2}" rgba="{t['rgba']}" contype="0" conaffinity="0"/>"""
    li = 0
    for lx in (-hx + 0.05, hx - 0.05):
        for ly in (-hy + 0.05, hy - 0.05):
            li += 1
            xml += f"""
    <geom name="leg{li}" type="box" pos="{cx + lx} {cy + ly} {-(h + thick)/2 - 0.0}" size="0.025 0.025 {(h - thick)/2}" rgba="0.3 0.3 0.32 1" contype="0" conaffinity="0"/>"""
    xml += f"""
    <geom name="floor" type="plane" pos="0 0 {-h}" size="0 0 0.05" friction="{fr}" rgba="{t.get('floor_rgba','0.22 0.23 0.26 1')}"/>
    <geom name="wall_back" type="plane" pos="-0.42 0 0" size="0 0 0.05" quat="0.7071068 0 0.7071068 0" rgba="{t.get('wall_rgba','0.85 0.83 0.78 1')}" contype="0" conaffinity="0"/>
    <light pos="{t.get('light_pos','-0.4 -0.6 1.6')}" dir="{t.get('light_dir','0.25 0.25 -1')}" directional="true"/>
    <camera name="main" pos="0.66 -0.55 0.48"/>
    <camera name="close" pos="0.47 -0.26 0.17"/>
    <camera name="front" pos="0.52 -0.02 0.34"/>
    <camera name="top" pos="0.24 0.0 0.85"/>"""
    return xml

def mesh_body(name, meshfile, pos, quat, mass, diaginertia, rgba, mu,
              visual_mesh=None, texfile=None, hull_files=None, mesh_scale=None,
              geom_pos=(0, 0, 0)):
    """Free body with a mesh visual + collision. If hull_files given, collision
    comes from those convex hull meshes and the visual mesh is non-colliding.
    geom_pos offsets every geom inside the body (mesh recentering)."""
    sc = f' scale="{mesh_scale} {mesh_scale} {mesh_scale}"' if mesh_scale else ""
    gp = f' pos="{geom_pos[0]:.5f} {geom_pos[1]:.5f} {geom_pos[2]:.5f}"'
    assets = f'<mesh name="{name}_vis" file="{visual_mesh or meshfile}"{sc}/>'
    if texfile:
        assets = f'<texture name="{name}_tex" type="2d" file="{texfile}"/>' + assets
        mat = f'<material name="{name}_mat" texture="{name}_tex" texuniform="false"/>'
        assets += mat
        vis_mat = f' material="{name}_mat"'
    else:
        vis_mat = ""
    if hull_files:
        geoms = ""
        for i, hf in enumerate(hull_files):
            assets += f'<mesh name="{name}_h{i}" file="{hf}"{sc}/>'
            geoms += f'<geom name="{name}_h{i}" type="mesh" mesh="{name}_h{i}"{gp} friction="{mu} {FRIC_TAIL}" rgba="0 0 0 0"/>'
        geoms += f'<geom name="{name}_vis" type="mesh" mesh="{name}_vis"{gp} contype="0" conaffinity="0" rgba="{rgba}"{vis_mat}/>'
    else:
        geoms = f'<geom name="{name}_geom" type="mesh" mesh="{name}_vis"{gp} friction="{mu} {FRIC_TAIL}" rgba="{rgba}"{vis_mat}/>'
    q = " ".join(f"{v:.6f}" for v in quat)
    di = " ".join(f"{v:.3e}" for v in diaginertia)
    body = f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q}">
      <freejoint name="{name}_free"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{di}"/>
      {geoms}
    </body>"""
    return assets, body

def prim_body(name, kind, size, pos, quat, mass, rgba, mu):
    """Spec-layer primitive body. size semantics by kind:
    box: half extents; sphere: radius; cylinder: (radius, half-height)."""
    q = " ".join(f"{v:.6f}" for v in quat)
    if kind == "box":
        di = [mass * (size[1]**2 + size[2]**2) * 4 / 12,
              mass * (size[0]**2 + size[2]**2) * 4 / 12,
              mass * (size[0]**2 + size[1]**2) * 4 / 12]
        sz = f"{size[0]} {size[1]} {size[2]}"
    elif kind == "sphere":
        di = [0.4 * mass * size**2] * 3
        sz = f"{size}"
    elif kind == "cylinder":
        di = [mass * (3 * size[0]**2 + 4 * size[1]**2) / 12,
              mass * (3 * size[0]**2 + 4 * size[1]**2) / 12,
              0.5 * mass * size[0]**2]
        sz = f"{size[0]} {size[1]}"
    di_s = " ".join(f"{v:.3e}" for v in di)
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q}">
      <freejoint name="{name}_free"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{di_s}"/>
      <geom name="{name}_geom" type="{kind}" size="{sz}" friction="{mu} {FRIC_TAIL}" rgba="{rgba}"/>
    </body>"""

def scene_xml(table_spec, inner_assets, inner_bodies, inner_static=""):
    return f"""<mujoco model="desc_scene">
  <compiler meshdir="{MODEL}/assets"/>
  <include file="so101.xml"/>
  <asset>
{inner_assets}
  </asset>
  <worldbody>
{table_xml(table_spec)}
{inner_static}
{inner_bodies}
  </worldbody>
</mujoco>"""

# ---------------------------------------------------------------- IK (battery)
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _ik_run(m, site_id, qadr, target_pos, R_des, q0, lo, hi, mode, iters=400):
    """One DLS IK run. mode: 'full' (position + full orientation), 'zaxis'
    (position + gripper z-axis direction only, yaw free), 'pos' (position)."""
    d = mujoco.MjData(m)
    q = np.clip(np.array(q0[:5], float), lo, hi)
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    grip = q0[5]
    best = (np.inf, q.copy())
    for _ in range(iters):
        d.qpos[qadr[:5]] = q; d.qpos[qadr[5]] = grip
        mujoco.mj_forward(m, d)
        p_cur = d.site_xpos[site_id].copy()
        e_pos = target_pos - p_cur
        R_cur = d.site_xmat[site_id].reshape(3, 3)
        if mode == "full":
            Re = R_des @ R_cur.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, Re.flatten())
            if quat[0] < 0: quat = -quat
            angle = 2 * np.arccos(np.clip(quat[0], -1, 1))
            e_rot = quat[1:] / max(np.sin(angle / 2), 1e-9) * angle if angle > 1e-9 else np.zeros(3)
            score = np.linalg.norm(e_pos) + 0.05 * np.linalg.norm(e_rot)
            if score < best[0]: best = (score, q.copy())
            if np.linalg.norm(e_pos) < 2e-3 and np.linalg.norm(e_rot) < 0.035:
                return q, True, score
            mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
            J = np.vstack([jacp[:, qadr[:5]], jacr[:, qadr[:5]]])
            err = np.concatenate([e_pos, e_rot])
        elif mode == "zaxis":
            z_cur = R_cur[:, 2]; z_des = R_des[:, 2]
            e_z = np.cross(z_cur, z_des)
            score = np.linalg.norm(e_pos) + 0.05 * np.linalg.norm(e_z)
            if score < best[0]: best = (score, q.copy())
            if np.linalg.norm(e_pos) < 2e-3 and np.linalg.norm(e_z) < 0.045:
                return q, True, score
            mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
            # d(z_cur)/dq rows: -[z_des]x * J_rot gives d(cross(z_cur,z_des))/dq
            Zx = np.array([[0, -z_des[2], z_des[1]], [z_des[2], 0, -z_des[0]],
                           [-z_des[1], z_des[0], 0]])
            J = np.vstack([jacp[:, qadr[:5]], Zx @ jacr[:, qadr[:5]]])
            err = np.concatenate([e_pos, e_z])
        else:
            score = np.linalg.norm(e_pos)
            if score < best[0]: best = (score, q.copy())
            if score < 2e-3:
                return q, True, score
            mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
            J = jacp[:, qadr[:5]]
            err = e_pos
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(len(err)), err)
        q = np.clip(q + np.clip(dq, -0.15, 0.15), lo, hi)
    return best[1], False, best[0]

def solve_ik(m, site_id, qadr, target_pos, R_des, q_seed, use_orient=True):
    """Multi-start IK. The original single-seed DLS (battery heritage) failed
    systematically at far/low orient targets: wrist_flex saturates at its
    1.66 rad limit and the iteration stalls in a bad basin (measured: 7-30 mm
    position error on container-place descents across S1/S4/S5/S6b). Strategy:
    try the chained seed plus canonical seeds in 'full' mode; if none
    converges, fall back to 'zaxis' mode (gripper axis down, yaw free — all a
    top-down grasp actually needs). Returns the best solution found."""
    lo = m.jnt_range[:5, 0] + 1e-4
    hi = m.jnt_range[:5, 1] - 1e-4
    seeds = [np.array(q_seed, float),
             np.array([0.0, -1.05, 1.05, 0.35, 0.0, q_seed[5]]),
             np.array([0.0, 0.0, 0.0, 0.0, 0.0, q_seed[5]]),
             np.array([0.3, -0.5, 0.8, 0.6, 0.0, q_seed[5]]),
             np.array([-0.3, -0.9, 1.3, 0.9, 0.0, q_seed[5]])]
    modes = ["full", "zaxis"] if use_orient else ["pos"]
    best_q, best_score = None, np.inf
    for mode in modes:
        for sd in seeds:
            q, ok, score = _ik_run(m, site_id, qadr, target_pos, R_des, sd, lo, hi, mode)
            if ok:
                q_full = np.zeros(6); q_full[:5] = q; q_full[5] = q_seed[5]
                return q_full, True
            if score < best_score:
                best_score, best_q = score, q
    q_full = np.zeros(6); q_full[:5] = best_q; q_full[5] = q_seed[5]
    return q_full, False

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
    def __init__(self, xml, bodies=(), lookat=(0.25, 0.0, 0.04), ts=0.002):
        self.m = load_model(xml, tag="pilot")
        self.q0 = HOME_Q.copy()   # runs start from the idle pose, like verification
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
        self.seed = self.q0.copy(); self.seed[5] = GRIP_OPEN
        self.q = self.q0.copy()
        self.segs = []; self.t = 0.0; self.events = {}
        self.ik_log = []
        aim_camera(self.m, self.m.cam("main").id, self.m.cam_pos[self.m.cam("main").id].copy(), lookat)
        aim_camera(self.m, self.m.cam("close").id, self.m.cam_pos[self.m.cam("close").id].copy(), lookat)
        aim_camera(self.m, self.m.cam("front").id, self.m.cam_pos[self.m.cam("front").id].copy(), lookat)
        self.m.cam_fovy[self.m.cam("top").id] = 55

    def _ik(self, pos, orient, roll=0.0):
        R = self.R_des
        if roll:
            R = self.R_des @ np.array([[1, 0, 0],
                                       [0, math.cos(roll), -math.sin(roll)],
                                       [0, math.sin(roll), math.cos(roll)]])
        q, ok = solve_ik(self.m, self.site_id, self.qadr, np.array(pos, float), R, self.seed, orient)
        if ok:
            self.seed = q.copy()
        return q, ok

    def _clamp_grip(self, v):
        """Clamp a gripper command into the joint's own range (2026-11-16,
        S6b forensics: the mug profile commanded -0.65 rad against a
        -0.1745 rad floor; the out-of-range command turned the torque-capped
        position servo into a permanent stall clamp. The pilot must never
        ask for a command the joint cannot legally reach)."""
        lo, hi = self.gate.jnt_range
        return float(min(max(float(v), lo), hi))

    def goto(self, pos, dur, grip=None, orient=False, roll=0.0, event=None):
        q_t, ok = self._ik(pos, orient, roll)
        if not ok and orient:
            # Corridor retry: a failed orient-IK returns a best-effort q that
            # may sit in a different elbow configuration — interpolating to it
            # sweeps the arm through a wild arc (it flung a grasped mug in
            # S1). Instead: translate to above the target with position-only
            # IK (robust), then descend vertically with orientation, seeded
            # from that nearby solution.
            above = (pos[0], pos[1], pos[2] + 0.07)
            q_a, ok_a = self._ik(above, False, roll)
            if ok_a:
                q_a[5] = self.q[5] if grip is None else self._clamp_grip(grip)
                self.segs.append((self.t, self.t + dur * 0.55, self.q.copy(), q_a))
                self.q = q_a; self.t += dur * 0.55
                q_t2, ok2 = self._ik(pos, True, roll)
                self.ik_log.append(dict(pos=[round(float(p), 3) for p in pos],
                                        orient=orient, roll=round(roll, 2),
                                        ok=bool(ok2), corridor=True))
                if ok2:
                    q_t, ok = q_t2, True
        self.ik_log.append(dict(pos=[round(float(p), 3) for p in pos], orient=orient,
                                roll=round(roll, 2), ok=bool(ok)))
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
        q_t = HOME_Q.copy()
        self.segs.append((self.t, self.t + dur, self.q.copy(), q_t))
        self.q = q_t; self.t += dur

    def run(self, record_cams=None, render=True):
        """record_cams: (nameA, nameB) node cameras to capture at 320x240, 30 Hz.
        Captures main+close at 640x480 for founder video unless render=False."""
        m, ts = self.m, self.ts
        d = mujoco.MjData(m)
        rend = {}
        if render:
            for cname, h, w in (("main", 480, 640), ("close", 480, 640),
                                (record_cams[0], 240, 320), (record_cams[1], 240, 320)):
                rend[cname] = mujoco.Renderer(m, height=h, width=w)
        d.qpos[:6] = self.q0
        d.qvel[:] = 0.0
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

def pick_place(P, obj_xy, grasp_z, dx, close_val, carry_z, place_xy, place_tcp_z,
               dy=0.0, roll=0.0, close_dur=2.5, tag="", settle_dwell=0.8,
               descend_dur=1.2, final_dur=1.5, approach_grip=None, carry_dur=1.8,
               hover_dwell=0.0, body=None):
    """One pick-and-place cycle (battery-proven shape). grasp_z/place_tcp_z are
    TCP heights; the object-TCP offset cancels when dx/dy match at pick/place.
    approach_grip: gripper value to close to DURING the descent — with the
    gripper fully open the jaw tips are the lowest part of the tool and hit
    the table at TCP~0.054, so low/round objects can only be contacted above
    their equator and squirt out; pre-closing retracts the tips and lets the
    TCP reach below the equator before the final close."""
    if body is not None:
        P.cycles.append(dict(body=body, tag=tag, close_cmd=close_val,
                             close_event=f"close_end_{tag}",
                             open_event=f"open_end_{tag}"))
    ox, oy = obj_xy
    P.goto((ox + dx, oy + dy, grasp_z + 0.115), 1.2, grip=GRIP_OPEN, roll=roll)
    if hover_dwell:
        P.dwell(hover_dwell)   # let the PD settle before descending (a pushed cube in S5 traced to descent overshoot after the long travel)
    P.goto((ox + dx, oy + dy, grasp_z), descend_dur, orient=True, roll=roll,
           grip=approach_grip)
    P.grip_to(close_val, close_dur, event=f"close_end_{tag}")
    P.dwell(0.4)
    P.goto((ox + dx, oy + dy, carry_z), 2.0, grip=close_val, roll=roll)
    P.goto((place_xy[0] + dx, place_xy[1] + dy, carry_z), carry_dur, grip=close_val, roll=roll)
    P.goto((place_xy[0] + dx, place_xy[1] + dy, place_tcp_z), final_dur, grip=close_val, orient=True, roll=roll)
    P.grip_to(GRIP_OPEN, 1.0, event=f"open_end_{tag}")
    P.dwell(settle_dwell)
    P.goto((place_xy[0] + dx, place_xy[1] + dy, carry_z), 1.5, grip=GRIP_OPEN, roll=roll)

# ---------------------------------------------------------------- verify
def settle_probe(xml, bodies_spec, dur=2.5, ts=0.002):
    """bodies_spec: {name: (spec_pos, spec_quat)}. Steps the scene with the arm
    parked at home; returns per-body displacement / tilt / final z."""
    m = load_model(xml, tag="settle")
    m.opt.timestep = ts
    d = mujoco.MjData(m)
    d.qpos[:6] = HOME_Q
    d.ctrl[:] = HOME_Q
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
            # pin objects while the arm's hold transient settles, then release:
            # verification measures the object's own stability, not the brush
            # of a sagging arm in the first fraction of a second
            for b, a in qadr.items():
                d.qpos[a:a + 7] = spawn[b]
                d.qvel[vadr[b]:vadr[b] + 6] = 0.0
        mujoco.mj_step(m, d)
    out = {}
    for b, a in qadr.items():
        p1, q1 = d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy()
        p0, q0 = start[b]
        # tilt between start and end orientation
        dq = np.zeros(4); mujoco.mju_mulQuat(dq, q1, [-q0[0], q0[1], q0[2], q0[3]])
        tilt = 2 * np.arccos(np.clip(abs(dq[0]), -1, 1))
        out[b] = dict(disp_mm=round(float(np.linalg.norm(p1 - p0)) * 1000, 2),
                      tilt_deg=round(float(np.degrees(tilt)), 2),
                      final_pos=np.round(p1, 4).tolist(),
                      spec_pos=np.round(np.array(bodies_spec[b][0]), 4).tolist())
    return out

def settled(log, body, window=0.5):
    t, pos = log["t"], log["bodies"][body]
    m_ = t >= t[-1] - window
    return float(np.max(np.linalg.norm(pos[m_] - pos[-1], axis=1)))

def tilt_deg(log, body, q0):
    q1 = log["quats"][body][-1]
    dq = np.zeros(4); mujoco.mju_mulQuat(dq, q1, [-q0[0], q0[1], q0[2], q0[3]])
    return float(np.degrees(2 * np.arccos(np.clip(abs(dq[0]), -1, 1))))

def arm_retreated(log):
    """Run ends with the arm back at the idle (park, gripper open) pose."""
    q = np.array(log["q"])
    return bool(np.all(np.abs(q[-20:] - HOME_Q) < 0.15, axis=1).all())
