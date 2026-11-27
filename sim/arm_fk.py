"""VM-side FK/IK for the SO-101 arm, parsed generically from the MuJoCo arm XML.

Used ONLY to generate trajectory proposals (Decision 113: trajectories are
proposed by the frontier model; physics executions happen on the DGX under
the job guard). No physics is simulated here -- FK is pure kinematics over
the body tree: T_body = T_parent * Trans(pos) * Rot(quat) * Rot(axis, q).
"""
import math
import xml.etree.ElementTree as ET
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _vec(s, default):
    if s is None:
        return np.array(default, float)
    return np.array([float(x) for x in s.split()], float)


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_R(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


class Body:
    def __init__(self, el):
        self.name = el.get("name")
        self.pos = _vec(el.get("pos"), [0, 0, 0])
        self.quat = _vec(el.get("quat"), [1, 0, 0, 0])
        self.joint = None  # (name, axis)
        self.sites = {}
        self.children = []
        for ch in el:
            if ch.tag == "joint":
                self.joint = (ch.get("name"), _vec(ch.get("axis"), [0, 0, 1]))
            elif ch.tag == "site":
                self.sites[ch.get("name")] = (_vec(ch.get("pos"), [0, 0, 0]),
                                              _vec(ch.get("quat"), [1, 0, 0, 0]))
            elif ch.tag == "body":
                self.children.append(Body(ch))


class ArmFK:
    def __init__(self, xml_path):
        root = ET.parse(xml_path).getroot()
        wb = root.find("worldbody")
        self.roots = [Body(b) for b in wb if b.tag == "body"]
        self.by_name = {}
        def walk(b):
            if b.name:
                self.by_name[b.name] = b
            for c in b.children:
                walk(c)
        for r in self.roots:
            walk(r)

    def _fk_body(self, body, q, T_parent):
        R = T_parent[:3, :3] @ quat_to_R(body.quat)
        p = T_parent[:3, 3] + T_parent[:3, :3] @ body.pos
        if body.joint is not None:
            jn, axis = body.joint
            if jn in q:
                R = R @ axis_angle_R(axis, q[jn])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        out = {body.name: T} if body.name else {}
        for c in body.children:
            out.update(self._fk_body(c, q, T))
        return out

    def fk(self, q_deg):
        """q_deg: dict joint name -> degrees (arm joints). Returns dict body name -> 4x4."""
        q = {k: math.radians(v) for k, v in q_deg.items()}
        T0 = np.eye(4)
        out = {}
        for r in self.roots:
            out.update(self._fk_body(r, q, T0))
        return out

    def site_pose(self, q_deg, body_name="gripper", site_name="gripperframe"):
        T = self.fk(q_deg)[body_name]
        body = self.by_name[body_name]
        spos, squat = body.sites[site_name]
        Ts = np.eye(4)
        Ts[:3, :3] = quat_to_R(squat)
        Ts[:3, 3] = spos
        return T @ Ts

    def site_pos(self, q_deg, **kw):
        return self.site_pose(q_deg, **kw)[:3, 3]


JOINT_RANGES_DEG = {   # from so101_new_calib_padded.xml joint range attrs (rad -> deg)
    "shoulder_pan": (-110.0, 110.0), "shoulder_lift": (-105.0, 100.0),
    "elbow_flex": (-96.8, 110.0), "wrist_flex": (-95.0, 102.0),
    "wrist_roll": (-157.2, 162.9), "gripper": (-10.0, 100.0),
}


def ik_site_full(fk, target_pos, target_R, q0_deg, iters=150, w_rot=0.03):
    """Numerical IK for the 5 arm joints matching a site pose in least squares
    (5 DOF vs 6 constraints). Position is weighted far above orientation
    (w_rot per radian vs position per metre): placement precision is what the
    seat test reads; orientation is kept near the target family, not exact.
    Returns (q_deg dict, pos_err_m, rot_err_rad)."""
    from scipy.spatial.transform import Rotation as Rot
    names = JOINTS[:5]
    lo = np.array([JOINT_RANGES_DEG[n][0] for n in names])
    hi = np.array([JOINT_RANGES_DEG[n][1] for n in names])
    q = np.clip(np.array([q0_deg.get(n, 0.0) for n in names], float), lo, hi)
    Rt = Rot.from_matrix(np.asarray(target_R, float))
    tp = np.asarray(target_pos, float)

    def resid(qv):
        qd = {n: v for n, v in zip(names, qv)}
        T = fk.site_pose(qd)
        pe = tp - T[:3, 3]
        re = (Rt * Rot.from_matrix(T[:3, :3]).inv()).as_rotvec()
        return np.concatenate([pe, w_rot * re])

    r = resid(q)
    lam = 1e-3
    for _ in range(iters):
        if np.linalg.norm(r[:3]) < 1e-6 and np.linalg.norm(r[3:]) < 1e-4:
            break
        J = np.zeros((6, 5))
        for i in range(5):
            dq = np.zeros(5)
            dq[i] = 1e-4
            J[:, i] = (resid(q + dq) - r) / 1e-4
        A = J.T @ J + lam * np.eye(5)
        step = -np.linalg.solve(A, J.T @ r)   # resid is (target - current): GN step is negative
        step = np.clip(step, -5.0, 5.0)
        q_new = np.clip(q + step, lo, hi)
        r_new = resid(q_new)
        if np.linalg.norm(r_new) < np.linalg.norm(r):
            q, r = q_new, r_new
            lam = max(lam * 0.5, 1e-6)
        else:
            lam = min(lam * 4, 1e-1)
    return ({n: float(v) for n, v in zip(names, q)},
            float(np.linalg.norm(r[:3])), float(np.linalg.norm(r[3:]) / max(w_rot, 1e-9)))


def ik_site(fk, target_pos, target_zaxis, q0_deg, roll_deg=0.0, iters=200, tol=1e-6):
    """Numerical IK (damped least squares, finite-diff Jacobian) for the 5 arm
    joints: residuals = site position error (3) + site z-axis direction error
    (2 independent components via (z x target) ) + wrist_roll pull to roll_deg.
    Returns (q_deg dict, final residual norm)."""
    names = JOINTS[:5]
    q = np.array([q0_deg.get(n, 0.0) for n in names], float)
    tz = np.asarray(target_zaxis, float)
    tz = tz / np.linalg.norm(tz)

    def resid(qv):
        qd = {n: v for n, v in zip(names, qv)}
        T = fk.site_pose(qd)
        p = T[:3, 3]
        z = T[:3, 2]
        cross = np.cross(z, tz)
        return np.concatenate([np.asarray(target_pos, float) - p, cross[:2], [0.02 * (qv[4] - roll_deg)]])

    r = resid(q)
    lam = 1e-3
    for _ in range(iters):
        if np.linalg.norm(r[:3]) < tol and np.linalg.norm(r[3:5]) < 1e-4:
            break
        J = np.zeros((len(r), 5))
        for i in range(5):
            dq = np.zeros(5)
            dq[i] = 1e-4
            J[:, i] = (resid(q + dq) - r) / 1e-4
        A = J.T @ J + lam * np.eye(5)
        step = -np.linalg.solve(A, J.T @ r)
        step = np.clip(step, -5.0, 5.0)
        q_new = q + step
        r_new = resid(q_new)
        if np.linalg.norm(r_new) < np.linalg.norm(r):
            q, r = q_new, r_new
            lam = max(lam * 0.5, 1e-6)
        else:
            lam = min(lam * 4, 1e-1)
    return {n: float(v) for n, v in zip(names, q)}, float(np.linalg.norm(r))
