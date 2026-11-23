"""Twin-suite v2 (Stage 1.2): programmatic Drake port of the v1.1 twin scenes.

Reads the SAME engine-independent inputs as the MuJoCo builder:
  - per-episode poses via v1.1 build_scenes.load_episode_poses (FK-affine
    corrected Builder B poses; the scene JSONs are the raw record),
  - fixtures via build_scenes.load_fixtures,
  - variant draws via build_scenes.make_draws (frozen seed 20261122),
  - all geometry/mass/friction values from v1.1 twin_params.py.
No per-episode hand-built files: one builder for all episodes/variants.

Global, logged engine-mapping choices (NOT tuned per episode):
  - Contact model: Drake point contact (kPointContactOnly). MuJoCo uses its
    soft constraint model at 2000 Hz; Drake plant step here is 1000 Hz.
  - Friction: twin_params tuples are MuJoCo (sliding, torsional, rolling).
    Drake CoulombFriction gets (static = sliding, dynamic = sliding):
    the MuJoCo sliding coefficient is the only sourced value; no second
    coefficient is invented.
  - Arm collision geoms are REMOVED (proximity role stripped): v1.1 replay
    sets arm contype/conaffinity = 0 (grasp is via the weld device only).
  - Ring: 12 capsule collision geoms on one body, exactly as MuJoCo
    (this Drake build has no Torus shape). Tool: one capsule, Drake axis
    is local +z, registered rotated so it matches MuJoCo's local +x axis.
  - Ring/tool inertia: UnitInertia.SolidCylinder approximations in the
    body frame (mass exact; inertia is a logged approximation).
"""
import math, sys
from pathlib import Path
import numpy as np

V1 = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
sys.path.insert(0, str(V1 / "code"))
import twin_params as P
import build_scenes as B

URDF = "/home/openclaw/muse-work/zk0-sim-eval2-20261115/assets/SO101/so101_drake.urdf"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

from pydrake.geometry import (Box, Capsule, Cylinder, HalfSpace, ProximityProperties,
                              Rgba, RoleAssign, AddContactMaterial,
                              AddCompliantHydroelasticProperties, AddRigidHydroelasticProperties)
import os
# Global contact-model choice (logged, never per-episode):
#  "point" (default): Drake point contact.
#  "hydro": hydroelastic-with-fallback, the Nov-15 Drake eval configuration
#  (compliant modulus 1e7, dissipation 10, resolution hint 5 mm on dynamic
#  objects; rigid hydroelastic on statics) -- patch contact resists the
#  capsule rolling that point contact allows.
CONTACT_MODEL = os.environ.get("DK_CONTACT", "point")
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ContactModel, CoulombFriction
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.systems.framework import DiagramBuilder


def mat2quat_wxyz(R):
    m = np.asarray(R, float); tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    return RotationMatrix(np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]))


def rot_z_to(d):
    """RotationMatrix whose local +z axis maps to unit direction d."""
    d = np.asarray(d, float); d /= (np.linalg.norm(d) + 1e-12)
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, d); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(d, x)
    return RotationMatrix(np.column_stack([x, y, d]))


def _props(mu, compliant=False):
    pr = ProximityProperties()
    if CONTACT_MODEL == "hydro":
        if compliant:
            AddCompliantHydroelasticProperties(0.005, 1e7, pr)
            AddContactMaterial(pr, dissipation=10.0, friction=CoulombFriction(float(mu), float(mu)))
        else:
            try:
                AddRigidHydroelasticProperties(0.01, pr)
            except TypeError:
                AddRigidHydroelasticProperties(pr)
            AddContactMaterial(pr, friction=CoulombFriction(float(mu), float(mu)))
    else:
        AddContactMaterial(pr, friction=CoulombFriction(float(mu), float(mu)))
    return pr


def _rgba(s):
    v = [float(x) for x in str(s).split()]
    return np.array([v[0], v[1], v[2], v[3] if len(v) > 3 else 1.0])


class DrakeScene:
    """Built Drake twin scene + handles the replay harness needs."""

    def __init__(self, task, epi, variant=0):
        self.task, self.epi, self.variant = task, epi, variant
        poses = B.load_episode_poses(task, epi)
        if poses is None:
            raise FileNotFoundError(f"no Builder B poses for {task} ep{epi}")
        self.poses = poses
        self.fixtures = B.load_fixtures(task, poses)
        draws = B.make_draws(task, epi, poses, variant)
        self.draws = draws
        var = draws
        objs = poses.get("objects", {})

        def jpose(body, pose):
            j = var["jitter"].get(body)
            if not j or pose is None:
                return pose
            return {"pos": [pose["pos"][0] + j[0], pose["pos"][1] + j[1], pose["pos"][2]],
                    "quat": list(B.qmul(B.yaw_quat(j[2]), np.array(pose["quat"])))}

        def vmass(body, nominal):
            return var["mass"].get(body, nominal)

        def vfric(body, tup):
            return float(tup[0]) * var["friction_mult"].get(body, 1.0)

        def obj_pose(*names, default=None):
            for n in names:
                if n in objs:
                    return objs[n]
            return default

        builder = DiagramBuilder()
        plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        self.plant, self.sg = plant, sg
        self.geom_body = {}   # GeometryId -> body name (for hydroelastic contacts)
        if CONTACT_MODEL == "hydro":
            plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
        else:
            try:
                plant.set_contact_model(ContactModel.kPointContactOnly)
            except Exception:
                plant.set_contact_model(ContactModel.kPoint)
        parser = Parser(plant)
        parser.AddModels(URDF)
        self.robot = plant.GetModelInstanceByName("so101_new_calib")
        base_body = plant.GetBodyByName("base_link", self.robot)
        plant.WeldFrames(plant.world_frame(), base_body.body_frame())
        # strip arm proximity role (v1.1: arm geoms have contype=0).
        # NOTE: AssignRole(kReplace, empty props) does NOT remove the role
        # in this Drake build (verified: 17/17 arm geoms stayed proximity);
        # RemoveRole is the working API.
        from pydrake.geometry import Role as _Role
        self.arm_proximity_removed = 0
        for bidx in plant.GetBodyIndices(self.robot):
            for gid in plant.GetCollisionGeometriesForBody(plant.get_body(bidx)):
                sg.RemoveRole(plant.get_source_id(), gid, _Role.kProximity)
                self.arm_proximity_removed += 1

        world = plant.world_body()
        mx, my = P.MAT_CENTER; sx, sy, st = P.MAT_SIZE
        # MuJoCo floor is an infinite plane at z=-0.001; Drake HalfSpace
        # reports no contact data in this build, so the floor is a large
        # thin box with its top at z=-0.001 (logged approximation).
        self.geom_body[plant.RegisterCollisionGeometry(world, RigidTransform(RotationMatrix(), np.array([0, 0, -0.026])),
                                        Box(4.0, 4.0, 0.05), "floor", _props(P.FLOOR_FRICTION[0]))] = "world"
        self.geom_body[plant.RegisterCollisionGeometry(world, RigidTransform(RotationMatrix(), np.array([mx, my, -st / 2])),
                                        Box(sx, sy, st), "mat", _props(P.MAT_FRICTION[0]))] = "world"
        for nm, px, py, bx, by in [("wall_back", mx, my + sy / 2 + 0.005, sx, 0.01),
                                   ("wall_front", mx, my - sy / 2 - 0.005, sx, 0.01),
                                   ("wall_left", mx - sx / 2 - 0.005, my, 0.01, sy),
                                   ("wall_right", mx + sx / 2 + 0.005, my, 0.01, sy)]:
            self.geom_body[plant.RegisterCollisionGeometry(world, RigidTransform(RotationMatrix(), np.array([px, py, 0.03])),
                                            Box(bx, by, 0.06), nm, _props(P.WALL_FRICTION[0]))] = "world"

        self.bodies = {}      # name -> RigidBody (dynamic objects + fixtures)
        self.body_mi = {}     # name -> ModelInstanceIndex
        self.weldables = []

        def static_body(name, pos):
            mi = plant.AddModelInstance(name + "_mi")
            body = plant.AddRigidBody(name, mi, SpatialInertia.SolidBoxWithDensity(1000.0, 0.1, 0.1, 0.1))
            self.body_mi[name] = mi
            plant.WeldFrames(plant.world_frame(), body.body_frame(),
                             RigidTransform(RotationMatrix(), np.array([pos[0], pos[1], 0.0])))
            self.bodies[name] = body
            return body

        def free_body(name, pose, inertia):
            mi = plant.AddModelInstance(name + "_mi")
            body = plant.AddRigidBody(name, mi, inertia)
            self.bodies[name] = body
            self.body_mi[name] = mi
            self._init_pose = getattr(self, "_init_pose", {})
            self._init_pose[name] = RigidTransform(quat_wxyz_to_R(pose["quat"]), np.array(pose["pos"], float))
            self.weldables.append(name)
            return body

        def box_geom(body, name, size, pos, mu, rgba, compliant=False):
            X = RigidTransform(RotationMatrix(), np.array(pos, float))
            gid = plant.RegisterCollisionGeometry(body, X, Box(*size), name, _props(mu, compliant=compliant))
            self.geom_body[gid] = body.name()
            plant.RegisterVisualGeometry(body, X, Box(*size), name + "_vis", _rgba(rgba))

        if task == "block_stack":
            e = P.CUBE_EDGE
            cols = {"cube_a": B.color_rgba(poses, "cube_a", "0.55 0.25 0.75 1"),
                    "cube_b": B.color_rgba(poses, "cube_b", "0.25 0.65 0.30 1")}
            for i, nm in enumerate(["cube_a", "cube_b"]):
                dflt = {"pos": [0.26 + 0.07 * i, -0.05 + 0.09 * i, e / 2], "quat": [1, 0, 0, 0]}
                pose = jpose(nm, obj_pose(nm, default=dflt))
                body = free_body(nm, pose, SpatialInertia(vmass(nm, P.CUBE_MASS), np.zeros(3),
                                                          UnitInertia.SolidBox(e, e, e)))
                box_geom(body, nm + "_geom", (e, e, e), (0, 0, 0), vfric(nm, P.CUBE_FRICTION), cols[nm], compliant=True)
        elif task == "eye_drops_to_basket":
            pf = self.fixtures["platform"]["pos"]; bk = self.fixtures["basket"]["pos"]
            tw, td, tt = P.PLATFORM_TOP; h = P.PLATFORM_HEIGHT
            plat = static_body("platform", pf)
            box_geom(plat, "platform_top", (tw, td, tt), (0, 0, h - tt / 2), P.PLATFORM_FRICTION[0], "0.92 0.92 0.92 1")
            box_geom(plat, "platform_web", (0.02, td / 2, h), (0, 0, h / 2), P.PLATFORM_FRICTION[0], "0.92 0.92 0.92 1")
            box_geom(plat, "platform_base", (tw, td, 0.004), (0, 0, 0.002), P.PLATFORM_FRICTION[0], "0.92 0.92 0.92 1")
            bask = static_body("basket", bk)
            ow, od = P.BASKET_OUTER; wt, wh = P.BASKET_WALL_T, P.BASKET_WALL_H
            box_geom(bask, "basket_floor", (ow, od, wt), (0, 0, wt / 2), P.BASKET_FRICTION[0], "0.85 0.90 0.95 1")
            box_geom(bask, "basket_w1", (ow, wt, wh), (0, od / 2 - wt / 2, wh / 2), P.BASKET_FRICTION[0], "0.85 0.90 0.95 1")
            box_geom(bask, "basket_w2", (ow, wt, wh), (0, -od / 2 + wt / 2, wh / 2), P.BASKET_FRICTION[0], "0.85 0.90 0.95 1")
            box_geom(bask, "basket_w3", (wt, od, wh), (ow / 2 - wt / 2, 0, wh / 2), P.BASKET_FRICTION[0], "0.85 0.90 0.95 1")
            box_geom(bask, "basket_w4", (wt, od, wh), (-ow / 2 + wt / 2, 0, wh / 2), P.BASKET_FRICTION[0], "0.85 0.90 0.95 1")
            csx, csy, csz = P.CARTON_SIZE
            dflt = {"pos": [pf[0], pf[1], h + csy / 2], "quat": list(B.yaw_quat(0))}
            pose = jpose("carton", obj_pose("carton", default=dflt))
            body = free_body("carton", pose, SpatialInertia(vmass("carton", P.CARTON_MASS), np.zeros(3),
                                                            UnitInertia.SolidBox(csx, csy, csz)))
            box_geom(body, "carton_geom", (csx, csy, csz), (0, 0, 0), vfric("carton", P.CARTON_FRICTION), "0.25 0.45 0.85 1", compliant=True)
        elif task == "ring_insert":
            bp = list(self.fixtures["board"]["pos"]); bt = P.BOARD_H
            pegp = list(self.fixtures.get("peg", {}).get("pos", bp))
            board = static_body("board", bp)
            Xb = RigidTransform(RotationMatrix(), np.array([0, 0, bt / 2]))
            self.geom_body[plant.RegisterCollisionGeometry(board, Xb, Cylinder(P.BOARD_R, bt), "board_geom", _props(P.BOARD_FRICTION[0]))] = "board"
            plant.RegisterVisualGeometry(board, Xb, Cylinder(P.BOARD_R, bt), "board_vis", _rgba("0.92 0.87 0.72 1"))
            peg = static_body("peg", pegp)
            Xp = RigidTransform(RotationMatrix(), np.array([0, 0, bt + P.PEG_H / 2]))
            self.geom_body[plant.RegisterCollisionGeometry(peg, Xp, Cylinder(P.PEG_R, P.PEG_H), "peg_geom", _props(P.PEG_FRICTION[0]))] = "peg"
            plant.RegisterVisualGeometry(peg, Xp, Cylinder(P.PEG_R, P.PEG_H), "peg_vis", _rgba("0.80 0.65 0.42 1"))
            n = P.RING_SEGMENTS; rmid = (P.RING_R_OUT + P.RING_R_IN) / 2
            tube = min((P.RING_R_OUT - P.RING_R_IN) / 2, P.RING_THICK / 2)
            dflt = {"pos": [bp[0] - 0.045, bp[1] - 0.045, bt + P.RING_THICK / 2], "quat": [1, 0, 0, 0]}
            pose = jpose("ring", obj_pose("ring", default=dflt))
            body = free_body("ring", pose, SpatialInertia(vmass("ring", P.RING_MASS), np.zeros(3),
                                                          UnitInertia.SolidCylinder(rmid, P.RING_THICK, np.array([0.0, 0, 1.0]))))
            mu = vfric("ring", P.RING_FRICTION)
            col = B.color_rgba(poses, "ring", "0.95 0.80 0.15 1")
            for k in range(n):
                a0 = 2 * math.pi * k / n; a1 = 2 * math.pi * (k + 1) / n
                p0 = np.array([rmid * math.cos(a0), rmid * math.sin(a0), 0.0])
                p1 = np.array([rmid * math.cos(a1), rmid * math.sin(a1), 0.0])
                mid = (p0 + p1) / 2; seglen = float(np.linalg.norm(p1 - p0))
                X = RigidTransform(rot_z_to(p1 - p0), mid)
                self.geom_body[plant.RegisterCollisionGeometry(body, X, Capsule(tube, seglen), f"ring_seg_{k}", _props(mu, compliant=True))] = "ring"
                plant.RegisterVisualGeometry(body, X, Capsule(tube, seglen), f"ring_seg_{k}_vis", _rgba(col))
        elif task == "tool_insert":
            tp = self.fixtures["tray"]["pos"]; sp = self.fixtures["slot"]["pos"]
            tw, td, th = P.TRAY_SIZE; sw, sd = P.SLOT_SIZE; depth = P.TOOL_R
            lx, ly = sp[0] - tp[0], sp[1] - tp[1]
            rx0, rx1 = max(lx - sw / 2, -tw / 2), min(lx + sw / 2, tw / 2)
            ry0, ry1 = max(ly - sd / 2, -td / 2), min(ly + sd / 2, td / 2)
            tray = static_body("tray", tp)
            box_geom(tray, "tray_geom", (tw, td, th - depth), (0, 0, (th - depth) / 2), P.TRAY_FRICTION[0], "0.10 0.10 0.10 1")
            box_geom(tray, "slot_floor", (rx1 - rx0, ry1 - ry0, 0.001),
                     ((rx0 + rx1) / 2, (ry0 + ry1) / 2, th - depth + 0.0005), P.TRAY_FRICTION[0], "0.45 0.08 0.08 1")
            def _rim(name, cx, cy, hx, hy):
                if hx > 1e-9 and hy > 1e-9:
                    box_geom(tray, name, (2 * hx, 2 * hy, depth), (cx, cy, th - depth / 2), P.TRAY_FRICTION[0], "0.10 0.10 0.10 1")
            _rim("tray_rim_w", (-tw / 2 + rx0) / 2, 0.0, (rx0 + tw / 2) / 2, td / 2)
            _rim("tray_rim_e", (rx1 + tw / 2) / 2, 0.0, (tw / 2 - rx1) / 2, td / 2)
            _rim("tray_rim_s", (rx0 + rx1) / 2, (-td / 2 + ry0) / 2, (rx1 - rx0) / 2, (ry0 + td / 2) / 2)
            _rim("tray_rim_n", (rx0 + rx1) / 2, (ry1 + td / 2) / 2, (rx1 - rx0) / 2, (td / 2 - ry1) / 2)
            dflt = {"pos": [0.28, -0.06, P.TOOL_R], "quat": list(B.yaw_quat(20))}
            pose = jpose("tool", obj_pose("tool", default=dflt))
            body = free_body("tool", pose, SpatialInertia(vmass("tool", P.TOOL_MASS), np.zeros(3),
                                                          UnitInertia.SolidCylinder(P.TOOL_R, P.TOOL_LEN, np.array([1.0, 0, 0]))))
            # Drake capsule axis = local +z; MuJoCo tool axis = local +x.
            X = RigidTransform(RotationMatrix.MakeYRotation(math.pi / 2), np.zeros(3))
            self.geom_body[plant.RegisterCollisionGeometry(body, X, Capsule(P.TOOL_R, P.TOOL_LEN), "tool_geom",
                                            _props(vfric("tool", P.TOOL_FRICTION), compliant=True))] = "tool"
            plant.RegisterVisualGeometry(body, X, Capsule(P.TOOL_R, P.TOOL_LEN), "tool_vis", _rgba("0.75 0.75 0.78 1"))
            self.tool_axis_local = np.array([1.0, 0, 0])  # body frame = MuJoCo frame; only the geom is rotated
        else:
            raise ValueError(task)

        plant.Finalize()
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.context)
        if plant.num_actuators():
            plant.get_actuation_input_port().FixValue(self.plant_context,
                                                      np.zeros(plant.num_actuators()))
        # joint name -> instance-vector index
        starts = []
        for ji in plant.GetJointIndices(self.robot):
            j = plant.get_joint(ji)
            if j.num_positions() == 1:
                starts.append((int(j.position_start()), j.name()))
        starts.sort()
        self._qbase = starts[0][0] if starts else 0
        self.joint_idx = {name: s - self._qbase for s, name in starts}
        self.grip_body = plant.GetBodyByName("gripper_link", self.robot)
        self.grip_frame_body = plant.GetBodyByName("gripper_frame_link", self.robot)
        # initial object poses
        for nm, X in getattr(self, "_init_pose", {}).items():
            plant.SetFreeBodyPose(self.plant_context, self.bodies[nm], X)

    # ---- kinematic arm + pose accessors ---------------------------------
    def set_arm(self, q_named):
        q = self.plant.GetPositions(self.plant_context, self.robot)
        for nm, v in q_named.items():
            if nm in self.joint_idx:
                q[self.joint_idx[nm]] = v
        self.plant.SetPositions(self.plant_context, self.robot, q)
        self.plant.SetVelocities(self.plant_context, self.robot,
                                 np.zeros(self.plant.num_velocities(self.robot)))

    def body_pose(self, name):
        body = self.bodies[name] if name in self.bodies else self.plant.GetBodyByName(name, self.robot)
        return self.plant.EvalBodyPoseInWorld(self.plant_context, body)

    def gripper_pose(self):
        return self.plant.EvalBodyPoseInWorld(self.plant_context, self.grip_body)

    def site_pos(self):
        return self.plant.EvalBodyPoseInWorld(self.plant_context, self.grip_frame_body).translation()

    def set_free_pose(self, name, X):
        self.plant.SetFreeBodyPose(self.plant_context, self.bodies[name], X)
        mi = self.body_mi[name]
        self.plant.SetVelocities(self.plant_context, mi,
                                 np.zeros(self.plant.num_velocities(mi)))

    def contact_pairs(self):
        """Set of frozenset({body_name_a, body_name_b}) from point contacts."""
        cr = self.plant.get_contact_results_output_port().Eval(self.plant_context)
        pairs = set()
        try:
            for i in range(cr.num_hydroelastic_contacts()):
                info = cr.hydroelastic_contact_info(i)
                surf = info.contact_surface()
                na = self.geom_body.get(surf.id_M()); nb = self.geom_body.get(surf.id_N())
                if na and nb:
                    pairs.add(frozenset((na, nb)))
        except Exception:
            pass
        n = cr.num_point_pair_contacts()
        for i in range(n):
            info = cr.point_pair_contact_info(i)
            ba = info.bodyA_index if not callable(getattr(info, "bodyA_index", None)) else info.bodyA_index()
            bb = info.bodyB_index if not callable(getattr(info, "bodyB_index", None)) else info.bodyB_index()
            na = self.plant.get_body(ba).name(); nb = self.plant.get_body(bb).name()
            pairs.add(frozenset((na, nb)))
        return pairs

    def contacts_with(self, name):
        out = set()
        for pr in self.contact_pairs():
            if name in pr:
                out.update(x for x in pr if x != name)
        return out
