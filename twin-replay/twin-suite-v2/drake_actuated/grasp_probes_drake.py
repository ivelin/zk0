"""Drake actuated grasp probe battery (twin-suite v2 Stage 1.2 part 2a).

Mirror of harness/fingerpad_probes.py (MuJoCo) semantics, in Drake:
scripted IK-centred grasps of the 30 mm cube (block_stack ep411 nominal,
cube_a) at descent heights +0.005 / 0.000 / -0.005 m, plus the site-centred
probe (xoff 0) expected to fail the same way as in MuJoCo. Executor
GENERATION: ORACLE-GUIDED, identical protocol: 6-DOF IK (gripperframe
site position + gripper body aligned to world, fingers down), site x
offset centres the cube between the pad faces, pre-close 0.7 rad on
descent, close to 0.05 rad target, lift in 1 cm Cartesian stages with
grip desired frozen at stall - 0.02 rad, then 2 s hold. PASS = cube held
>= 2 s by finger contact alone (moving jaw in the contact set, no world
contact, z > 0.05). No pin, no weld -- the actuated scene has neither.

Usage: grasp_probes_drake.py [out.json]   (env DK_CONTACT=hydro|point)
Also:  grasp_probes_drake.py --padpos     (pad world positions at the
recorded frame-130 pose, for the cross-engine geometry parity check)
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("DK_CONTACT", "hydro")
import actuated_port as AP
from actuated_port import DrakeScene, JOINTS, pad_world
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.analysis import Simulator
from pydrake.multibody.tree import JacobianWrtVariable

V1 = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
sys.path.insert(0, str(V1 / "code"))
import twin_params as P

TCP_OFF = np.array([-0.0079, -0.000218121, -0.0981274])  # gripperframe site, gripper body frame
GRIP_DEG_MAX = 52.3
GRIP_RAD_CLOSED, GRIP_RAD_OPEN = 0.10, 1.45
PRE, CLOSE = 0.7, 0.05
ARM = JOINTS[:5]

def grip_deg_to_rad(d):
    return GRIP_RAD_CLOSED + (float(d) / GRIP_DEG_MAX) * (GRIP_RAD_OPEN - GRIP_RAD_CLOSED)

def action_deg_to_ctrl(a_deg):
    a = np.asarray(a_deg, float)
    return np.array(np.radians(a[:5]).tolist() + [grip_deg_to_rad(a[5])])

scene = DrakeScene("block_stack", 411, 0)
plant, robot = scene.plant, scene.robot
sim = Simulator(scene.diagram, scene.context)
sim.set_target_realtime_rate(0.0)
sim.Initialize()
NOW = [0.0]

def advance(dt):
    NOW[0] += dt
    sim.AdvanceTo(NOW[0])

def site_pos():
    return scene.site_pos()

def gripper_R():
    return scene.gripper_pose().rotation().matrix()

def ik6(target, seed_deg):
    """DLS IK: site position -> target, gripper body rotation -> identity.
    Mirrors fingerpad_probes.ik6 (500 iters, damped 0.01, clip 0.03)."""
    ctx = plant.CreateDefaultContext()
    c = action_deg_to_ctrl(seed_deg)
    q = {j: float(c[k]) for k, j in enumerate(JOINTS)}
    frame = plant.GetFrameByName("gripper_link", robot)
    joints = [plant.GetJointByName(n, robot) for n in ARM]
    cols = [j.velocity_start() for j in joints]
    lo = np.array([j.position_lower_limit() for j in joints]) + 1e-4
    hi = np.array([j.position_upper_limit() for j in joints]) - 1e-4
    for _ in range(500):
        plant.SetPositions(ctx, robot, np.array([q[j] for j in ARM] + [q["gripper"]])
                           if False else _qvec(ctx, q))
        X = plant.EvalBodyPoseInWorld(ctx, scene.grip_body)
        p_cur = X.translation() + X.rotation().matrix() @ TCP_OFF
        perr = np.array(target) - p_cur
        R = X.rotation().matrix(); Rt = R.T
        ang = np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1))
        if np.linalg.norm(perr) < 0.002 and ang < 0.02:
            break
        if ang < 1e-6:
            oerr = np.zeros(3)
        else:
            axis = np.array([Rt[2, 1] - Rt[1, 2], Rt[0, 2] - Rt[2, 0], Rt[1, 0] - Rt[0, 1]]) / (2 * np.sin(ang))
            oerr = R @ (axis * ang)
        err = np.concatenate([perr, oerr * 0.5])
        J = plant.CalcJacobianSpatialVelocity(ctx, JacobianWrtVariable.kV, frame, TCP_OFF,
                                              plant.world_frame(), plant.world_frame())
        J6 = np.vstack([J[3:6][:, cols], J[0:3][:, cols]])  # rows: linear, angular (MuJoCo jacp/jacr order)
        err6 = np.concatenate([perr, oerr * 0.5])
        dq = J6.T @ np.linalg.solve(J6 @ J6.T + 0.01 * np.eye(6), err6)
        cur = np.array([q[j] for j in ARM])
        cur = np.clip(cur + np.clip(dq, -0.03, 0.03), lo, hi)
        for k, j in enumerate(ARM):
            q[j] = float(cur[k])
    return np.array([q[j] for j in JOINTS])

def _qvec(ctx, q):
    full = plant.GetPositions(ctx, robot)
    for nm, v in q.items():
        full[scene.joint_idx[nm]] = v
    return full

def contacts_cube():
    out = set()
    for nm in scene.contacts_with("cube_a"):
        out.add({"gripper_link": "gripper", "moving_jaw_so101_v1_link": "moving_jaw_so101_v1"}.get(nm, nm))
    # hydro path returns geom_body names already in MuJoCo naming
    return sorted(out)

def cube_pos():
    return scene.body_pose("cube_a").translation().copy()

TIMESCALE = float(os.environ.get("PROBE_TIMESCALE", "1"))  # diagnostic only; battery = 1

def run_to(q_target, seconds):
    seconds = seconds * TIMESCALE
    start = scene._qdes.copy()
    steps = max(1, int(seconds / 0.02))
    for s in range(steps):
        f = (s + 1) / steps
        cc = start * (1 - f) + np.array(q_target) * f
        scene.set_desired({j: float(cc[k]) for k, j in enumerate(JOINTS)})
        advance(0.02)

gt = np.load(P.RECORDS / "base__block_stack__ep411.npz")["gt"].astype(float)

def seed_now():
    aq = scene.arm_q()
    return [np.degrees(aq[j]) for j in JOINTS[:5]] + [10.0]

_last_seed = None
def staged_move(target, grip, dz_stage=0.01, secs=0.6):
    global _last_seed
    site0 = site_pos(); target = np.array(target, float)
    delta = target - site0; n = max(1, int(np.ceil(np.linalg.norm(delta) / dz_stage)))
    for i in range(n):
        t = site0 + delta * ((i + 1) / n)
        seed = _last_seed if _last_seed is not None else seed_now()
        q = ik6(t, seed); q[5] = grip
        _last_seed = [np.degrees(q[j]) for j in range(5)] + [10.0]
        run_to(q, secs)

def reset_arm():
    c0 = action_deg_to_ctrl(gt[0])
    q0 = {j: float(c0[k]) for k, j in enumerate(JOINTS)}
    scene.set_arm_and_desired(q0)
    # objects: restore initial poses (fresh scene per probe is cleaner --
    # caller rebuilds; here just re-set cube poses to scene init)
    for nm, X in scene._init_pose.items():
        scene.set_free_pose(nm, X)
    advance(0.25)

def grasp_and_lift(H, xoff):
    global _last_seed; _last_seed = None
    reset_arm()
    cube0 = cube_pos()
    OFF = np.array([xoff, 0.0, 0.0])
    wp = []
    for tgt, grip in ((cube0 + OFF + [0, 0, 0.13], PRE), (cube0 + OFF + [0, 0, H], PRE), (cube0 + OFF + [0, 0, H], CLOSE)):
        q = ik6(tgt, gt[130]); q[5] = grip; wp.append(q)
    c0 = action_deg_to_ctrl(gt[0])
    scene.set_desired({j: float(c0[k]) for k, j in enumerate(JOINTS)})
    if os.environ.get("PROBE_SKIP_APPROACH") == "1":
        pass  # DIAGNOSTIC variant: direct descent from the reset pose
    else:
        run_to(wp[0], 1.0)
    run_to(wp[1], 1.0)
    pre = cube_pos()
    run_to(wp[2], 1.2)
    shove = float(np.linalg.norm(cube_pos()[:2] - pre[:2]) * 1000)
    cs_close = contacts_cube(); grip_q = float(scene.arm_q()["gripper"])
    # pinch-centre measurement at stall: midpoint of the two pad centres
    pf, pm = pad_world(scene, "finger_pad_fixed"), pad_world(scene, "finger_pad_moving")
    pinch_mid = (pf + pm) / 2
    site = site_pos()
    pinch = {"pad_fixed_world": pf.tolist(), "pad_moving_world": pm.tolist(),
             "site_world": site.tolist(), "pinch_midpoint_world": pinch_mid.tolist(),
             "site_to_pinch_midpoint_mm": float(np.linalg.norm(site - pinch_mid) * 1000),
             "pad_centre_distance_mm": float(np.linalg.norm(pf - pm) * 1000)}
    s0 = site_pos()
    staged_move(s0 + [0, 0, 0.07], grip_q - 0.02)
    cs_lift = contacts_cube(); z_lift = float(cube_pos()[2])
    run_to(scene._qdes.copy(), 2.0)
    z_hold = float(cube_pos()[2]); cs_hold = contacts_cube()
    ok = ("moving_jaw_so101_v1" in cs_hold and "world" not in cs_hold and z_hold > 0.05)
    print(f"HEIGHT {H:+.3f} XOFF {xoff:+.3f}: shove_mm {shove:.1f} grip_qpos {grip_q:.3f} close_contacts {cs_close} "
          f"lift_z {z_lift:.3f} hold2s_z {z_hold:.3f} hold_contacts {cs_hold} PINCH_LIFT_HOLD {'PASS' if ok else 'FAIL'}", flush=True)
    return {"height": H, "xoff": xoff, "shove_mm": shove, "grip_qpos": grip_q,
            "close_contacts": cs_close, "lift_z": z_lift, "hold_z": z_hold,
            "hold_contacts": cs_hold, "pass": bool(ok), "pinch": pinch}

if "--padpos" in sys.argv:
    q130 = action_deg_to_ctrl(gt[130])
    scene.set_arm({j: float(q130[k]) for k, j in enumerate(JOINTS)})
    print(json.dumps({
        "finger_geoms": scene.finger_geoms, "mesh_failed": scene.finger_mesh_failed,
        "pad_fixed_world": pad_world(scene, "finger_pad_fixed").tolist(),
        "pad_moving_world": pad_world(scene, "finger_pad_moving").tolist(),
        "site_world": scene.site_pos().tolist(),
        "actuator_info": scene.actuator_info}, indent=1))
    sys.exit(0)

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "probe_results.json"
    print("finger geoms:", scene.finger_geoms, "mesh_failed:", scene.finger_mesh_failed, flush=True)
    print("contact model:", os.environ.get("DK_CONTACT"), flush=True)
    res = {"contact_model": os.environ.get("DK_CONTACT"), "timescale": TIMESCALE, "skip_approach": os.environ.get("PROBE_SKIP_APPROACH") == "1", "finger_geoms": scene.finger_geoms,
           "mesh_failed": scene.finger_mesh_failed, "probes": []}
    for H in (0.005, 0.0, -0.005):
        # fresh scene per height: object/arm state fully reset
        res["probes"].append(grasp_and_lift(H, -0.020))
    res["probes"].append(grasp_and_lift(0.0, 0.0))  # site-centred, expected FAIL
    Path(out).write_text(json.dumps(res, indent=1))
    print("PROBES_DONE", flush=True)

if __name__ == "__main__":
    main()
