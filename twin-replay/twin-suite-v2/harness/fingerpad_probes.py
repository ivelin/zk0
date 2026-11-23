"""Finger-pad probe battery (v2, padded model): scripted IK-centred grasps
at three descent heights. Executor (GENERATION: ORACLE-GUIDED): 6-DOF IK
(site position + gripper body aligned to world, fingers down, pinch axis
world x), site offset XOFF centres the cube between the pad faces (fixed
face at site_x-0.002, moving face measured), pre-close 0.7 rad on descent,
close to 0.05 rad (servo stalls at ~0.12 on the 30 mm cube -> pinch force),
lift in 1 cm Cartesian stages (IK save/restore, seeded from current state)
with grip ctrl frozen near stall, then 2 s hold.
Usage: fingerpad_probes.py [xoff]"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness")
import numpy as np, mujoco
import sim_runner_padded as SR
XOFF = float(sys.argv[1]) if len(sys.argv) > 1 else -0.020
PRE, CLOSE = 0.7, 0.05
model = SR.build_actuated_model(SR.SCENES / "block_stack__ep411_v0.xml")
info = SR.prepare_model(model); model.opt.timestep = SR.P.TIMESTEP
data = mujoco.MjData(model)
qadr = {j: int(model.joint(j).qposadr[0]) for j in SR.JOINTS}
vadr = {j: int(model.joint(j).dofadr[0]) for j in SR.JOINTS}
act = {j: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)) for j in SR.JOINTS}
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
gbody = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")
gt = np.load(SR.RECORDS / "base__block_stack__ep411.npz")["gt"].astype(float)
ARM = SR.JOINTS[:5]
def ik6(target, seed_deg):
    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    c = SR.action_deg_to_ctrl(seed_deg)
    for k, j in enumerate(SR.JOINTS): data.qpos[qadr[j]] = c[k]
    mujoco.mj_forward(model, data)
    cols = [vadr[j] for j in ARM]
    for _ in range(500):
        perr = np.array(target) - data.site_xpos[sid]
        R = data.xmat[gbody].reshape(3, 3); Rt = R.T
        ang = np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1))
        if np.linalg.norm(perr) < 0.002 and ang < 0.02: break
        if ang < 1e-6: oerr = np.zeros(3)
        else:
            axis = np.array([Rt[2,1]-Rt[1,2], Rt[0,2]-Rt[2,0], Rt[1,0]-Rt[0,1]]) / (2*np.sin(ang))
            oerr = R @ (axis * ang)
        err = np.concatenate([perr, oerr * 0.5])
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, cols], jacr[:, cols]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(6), err)
        for i, j in enumerate(ARM): data.qpos[qadr[j]] += np.clip(dq[i], -0.03, 0.03)
        mujoco.mj_forward(model, data)
    q = np.array([data.qpos[qadr[j]] for j in SR.JOINTS])
    data.qpos[:] = saved[0]; data.qvel[:] = saved[1]; data.ctrl[:] = saved[2]
    mujoco.mj_forward(model, data)
    return q
def contacts(body=None):
    body = cube if body is None else body
    out = set()
    for i in range(data.ncon):
        cc = data.contact[i]
        b1 = int(model.geom_bodyid[cc.geom1]); b2 = int(model.geom_bodyid[cc.geom2])
        if body in (b1, b2):
            out.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2 if b1 == body else b1))
    return sorted(str(x) for x in out)
def run_to(ctrl_target, seconds):
    start = np.array([data.ctrl[act[j]] for j in SR.JOINTS])
    steps = max(1, int(seconds / SR.P.TIMESTEP))
    for s in range(steps):
        f = (s + 1) / steps
        cc = start * (1 - f) + np.array(ctrl_target) * f
        for k, j in enumerate(SR.JOINTS): data.ctrl[act[j]] = cc[k]
        mujoco.mj_step(model, data)
def seed_now():
    return [np.degrees(data.qpos[qadr[j]]) for j in SR.JOINTS[:5]] + [10.0]
_last_seed = None
def staged_move(target, grip, dz_stage=0.01, secs=0.6):
    global _last_seed
    site0 = np.array(data.site_xpos[sid]); target = np.array(target, float)
    delta = target - site0; n = max(1, int(np.ceil(np.linalg.norm(delta) / dz_stage)))
    for i in range(n):
        t = site0 + delta * ((i + 1) / n)
        seed = _last_seed if _last_seed is not None else seed_now()
        q = ik6(t, seed); q[5] = grip
        _last_seed = [np.degrees(q[j]) for j in range(5)] + [10.0]
        run_to(q, secs)
def grasp_and_lift(H):
    mujoco.mj_resetData(model, data)
    c0 = SR.action_deg_to_ctrl(gt[0])
    for k, j in enumerate(SR.JOINTS):
        data.qpos[qadr[j]] = c0[k]; data.ctrl[act[j]] = c0[k]
    mujoco.mj_forward(model, data)
    for _ in range(int(0.25 / SR.P.TIMESTEP)): mujoco.mj_step(model, data)
    cube0 = np.array(data.xpos[cube])
    OFF = np.array([XOFF, 0.0, 0.0])
    wp = []
    for tgt, grip in ((cube0 + OFF + [0, 0, 0.13], PRE), (cube0 + OFF + [0, 0, H], PRE), (cube0 + OFF + [0, 0, H], CLOSE)):
        q = ik6(tgt, gt[130]); q[5] = grip; wp.append(q)
    for k, j in enumerate(SR.JOINTS):
        data.qpos[qadr[j]] = c0[k]; data.ctrl[act[j]] = c0[k]
    mujoco.mj_forward(model, data)
    run_to(wp[0], 1.0); run_to(wp[1], 1.0)
    pre = np.array(data.xpos[cube])
    run_to(wp[2], 1.2)
    shove = float(np.linalg.norm(np.array(data.xpos[cube])[:2] - pre[:2]) * 1000)
    cs_close = contacts(); grip_q = float(data.qpos[qadr['gripper']])
    site0 = np.array(data.site_xpos[sid])
    staged_move(site0 + [0, 0, 0.07], grip_q - 0.02)
    cs_lift = contacts(); z_lift = float(data.xpos[cube][2])
    run_to(np.array([data.ctrl[act[j]] for j in SR.JOINTS]), 2.0)
    z_hold = float(data.xpos[cube][2]); cs_hold = contacts()
    ok = ("moving_jaw_so101_v1" in cs_hold and "world" not in cs_hold and z_hold > 0.05)
    print(f"HEIGHT {H:+.3f}: shove_mm {shove:.1f} grip_qpos {grip_q:.3f} close_contacts {cs_close} "
          f"lift_z {z_lift:.3f} hold2s_z {z_hold:.3f} hold_contacts {cs_hold} PINCH_LIFT_HOLD {'PASS' if ok else 'FAIL'}")
    return ok
print("finger pads:", [g for g in info["finger_geoms"] if g and "pad" in g], "xoff", XOFF)
for H in (0.005, 0.0, -0.005):
    grasp_and_lift(H)
