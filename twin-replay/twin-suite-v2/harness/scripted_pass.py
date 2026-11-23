"""Scripted reference pass, block_stack ep411 nominal (A2.2 solvability
executor evidence). GENERATION: ORACLE-GUIDED.
Grasp cube_a with the padded gripper (executor from fingerpad_probes.py),
lift, translate above cube_b, place cube_a on cube_b, release, retreat,
settle, then score with the v1.1 mechanical criteria (replay.evaluate,
read-only import). No welds (asserted inactive by construction: scene
welds ship active=false and nothing activates them)."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-20261122/code")
import numpy as np, mujoco
import sim_runner_padded as SR
import replay as v1replay
XOFF, PRE, CLOSE, H = -0.020, 0.7, 0.05, 0.0
model = SR.build_actuated_model(SR.SCENES / "block_stack__ep411.xml")  # nominal
info = SR.prepare_model(model); model.opt.timestep = SR.P.TIMESTEP
data = mujoco.MjData(model)
qadr = {j: int(model.joint(j).qposadr[0]) for j in SR.JOINTS}
vadr = {j: int(model.joint(j).dofadr[0]) for j in SR.JOINTS}
act = {j: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)) for j in SR.JOINTS}
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
gbody = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
cubeA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")
cubeB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_b")
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
def contacts(body):
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
        assert not np.any(data.eq_active[:model.neq]), "weld activated!"
def seed_now():
    return [np.degrees(data.qpos[qadr[j]]) for j in SR.JOINTS[:5]] + [10.0]
_last = [None]
def ik3(target, seed_deg):
    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    c = SR.action_deg_to_ctrl(seed_deg)
    for k, j in enumerate(SR.JOINTS): data.qpos[qadr[j]] = c[k]
    mujoco.mj_forward(model, data)
    cols = [vadr[j] for j in ARM]
    for _ in range(300):
        err = np.array(target) - data.site_xpos[sid]
        if np.linalg.norm(err) < 0.002: break
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = jacp[:, cols]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
        for i, j in enumerate(ARM): data.qpos[qadr[j]] += np.clip(dq[i], -0.05, 0.05)
        mujoco.mj_forward(model, data)
    q = np.array([data.qpos[qadr[j]] for j in SR.JOINTS])
    data.qpos[:] = saved[0]; data.qvel[:] = saved[1]; data.ctrl[:] = saved[2]
    mujoco.mj_forward(model, data)
    return q
def staged_move3(target, grip, stage=0.01, secs=0.4, seed_fn=None):
    site0 = np.array(data.site_xpos[sid]); target = np.array(target, float)
    delta = target - site0; n = max(1, int(np.ceil(np.linalg.norm(delta) / stage)))
    for i in range(n):
        t = site0 + delta * ((i + 1) / n)
        q = ik3(t, (seed_fn() if seed_fn else seed_now())); q[5] = grip
        run_to(q, secs)
def staged_move(target, grip, stage=0.01, secs=0.4):
    site0 = np.array(data.site_xpos[sid]); target = np.array(target, float)
    delta = target - site0; n = max(1, int(np.ceil(np.linalg.norm(delta) / stage)))
    for i in range(n):
        t = site0 + delta * ((i + 1) / n)
        seed = seed_now()
        q = ik6(t, seed); q[5] = grip
        _last[0] = [np.degrees(q[j]) for j in range(5)] + [10.0]
        run_to(q, secs)
mujoco.mj_resetData(model, data)
c0 = SR.action_deg_to_ctrl(gt[0])
for k, j in enumerate(SR.JOINTS):
    data.qpos[qadr[j]] = c0[k]; data.ctrl[act[j]] = c0[k]
mujoco.mj_forward(model, data)
for _ in range(int(0.25 / SR.P.TIMESTEP)): mujoco.mj_step(model, data)
A0 = np.array(data.xpos[cubeA]); B0 = np.array(data.xpos[cubeB])
print("cube_a", np.round(A0, 4), "cube_b", np.round(B0, 4))
OFF = np.array([XOFF, 0.0, 0.0])
wp = []
for tgt, grip in ((A0 + OFF + [0, 0, 0.13], PRE), (A0 + OFF + [0, 0, H], PRE), (A0 + OFF + [0, 0, H], CLOSE)):
    q = ik6(tgt, gt[130]); q[5] = grip; wp.append(q)
for k, j in enumerate(SR.JOINTS):
    data.qpos[qadr[j]] = c0[k]; data.ctrl[act[j]] = c0[k]
mujoco.mj_forward(model, data)
run_to(wp[0], 1.0); run_to(wp[1], 1.0); run_to(wp[2], 1.2)
grip_q = float(data.qpos[qadr['gripper']])
print("grasped: cube_a", np.round(data.xpos[cubeA], 4), "grip_qpos", round(grip_q, 3), "contacts", contacts(cubeA))
hold_grip = grip_q - 0.05
site0 = np.array(data.site_xpos[sid])
staged_move(site0 + [0, 0, 0.07], hold_grip)
print("lifted: cube_a", np.round(data.xpos[cubeA], 4), "contacts", contacts(cubeA))
place_site = B0 + np.array([-0.008, 0.005, 0.0417])  # cube-centre target B0+(0,0,0.030) minus measured cube-minus-site offset (+0.008,-0.005,-0.0116)
cur = np.array(data.site_xpos[sid])
staged_move3([place_site[0], place_site[1], cur[2]], hold_grip, seed_fn=lambda: gt[130])
staged_move3(place_site, hold_grip, seed_fn=lambda: gt[130])
print("site after place move", np.round(data.site_xpos[sid], 4), "target", np.round(place_site, 4))
print("placed: cube_a", np.round(data.xpos[cubeA], 4), "contacts", contacts(cubeA))
q = np.array([data.ctrl[act[j]] for j in SR.JOINTS]); q[5] = 1.20
run_to(q, 1.0)
print("released: cube_a", np.round(data.xpos[cubeA], 4), "contacts", contacts(cubeA))
cur = np.array(data.site_xpos[sid])
staged_move3(cur + [0, 0, 0.10], 1.20)
for _ in range(int(1.0 / SR.P.TIMESTEP)): mujoco.mj_step(model, data)
mujoco.mj_forward(model, data)
res = v1replay.evaluate(model, data, "block_stack", 411, mujoco)
print("FINAL cube_a", np.round(data.xpos[cubeA], 4), "cube_b", np.round(data.xpos[cubeB], 4))
print("SCORE", json.dumps({k: res.get(k) for k in ("sim_success", "secondary_pass", "solver_failure")}, default=str))
print("CHECKS", json.dumps(res.get("checks"), default=str))
print("OBJECTS_FINAL", json.dumps(res.get("objects_final"), default=str))
