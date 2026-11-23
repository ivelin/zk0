"""Scripted IK grasp, face-centred (oracle-guided): IK targets the
gripperframe site at cube + FACE_OFFSET, where FACE_OFFSET centres the
cube between the two pad faces (measured in geomprobe: fixed face plane
at site_x+~0.000, moving face at site_x+~0.021 at grip 0.1 rad, in the
grasp-pose world frame -> offset x -0.010). Closes slowly to CLOSE_TO,
lifts, holds >=2 s. Three descent heights via argv."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness")
import numpy as np, mujoco
import sim_runner_padded as SR

HEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
CLOSE_TO = float(sys.argv[2]) if len(sys.argv) > 2 else -0.10
model = SR.build_actuated_model(SR.SCENES / "block_stack__ep411_v0.xml")
SR.prepare_model(model); model.opt.timestep = SR.P.TIMESTEP
data = mujoco.MjData(model)
qadr = {j: int(model.joint(j).qposadr[0]) for j in SR.JOINTS}
vadr = {j: int(model.joint(j).dofadr[0]) for j in SR.JOINTS}
act = {j: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)) for j in SR.JOINTS}
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")
gt = np.load(SR.RECORDS / "base__block_stack__ep411.npz")["gt"].astype(float)
ARM = SR.JOINTS[:5]
def ik(target, seed_deg):
    c = SR.action_deg_to_ctrl(seed_deg)
    for k, j in enumerate(SR.JOINTS): data.qpos[qadr[j]] = c[k]
    mujoco.mj_forward(model, data)
    for _ in range(300):
        err = np.array(target) - data.site_xpos[sid]
        if np.linalg.norm(err) < 0.002: break
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        cols = [vadr[j] for j in ARM]; J = jacp[:, cols]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
        for i, j in enumerate(ARM): data.qpos[qadr[j]] += np.clip(dq[i], -0.05, 0.05)
        mujoco.mj_forward(model, data)
    return np.array([data.qpos[qadr[j]] for j in SR.JOINTS]), float(np.linalg.norm(np.array(target) - data.site_xpos[sid]))
def contacts():
    out = set()
    for i in range(data.ncon):
        cc = data.contact[i]
        b1 = int(model.geom_bodyid[cc.geom1]); b2 = int(model.geom_bodyid[cc.geom2])
        if cube in (b1, b2):
            out.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2 if b1 == cube else b1))
    return sorted(str(x) for x in out)
def run_to(ctrl_target, seconds):
    start = np.array([data.ctrl[act[j]] for j in SR.JOINTS])
    steps = max(1, int(seconds / SR.P.TIMESTEP))
    for s in range(steps):
        f = (s + 1) / steps
        cc = start * (1 - f) + np.array(ctrl_target) * f
        for k, j in enumerate(SR.JOINTS): data.ctrl[act[j]] = cc[k]
        mujoco.mj_step(model, data)
mujoco.mj_resetData(model, data)
c0 = SR.action_deg_to_ctrl(gt[0])
for k, j in enumerate(SR.JOINTS):
    data.qpos[qadr[j]] = c0[k]; data.ctrl[act[j]] = c0[k]
mujoco.mj_forward(model, data)
cube0 = np.array(data.xpos[cube]); print("cube at", np.round(cube0, 4), "height", HEIGHT, "close_to", CLOSE_TO)
OFF = np.array([float(sys.argv[3]) if len(sys.argv)>3 else -0.010, 0.0, 0.0])
OPEN = 1.20
wp = []
for tgt, grip in ((cube0 + OFF + [0, 0, 0.13], OPEN), (cube0 + OFF + [0, 0, HEIGHT], OPEN),
                  (cube0 + OFF + [0, 0, HEIGHT], CLOSE_TO), (cube0 + OFF + [0, 0, 0.14], CLOSE_TO)):
    q, err = ik(tgt, gt[130]); q[5] = grip; wp.append(q)
    print("waypoint", np.round(tgt, 3), "ik err", round(err, 4))
for k, j in enumerate(SR.JOINTS):
    data.qpos[qadr[j]] = c0[k]; data.ctrl[act[j]] = c0[k]
mujoco.mj_forward(model, data)
pre_close = None
for i, q in enumerate(wp):
    run_to(q, 1.2)
    if i == 1: pre_close = np.array(data.xpos[cube])
    if i == 2:
        dxy = np.linalg.norm(np.array(data.xpos[cube])[:2] - pre_close[:2])
        print(f"after wp{i} (close): cube {np.round(data.xpos[cube],4)} contacts {contacts()} grip_qpos {float(data.qpos[qadr['gripper']]):.3f} shove_xy_mm {dxy*1000:.1f}")
    else:
        print(f"after wp{i}: cube {np.round(data.xpos[cube],4)} contacts {contacts()} grip_qpos {float(data.qpos[qadr['gripper']]):.3f}")
run_to(wp[-1], 2.0)
print("final hold 2s: cube", np.round(data.xpos[cube], 4), contacts(), "lifted_mm", round((float(data.xpos[cube][2]) - 0.015) * 1000, 1))
