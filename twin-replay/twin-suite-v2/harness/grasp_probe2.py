"""Grasp probe 2: close-sweep + vertical lift test at the recorded grasp pose."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness")
import numpy as np, mujoco
import sim_runner as SR

F0 = 130
model = SR.build_actuated_model(SR.SCENES / "block_stack__ep411_v0.xml")
SR.prepare_model(model)
model.opt.timestep = SR.P.TIMESTEP
data = mujoco.MjData(model)
qadr = {j: int(model.joint(j).qposadr[0]) for j in SR.JOINTS}
act = {j: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)) for j in SR.JOINTS}
gt = np.load(SR.RECORDS / "base__block_stack__ep411.npz")["gt"].astype(float)
cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")
jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

def cube_contacts():
    out = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1]); b2 = int(model.geom_bodyid[c.geom2])
        if cube in (b1, b2):
            out.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2 if b1 == cube else b1))
    return sorted(str(x) for x in out)

mujoco.mj_resetData(model, data)
c0 = SR.action_deg_to_ctrl(gt[F0])
for k, j in enumerate(SR.JOINTS):
    data.qpos[qadr[j]] = c0[k]; data.ctrl[act[j]] = c0[k]
mujoco.mj_forward(model, data)
base_ctrl = c0.copy()
def hold(sec):
    for _ in range(int(sec / SR.P.TIMESTEP)): mujoco.mj_step(model, data)
hold(0.5)
print("start cube", np.round(data.xpos[cube], 4), cube_contacts())
for target in (0.162, 0.10, 0.05, 0.0, -0.05):
    data.ctrl[act["gripper"]] = target
    hold(0.6)
    print(f"close->{target:+.3f}: grip_qpos {float(data.qpos[qadr['gripper']]):+.3f} cube {np.round(data.xpos[cube],4)} contacts {cube_contacts()}")
# vertical lift: raise shoulder_lift ctrl by +0.25 rad over 1 s, hold gripper at -0.05
c_lift = base_ctrl.copy(); c_lift[SR.JOINTS.index("shoulder_lift")] += 0.25; c_lift[SR.JOINTS.index("gripper")] = -0.05
steps = int(1.0 / SR.P.TIMESTEP)
start = np.array([data.ctrl[act[j]] for j in SR.JOINTS])
for s in range(steps):
    f = (s + 1) / steps
    cc = start * (1 - f) + c_lift * f
    for k, j in enumerate(SR.JOINTS): data.ctrl[act[j]] = cc[k]
    mujoco.mj_step(model, data)
hold(0.5)
print("after lift: cube", np.round(data.xpos[cube], 4), cube_contacts(),
      "grip_qpos", round(float(data.qpos[qadr['gripper']]), 3))
