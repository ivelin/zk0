"""Isolated physical-grasp probe (diagnostic, not a gate input).

Places the actuated arm kinematically at the recorded grasp pose of
block_stack ep411 (frame F0, just before the recorded close), then under
POSITION CONTROL ONLY: closes the gripper to the recorded closed value,
lifts by interpolating ctrl toward the recorded transit pose (frame F1),
and reports cube height + finger contact pairs throughout. Answers:
can the vendored fingers hold a 30 mm / 20 g cube by friction alone?
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness")
import numpy as np, mujoco
import sim_runner_padded as SR

F0, F1 = 190, 245
model = SR.build_actuated_model(SR.SCENES / "block_stack__ep411_v0.xml")
info = SR.prepare_model(model)
model.opt.timestep = SR.P.TIMESTEP
data = mujoco.MjData(model)
qadr = {j: int(model.joint(j).qposadr[0]) for j in SR.JOINTS}
act = {j: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)) for j in SR.JOINTS}
gt = np.load(SR.RECORDS / "base__block_stack__ep411.npz")["gt"].astype(float)
cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")

def set_ctrl(deg):
    c = SR.action_deg_to_ctrl(deg)
    for k, j in enumerate(SR.JOINTS): data.ctrl[act[j]] = c[k]
    return c

def cube_contacts():
    out = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1]); b2 = int(model.geom_bodyid[c.geom2])
        if cube in (b1, b2):
            out.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2 if b1 == cube else b1))
    return sorted(str(x) for x in out)

# kinematic placement at F0 (open gripper per recording), then hold by ctrl
mujoco.mj_resetData(model, data)
c0 = SR.action_deg_to_ctrl(gt[F0])
for k, j in enumerate(SR.JOINTS): data.qpos[qadr[j]] = c0[k]
set_ctrl(gt[F0]); mujoco.mj_forward(model, data)
print("at F0: cube", np.round(data.xpos[cube], 4), "contacts", cube_contacts(),
      "grip_qpos", round(float(data.qpos[qadr['gripper']]), 3))
spf = int(round((1.0 / SR.FPS) / SR.P.TIMESTEP))
def run_ctrl(deg, seconds):
    set_ctrl(deg)
    for _ in range(int(seconds / SR.P.TIMESTEP)): mujoco.mj_step(model, data)
run_ctrl(gt[F0], 0.5)
print("held 0.5s: cube", np.round(data.xpos[cube], 4), "contacts", cube_contacts())
# close gripper to the recorded minimum (fully closed phase value)
closed_deg = gt[F0:F1, 5].min()
tgt = gt[F0].copy(); tgt[5] = float(closed_deg)
run_ctrl(tgt, 1.0)
print(f"closed to deg={closed_deg:.1f}: cube", np.round(data.xpos[cube], 4),
      "contacts", cube_contacts(), "grip_qpos", round(float(data.qpos[qadr['gripper']]), 3))
# lift: interpolate ctrl F0->F1 arm joints over the recorded duration, gripper held closed
dur = (F1 - F0) / SR.FPS
steps = int(dur * SR.FPS)
zmax = float(data.xpos[cube][2])
for s in range(steps):
    f = (s + 1) / steps
    deg = gt[F0][:5] * (1 - f) + gt[F1][:5] * f
    set_ctrl(np.concatenate([deg, [closed_deg]]))
    for _ in range(spf): mujoco.mj_step(model, data)
    zmax = max(zmax, float(data.xpos[cube][2]))
print(f"after lift: cube", np.round(data.xpos[cube], 4), "z_max", round(zmax, 4),
      "contacts", cube_contacts())
run_ctrl(np.concatenate([gt[F1][:5], [closed_deg]]), 1.0)
print("held 1s at transit: cube", np.round(data.xpos[cube], 4), "contacts", cube_contacts())
