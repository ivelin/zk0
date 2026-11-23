"""Cross-engine geometry parity: MuJoCo pad geom world positions at the
recorded frame-130 pose of block_stack ep411 (padded model). Companion to
grasp_probes_drake.py --padpos. Runs in the sim venv."""
import json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness")
import numpy as np, mujoco
import sim_runner_padded as SR

model = SR.build_actuated_model(SR.SCENES / "block_stack__ep411_v0.xml")
SR.prepare_model(model); model.opt.timestep = SR.P.TIMESTEP
data = mujoco.MjData(model)
qadr = {j: int(model.joint(j).qposadr[0]) for j in SR.JOINTS}
gt = np.load(SR.RECORDS / "base__block_stack__ep411.npz")["gt"].astype(float)
c = SR.action_deg_to_ctrl(gt[130])
for k, j in enumerate(SR.JOINTS):
    data.qpos[qadr[j]] = c[k]
mujoco.mj_forward(model, data)
out = {}
for nm in ("finger_pad_fixed", "finger_pad_moving"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
    out[nm] = np.array(data.geom_xpos[gid]).tolist()
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
out["gripperframe_site"] = np.array(data.site_xpos[sid]).tolist()
print(json.dumps(out, indent=1))
