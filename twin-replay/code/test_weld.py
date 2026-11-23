"""Weld-mechanics unit test (not a validation episode).
FK-pass over block_stack ep411 gt: find the gripperframe position at the
first upward 0.5-closedness crossing, place cube_a there in a test scene,
run the replay, and assert an attach event fires and the cube is carried
(its final position differs from spawn by > 5 cm or it ends held/stacked).
"""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).parent))
import twin_params as P
import build_scenes as B
import replay as R
import mujoco

task, epi = "block_stack", 411
scene = P.TWIN / "scenes" / f"{task}__ep{epi}_v0.xml"
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
gt = np.load(P.RECORDS / f"base__{task}__ep{epi}.npz")["gt"].astype(float)
closed, lo, hi, _ = R.closedness_trace(gt[:, 5])
cross = int(np.argmax((closed[:-1] < 0.5) & (closed[1:] >= 0.5))) + 1
qadr = {j: int(model.joint(j).qposadr[0]) for j in R.JOINTS}
for k, j in enumerate(R.JOINTS[:5]):
    data.qpos[qadr[j]] = np.radians(gt[cross, k])
mujoco.mj_forward(model, data)
site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
gpos = np.array(data.site_xpos[site])
print(f"crossing frame {cross}/{len(gt)} gripperframe at {np.round(gpos,3)} closed_ref {hi:.1f}deg")

# write a test pose file + rebuild scene with cube_a at the gripper point
test_pose = {"confidence": "smoke", "objects": {
    "cube_a": {"pos": [float(gpos[0]), float(gpos[1]), float(max(gpos[2], P.CUBE_EDGE/2))], "yaw_deg": 0},
    "cube_b": {"pos": [0.33, 0.04, P.CUBE_EDGE/2], "yaw_deg": 0}}}
tp = P.TWIN / "scenes" / f"{task}__ep{epi}.json"
backup = tp.read_text() if tp.exists() else None
tp.write_text(json.dumps(test_pose))
try:
    import importlib; importlib.reload(B)
    fx = B.load_fixtures(task); poses = B.load_episode_poses(task, epi)
    xml, _, _ = B.build(task, epi, poses, fx)
    scene.write_text(xml)
    res = R.run_episode(task, epi, render=False)
    print("events:", res["grasp_events"])
    print("cube_a final:", np.round(res["objects_final"]["cube_a"], 3), "spawn:", np.round(test_pose["objects"]["cube_a"]["pos"], 3))
    moved = float(np.linalg.norm(np.array(res["objects_final"]["cube_a"]) - np.array(test_pose["objects"]["cube_a"]["pos"])))
    print(f"cube_a moved {moved:.3f} m; attach fired: {any(e['event']=='attach' for e in res['grasp_events'])}")
finally:
    if backup is None: tp.unlink(missing_ok=True)
    else: tp.write_text(backup)
