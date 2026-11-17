"""Sim-generated demonstration datasets (the FL nodes' local data).

Each node's local training set is built the same way: the certified
scripted pilot (generation mark ORACLE-GUIDED) performs SIM-1 in the
package's sim, and the rollout is recorded at exactly 30 fps into a
LeRobot dataset with the checkpoint's own feature shapes (state in
degrees, front + top camera images, actions in degrees). Nodes
differ by their episode draws (object/place jitter from a per-node
seed) — small, local, and *labelled*: this is synthetic sim data
for exercising the federation machinery, provenance sim-trained /
generation ORACLE-GUIDED wherever it is cited. It is not operator
data and is never described as such.

Process note: the vendored multitask module reads its model dir
from the environment at import time, so this module must be the
first multitask consumer in its process (the CLI's dataset-build
command is exactly that).
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from .config import zk0_home
from .records import utcnow_iso
from .sim import assets

FPS = 30
TS = 1.0 / 480.0          # 16 physics steps per recorded frame
STEPS_PER_FRAME = 16


def build_demo_dataset(node_tag: str, episodes: int = 4, seed: int = 0,
                       out_root=None) -> dict:
    """Build (or rebuild) one node's demo dataset. Returns its
    identity record: repo id, root, episode/frame counts, and a
    hash over the generation plan."""
    from .sim import policy_eval as PE

    root = Path(out_root) if out_root else (
        zk0_home() / "datasets" / f"{node_tag}-sim1-demo")
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    # Private model copy for planning + execution (assets stay clean).
    work = root.parent / f".{node_tag}-demodata-model"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(assets.model_dir("realistic"), work)
    os.environ["ZK0_SIM_MODEL_DIR"] = str(work)
    sys.path.insert(0, str(PE.HARNESS_DIR))
    import multitask as MT

    import mujoco

    rng = np.random.default_rng(seed)
    plan = []
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,),
                              "names": ["state"]},
        "observation.images.front": {"dtype": "video",
                                     "shape": (240, 320, 3),
                                     "names": ["height", "width",
                                               "channels"]},
        "observation.images.top": {"dtype": "video",
                                   "shape": (240, 320, 3),
                                   "names": ["height", "width",
                                             "channels"]},
        "action": {"dtype": "float32", "shape": (6,),
                   "names": ["action"]},
    }
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset.create(repo_id=f"zk0/{node_tag}-sim1-demo",
                               fps=FPS, features=features, root=root,
                               use_videos=True)
    total_frames = 0
    for ep in range(episodes):
        obj_xy = (0.26 + float(rng.uniform(-0.006, 0.006)),
                  0.02 + float(rng.uniform(-0.006, 0.006)))
        place_xy = (0.27 + float(rng.uniform(-0.006, 0.006)),
                    -0.14 + float(rng.uniform(-0.006, 0.006)))
        plan.append({"episode": ep,
                     "obj_xy": [round(v, 4) for v in obj_xy],
                     "place_xy": [round(v, 4) for v in place_xy]})
        # Plan with the certified pilot (planning only — the segments
        # are joint-space and transfer to the execution model, which
        # is the same arm in the policy scene with cameras).
        P = MT.Pilot("task1", bodies=["cube_green"])
        MT.pick_place(P, obj_xy, 0.02, -0.01, MT.GRIP_CLOSE_CUBE, 0.135,
                      place_xy, 0.028, tag="t1", body="cube_green")
        P.home()
        P.dwell(1.2)
        xml = PE._scene_xml("sim1", obj_xy, 0.0, 0.35)
        scene_path = work / "scene_demodata_sim1.xml"
        scene_path.write_text(xml)
        m = mujoco.MjModel.from_xml_path(str(scene_path))
        m.opt.timestep = TS
        MT.aim_camera(m, m.cam("front").id,
                      np.array(PE.FRONT_CAM_POS),
                      np.array(PE.FRONT_CAM_LOOKAT))
        qadr = [m.jnt(n).qposadr[0] for n in MT.JOINTS]
        d = mujoco.MjData(m)
        d.qpos[qadr[5]] = MT.GRIP_OPEN
        mujoco.mj_forward(m, d)
        r_front = mujoco.Renderer(m, height=PE.IMG_H, width=PE.IMG_W)
        r_top = mujoco.Renderer(m, height=PE.IMG_H, width=PE.IMG_W)
        n_frames = int(P.t * FPS)
        for fi in range(n_frames):
            t = fi / FPS
            ctrl_rad = MT.target_q(P.segs, t)
            mujoco.mj_forward(m, d)
            frame = {
                "observation.state": np.degrees(
                    d.qpos[qadr]).astype(np.float32),
                "observation.images.front": PE._img_tensor(
                    r_front, m, d, "front"),
                "observation.images.top": PE._img_tensor(
                    r_top, m, d, "top"),
                "action": np.degrees(ctrl_rad).astype(np.float32),
                "task": PE.TASK_TEXT["sim1"],
            }
            ds.add_frame(frame)
            total_frames += 1
            for _ in range(STEPS_PER_FRAME):
                d.ctrl[:] = ctrl_rad
                mujoco.mj_step(m, d)
        ds.save_episode()
    ds.finalize()  # LeRobot 0.7: flushes episode metadata
    shutil.rmtree(work, ignore_errors=True)
    plan_hash = hashlib.sha256(json.dumps(
        {"node_tag": node_tag, "seed": seed, "plan": plan,
         "frames": total_frames}, sort_keys=True).encode()).hexdigest()
    return {"repo_id": f"zk0/{node_tag}-sim1-demo", "root": str(root),
            "episodes": episodes, "frames": total_frames,
            "fps": FPS, "plan": plan, "plan_hash": plan_hash,
            "generation_mark": "ORACLE-GUIDED",
            "provenance": "sim-generated demonstrations (scripted "
                          "pilot), built by zk0 demodata",
            "created_utc": utcnow_iso()}
