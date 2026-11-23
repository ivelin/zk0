import os, sys, json
os.environ.setdefault("DK_CONTACT", "hydro")
HERE = "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake_actuated"
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-20261122/code")
import numpy as np
from PIL import Image
import twin_params as P
from actuated_port import DrakeScene, JOINTS
from sim_runner_drake import Renderer, action_deg_to_ctrl, FRONT_POS, TOP_POS, FRONT_FOVY, TOP_FOVY, WRIST_FOVY

OUT = os.path.join(HERE, "render_fix_evidence")
os.makedirs(OUT, exist_ok=True)
scene = DrakeScene("block_stack", 411, 0)
print("arm_visual_obj_clones:", scene.arm_visual_obj_clones); print("arm_visual_strip_failed:", scene.arm_visual_strip_failed)
renderer = Renderer(scene)
gt = np.load(P.RECORDS / "base__block_stack__ep411.npz")["gt"].astype(float)

def views():
    site = scene.site_pos()
    gp = scene.gripper_pose().translation()
    fwd = site - gp; fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    eye = gp - fwd * 0.06 + np.array([0.0, 0.0, 0.05])
    tgt = site + fwd * 0.10
    return {"front": (FRONT_POS, np.array([0.30, 0.0, 0.05]), FRONT_FOVY),
            "top": (TOP_POS, TOP_POS + np.array([0.0, 0.0, -1.0]), TOP_FOVY),
            "wrist": (eye, tgt, WRIST_FOVY)}

def colour_stats(img, rgb, tol=70):
    d = np.linalg.norm(img.astype(int) - np.array(rgb)[None, None, :], axis=2)
    mask = d < tol
    n = int(mask.sum())
    if n:
        ys, xs = np.nonzero(mask)
        return {"pixels": n, "centroid_xy": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]}
    return {"pixels": 0, "centroid_xy": None}

GREEN = (64, 166, 76)    # cube_a ep411 rgba 0.25 0.65 0.30
PURPLE = (140, 64, 191)  # cube_b ep411 rgba 0.55 0.25 0.75
YELLOW = (255, 209, 31)  # arm 3d_printed 1.0 0.82 0.12
MAT = (41, 48, 61)       # mat 0.16 0.19 0.24
WALL = (194, 168, 128)   # walls 0.76 0.66 0.50

report = {}
for tag, frame_idx in [("frame0", 0), ("mid130", 130)]:
    ctrl = action_deg_to_ctrl(gt[frame_idx])
    q0 = {j: float(ctrl[k]) for k, j in enumerate(JOINTS)}
    scene.set_arm_and_desired(q0)
    report[tag] = {}
    for ds, (pos, target, fovy) in views().items():
        img = renderer.render(pos, target, fovy)
        Image.fromarray(img).save(os.path.join(OUT, f"drake_{tag}_{ds}.png"))
        np.save(os.path.join(OUT, f"drake_{tag}_{ds}.npy"), img)
        uniq = int(len(np.unique(img.reshape(-1, 3), axis=0)))
        clear = np.array([204, 229, 255])
        nonclear = round(float(np.mean(np.any(img != clear, axis=2))), 4)
        report[tag][ds] = {"unique_colors": uniq, "non_clear_frac": nonclear,
                           "cube_a_green": colour_stats(img, GREEN),
                           "cube_b_purple": colour_stats(img, PURPLE),
                           "arm_yellow": colour_stats(img, YELLOW),
                           "mat_dark": colour_stats(img, MAT),
                           "wall_beige": colour_stats(img, WALL)}
        print(tag, ds, json.dumps(report[tag][ds]))
with open(os.path.join(OUT, "render_proof.json"), "w") as f:
    json.dump({"arm_visual_obj_clones": scene.arm_visual_obj_clones, "frames": report}, f, indent=1)
print("PROOF_DONE")
