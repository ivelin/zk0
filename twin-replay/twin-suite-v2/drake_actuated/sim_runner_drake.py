"""Twin-suite v2 closed-loop policy rollout runner -- DRAKE engine.

Same contract as harness/sim_runner.py (MuJoCo): a SmolVLA checkpoint via
the SAME policy_service.py in the FL venv, JSON-lines over a subprocess
pipe, one request per 50-action chunk; renders 256x256 from the three
matched viewpoints; state [6] in degrees; same prompts; horizon =
recorded length + 20 settle steps at 20 fps; per-rollout records in the
same format (rollouts.jsonl + traj npz), appended + fsync'd, completed
rollouts skipped on re-run.

Engine differences (logged, global): Drake discrete plant dt 1 ms
(physics substeps are internal to AdvanceTo at the 20 fps boundary);
actuation is the plant's built-in PD desired-state controller with the
STS3215 gains/effort in actuated_port (MuJoCo: position actuators with
the same gains); ctrl targets slew-limited identically (6.16 rad/s);
renders via SceneGraph.RenderColorImage with per-render camera poses
(front = the MuJoCo harness's frontfix solution: frontcam position,
fovy 55, look-at workspace centre (0.30, 0, 0.05); top = v1.1 topcam
pos (0.3, 0, 0.85), fovy 50, looking down; wrist = the MuJoCo harness's
per-render site-relative pose, fovy 60). Scoring: drake_replay.evaluate
(the Drake re-implementation of the v1.1 mechanical criteria, Stage 1.2
part 1) + the same generalised solver-failure rule.
No weld, no pin: grasp is finger contact alone.
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("DK_CONTACT", "hydro")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake")
V1 = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
sys.path.insert(0, str(V1 / "code"))
import twin_params as P
import actuated_port as AP
from actuated_port import DrakeScene, JOINTS
import drake_replay
from pydrake.systems.analysis import Simulator
from pydrake.geometry import (RenderCameraCore, ColorRenderCamera, ClippingRange)
from pydrake.systems.sensors import CameraInfo  # this Drake build: sensors, not geometry
from pydrake.math import RigidTransform, RotationMatrix

FL_PY = "/home/openclaw/muse-work/zk0-study001-fl-20261121/venv/bin/python"
POLICY_SERVICE = "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness/policy_service.py"
RECORDS = P.RECORDS
FPS = 20.0
CHUNK = 50
SETTLE_STEPS = int(FPS * 1.0)
STS3215_SPEED_RAD_S = 6.16
GRIP_DEG_MAX = 52.3
GRIP_RAD_CLOSED, GRIP_RAD_OPEN = 0.10, 1.45
TASK_PROMPTS = {
    "block_stack": "Stack the colouful blocks on top of each other",
    "ring_insert": "Insert the colourful ring into the central wooden peg",
    "tool_insert": "Insert the missing tool into the empty slot on the toolbox",
    "eye_drops_to_basket": "Put the eye drops into the basket",
    "cable_clip": "Push the Displayport cable into the cable holder on the white block",
}
FRONT_POS = np.array([0.75, -0.55, 0.35]); FRONT_FOVY = 55.0
TOP_POS = np.array([0.3, 0.0, 0.85]); TOP_FOVY = 50.0
WRIST_FOVY = 60.0

def grip_deg_to_rad(d):
    return GRIP_RAD_CLOSED + (float(d) / GRIP_DEG_MAX) * (GRIP_RAD_OPEN - GRIP_RAD_CLOSED)

def grip_rad_to_deg(r):
    return (float(r) - GRIP_RAD_CLOSED) / (GRIP_RAD_OPEN - GRIP_RAD_CLOSED) * GRIP_DEG_MAX

def action_deg_to_ctrl(a_deg):
    a = np.asarray(a_deg, float)
    return np.array(np.radians(a[:5]).tolist() + [grip_deg_to_rad(a[5])])

def _lookat_R(pos, target):
    """Camera rotation in Drake's render convention (FIXED 2026-11-23,
    verification finding F1 -- BLANK RENDERS). The Drake RenderEngine
    camera frame is x=right, y=DOWN, z=FORWARD (into the scene). The
    previous version built R with -z toward the target (the OpenGL /
    MuJoCo camera convention), so every camera pointed exactly AWAY
    from the scene and every frame -- saved PNGs and the .npy frames
    fed to the policy alike -- was the renderer's clear colour
    (204, 229, 255) alone: the policy ran blind. Columns here are the
    MuJoCo harness's look-at (world up +z) mapped into Drake's frame:
    z = fwd, x = normalize(cross(z, up)) (image right), y = cross(z, x)
    (image down). The up fallback (0,1,0) reproduces the v1.1 topcam's
    identity-quat orientation (image right = +x, image up = +y)."""
    pos = np.array(pos, float); target = np.array(target, float)
    fwd = target - pos; fwd /= np.linalg.norm(fwd)
    z = fwd
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return RotationMatrix(np.column_stack([x, y, z]))

class Renderer:
    def __init__(self, scene):
        from pydrake.geometry import MakeRenderEngineVtk, RenderEngineVtkParams
        self.sg = scene.sg
        if not self.sg.HasRenderer("drake_actuated_renderer"):
            self.sg.AddRenderer("drake_actuated_renderer",
                                MakeRenderEngineVtk(RenderEngineVtkParams()))
        self.sg_context = scene.diagram.GetSubsystemContext(self.sg, scene.context)
        self.world_id = self.sg.world_frame_id()
    def render(self, pos, target, fovy_deg):
        core = RenderCameraCore("drake_actuated_renderer",
                                CameraInfo(256, 256, math.radians(fovy_deg)),
                                ClippingRange(0.01, 10.0), RigidTransform())
        cam = ColorRenderCamera(core, False)
        X = RigidTransform(_lookat_R(pos, target), np.array(pos, float))
        qo = self.sg.get_query_output_port().Eval(self.sg_context)
        img = qo.RenderColorImage(cam, self.world_id, X)
        return np.asarray(img.data)[:, :, :3].copy()

class PolicyService:
    def __init__(self, ckpt_path):
        self.proc = subprocess.Popen(
            [FL_PY, POLICY_SERVICE, ckpt_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "POLICY_THREADS": "4", "CUDA_VISIBLE_DEVICES": ""})
        self.ready = None
        for _ in range(200):
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("policy service exited during startup")
            if line.startswith("READY "):
                self.ready = json.loads(line[6:]); break
        if self.ready is None:
            raise RuntimeError("policy service never became ready")
    def query(self, state_deg, task, episode, chunk_idx, frames):
        req = {"state_deg": state_deg, "task": task, "episode": episode,
               "chunk": chunk_idx, "frames": frames}
        self.proc.stdin.write(json.dumps(req) + "\n"); self.proc.stdin.flush()
        for _ in range(50):
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("policy service exited mid-rollout")
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if "chunk_deg" in msg:
                return np.array(msg["chunk_deg"])
        raise RuntimeError("no chunk response from policy service")
    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--variant", default="nominal")
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--checkpoint-path", default="")
    ap.add_argument("--policy", choices=["service", "gt"], default="service")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-frames", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "rollouts.jsonl"
    key = {"task": args.task, "episode": args.episode, "variant": str(args.variant),
           "checkpoint_id": args.checkpoint_id}
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            try: r = json.loads(line)
            except Exception: continue
            if all(r.get(k) == v for k, v in key.items()) and r.get("status") == "complete":
                print(f"[rollout] already recorded, skipping {key}"); return

    variant = 0 if args.variant == "nominal" else int(args.variant)
    scene = DrakeScene(args.task, args.episode, variant)
    plant = scene.plant
    renderer = Renderer(scene)
    sim = Simulator(scene.diagram, scene.context)
    sim.set_target_realtime_rate(0.0)
    sim.Initialize()

    gt = np.load(RECORDS / f"base__{args.task}__ep{args.episode}.npz")["gt"].astype(float)
    n_steps = len(gt) + SETTLE_STEPS
    var_def = None
    vj = V1 / "scenes" / f"{args.task}__ep{args.episode}_variants.json"
    if vj.exists() and args.variant != "nominal":
        vd = json.loads(vj.read_text())
        var_def = vd["variants"][int(args.variant)]

    ctrl0 = action_deg_to_ctrl(gt[0])
    q0 = {j: float(ctrl0[k]) for k, j in enumerate(JOINTS)}
    scene.set_arm_and_desired(q0)
    now = [0.0]
    def advance(dt):
        now[0] += dt; sim.AdvanceTo(now[0])
    advance(0.25)  # pre-settle, arm held at recorded first-frame pose

    frame_dir = out / "frames_tmp"; frame_dir.mkdir(exist_ok=True)
    frame_stats_chunks = []  # per-chunk {view: {unique_colors, non_clear_frac}}
    svc = PolicyService(args.checkpoint_path) if args.policy == "service" else None
    prompt = TASK_PROMPTS[args.task]
    traj_qpos, traj_ctrl, traj_obj = [], [], {}
    cur_ctrl = ctrl0.copy()
    chunk_actions, chunk_idx = None, -1
    t0 = time.time(); solver_bad = False

    def state_deg():
        aq = scene.arm_q()
        deg = [math.degrees(aq[j]) for j in JOINTS[:5]]
        deg.append(grip_rad_to_deg(aq["gripper"]))
        return deg

    def render_views(chunk_idx):
        frames = {}
        site = scene.site_pos()
        gp = scene.gripper_pose().translation()
        fwd = site - gp; fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        eye = gp - fwd * 0.06 + np.array([0.0, 0.0, 0.05])
        tgt = site + fwd * 0.10
        views = {"front": (FRONT_POS, np.array([0.30, 0.0, 0.05]), FRONT_FOVY),
                 "top": (TOP_POS, TOP_POS + np.array([0.0, 0.0, -1.0]), TOP_FOVY),
                 "wrist": (eye, tgt, WRIST_FOVY)}
        _stats = {}
        for ds, (pos, target, fovy) in views.items():
            img = renderer.render(pos, target, fovy)
            _clear = np.array([204, 229, 255])  # RenderEngineVtk default clear colour
            _stats[ds] = {"unique_colors": int(len(np.unique(img.reshape(-1, 3), axis=0))),
                          "non_clear_frac": round(float(np.mean(np.any(img != _clear, axis=2))), 4)}
            fp = frame_dir / f"{ds}.npy"; np.save(fp, img)
            frames[ds] = str(fp)
            if args.save_frames and chunk_idx == 0:
                try:
                    from PIL import Image
                    Image.fromarray(img).save(out / f"frame0_{ds}.png")
                except Exception:
                    np.save(out / f"frame0_{ds}.npy", img)
        frame_stats_chunks.append(_stats)
        return frames

    for step in range(n_steps):
        if step < len(gt) and step % CHUNK == 0:
            chunk_idx += 1
            frames = render_views(chunk_idx)
            if args.policy == "gt":
                chunk_actions = gt[step:step + CHUNK]
            else:
                chunk_actions = svc.query(state_deg(), prompt, args.episode, chunk_idx, frames)
        if step < len(gt):
            target = action_deg_to_ctrl(chunk_actions[min(step % CHUNK, len(chunk_actions) - 1)])
        else:
            target = cur_ctrl
        max_d = STS3215_SPEED_RAD_S / FPS
        cur_ctrl = cur_ctrl + np.clip(target - cur_ctrl, -max_d, max_d)
        scene.set_desired({j: float(cur_ctrl[k]) for k, j in enumerate(JOINTS)})
        advance(1.0 / FPS)
        aq = scene.arm_q()
        traj_qpos.append([aq[j] for j in JOINTS])
        traj_ctrl.append(cur_ctrl.tolist())
        for nm in scene.weldables:
            traj_obj.setdefault(nm, []).append(scene.body_pose(nm).translation().tolist())
        if not np.all(np.isfinite(np.array(traj_qpos[-1]))):
            solver_bad = True; break
    if svc: svc.close()

    result = drake_replay.evaluate(scene, args.task, args.episode)
    gen_solver = bool(solver_bad)
    for nm, arr in traj_obj.items():
        a = np.array(arr[-1]) if arr else np.array([np.nan])
        if not np.all(np.isfinite(a)) or np.linalg.norm(a[:2]) > 1.0 or not (-0.05 <= a[2] <= 0.50):
            gen_solver = True
    record = {
        **key, "status": "complete", "suite": "twin-suite-v2-dev", "engine": "drake",
        "policy_mode": args.policy, "prompt": prompt,
        "horizon_steps": n_steps, "fps": FPS,
        "variant_draws": (var_def or {}).get("draws") if var_def else "nominal",
        "sim_success": bool(result["sim_success"]) and not gen_solver,
        "secondary_pass": result.get("secondary_pass"),
        "solver_failure": bool(result.get("solver_failure") or gen_solver),
        "excluded": bool(result.get("solver_failure") or gen_solver),
        "checks": result.get("checks"),
        "objects_final": result.get("objects_final"),
        "actuator_info": {"engine": "drake", "gains": scene.actuator_info,
                          "finger_geoms": scene.finger_geoms,
                          "finger_mesh_failed": scene.finger_mesh_failed,
                          "arm_visual_obj_clones": getattr(scene, "arm_visual_obj_clones", None),
                          "contact_model": os.environ.get("DK_CONTACT")},
        "wall_s": round(time.time() - t0, 2),
        "frame_content": {"chunk0": frame_stats_chunks[0] if frame_stats_chunks else {},
                          "min_non_clear_frac": {ds: min((c[ds]["non_clear_frac"] for c in frame_stats_chunks if ds in c), default=None)
                                                 for ds in ("front", "top", "wrist")}},
    }
    np.savez_compressed(out / f"traj_{args.task}__ep{args.episode}_{args.variant}.npz",
                        qpos=np.array(traj_qpos), ctrl=np.array(traj_ctrl),
                        **{f"obj_{k}": np.array(v) for k, v in traj_obj.items()})
    with open(rec_path, "a") as f:
        f.write(json.dumps(record) + "\n"); f.flush(); os.fsync(f.fileno())
    print(json.dumps({k: record[k] for k in
          ("task", "episode", "variant", "sim_success", "secondary_pass",
           "solver_failure", "wall_s")}, default=str), flush=True)

if __name__ == "__main__":
    main()
