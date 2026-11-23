"""Twin-suite v2 closed-loop rollout runner (MuJoCo 4.0.1 sim venv).

Drives ONE rollout: a SmolVLA checkpoint (via policy_service.py in the FL
venv, JSON-lines over a subprocess pipe, one request per 50-action chunk)
runs closed-loop in an ACTUATED twin scene at 20 fps (dataset rate).

Actuation (v2, per pre-registration s.4): the vendored SO-101 model already
carries <position> actuators of class "sts3215" (kp=998.22, kv=2.731,
forcerange +-3.35 N m at the actuator, class forcerange +-2.94 N m;
so101_new_calib.xml). v1.1 replay never used them (it set qpos
kinematically). Here data.ctrl is the ONLY way the arm moves, with an
additional per-joint slew limit of STS3215_SPEED_RAD_S applied to ctrl
targets (Feetech STS3215 rated 0.17 s/60 deg at 7.4 V ~= 6.16 rad/s;
instrument parameter, see report).

Grasp: PHYSICAL. The v1 weld equalities are forced inactive for the whole
rollout (asserted), arm geoms keep v1's no-contact setting EXCEPT the two
gripper-end collision geoms (fixed jaw on body "gripper", moving jaw on
body "moving_jaw_so101_v1"), whose contacts are enabled with the
parameter table's finger friction (1.0, 0.005, 0.0001)
(docs/physics-parameters.md: SO-101 gripper fingers/pads). No grasp
assistance of any kind.

Cameras: frontcam/topcam are the v1.1 scene cameras. The v1.1 scenes have
NO wrist camera; "wristcam" is added by this harness on the gripper body
(MjSpec) as an instrument addition, documented in the report.
Renders are 256x256 (base policy config declares [3,256,256] inputs).

Scoring: v1.1 replay.evaluate() imported read-only from the v1.1 workdir
(contacts_with criteria + secondary + cable solver detection), plus a
generalised solver-failure check (non-finite / out-of-workspace object
state => excluded, counted, never scored as success).

Records: appended incrementally to <out>/rollouts.jsonl (one JSON object
per completed rollout, fsync'd) + per-rollout trajectory .npz. Re-running
skips rollouts already recorded complete (kill-and-resume standard).
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

V1 = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
V2 = Path("/home/openclaw/muse-work/zk0-study001-twin-v2-20261122")
SCENES = V2 / "scenes-v2"
RECORDS = Path("/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/records")
FL_PY = "/home/openclaw/muse-work/zk0-study001-fl-20261121/venv/bin/python"
sys.path.insert(0, str(V1 / "code"))
import twin_params as P  # v1.1, read-only

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 20.0
CHUNK = 50
SETTLE_STEPS = int(FPS * 1.0)          # +1.0 s settle (pre-reg s.4)
STS3215_SPEED_RAD_S = 6.16             # 0.17 s/60 deg @ 7.4 V (datasheet)
# Gripper calibration (fixed global mapping, instrument parameter):
# dataset gripper.pos degrees: LOW=closed, HIGH=open (v1.1 calibration
# finding, replay.closedness_trace). v1.1 visual endpoints for the gripper
# joint: 0.10 rad closed, 1.45 rad open. Dataset global gripper range
# 0..52.3 deg (meta/stats.json max 52.3).
GRIP_DEG_MAX = 52.3
GRIP_RAD_CLOSED, GRIP_RAD_OPEN = 0.10, 1.45
FINGER_FRICTION = (1.0, 0.005, 0.0001)  # physics-parameters.md, fingers/pads
FINGER_BODIES = {"gripper", "moving_jaw_so101_v1"}

TASK_PROMPTS = {  # verbatim dataset task strings (meta/tasks.parquet)
    "block_stack": "Stack the colouful blocks on top of each other",
    "ring_insert": "Insert the colourful ring into the central wooden peg",
    "tool_insert": "Insert the missing tool into the empty slot on the toolbox",
    "eye_drops_to_basket": "Put the eye drops into the basket",
    "cable_clip": "Push the Displayport cable into the cable holder on the white block",
}

def grip_deg_to_rad(d):
    return GRIP_RAD_CLOSED + (float(d) / GRIP_DEG_MAX) * (GRIP_RAD_OPEN - GRIP_RAD_CLOSED)

def grip_rad_to_deg(r):
    return (float(r) - GRIP_RAD_CLOSED) / (GRIP_RAD_OPEN - GRIP_RAD_CLOSED) * GRIP_DEG_MAX

def action_deg_to_ctrl(a_deg):
    a = np.asarray(a_deg, float)
    ctrl = np.radians(a[:5]).tolist()
    ctrl.append(grip_deg_to_rad(a[5]))
    return np.array(ctrl)

def state_deg_from_qpos(qpos, qadr):
    deg = [math.degrees(qpos[qadr[j]]) for j in JOINTS[:5]]
    deg.append(grip_rad_to_deg(qpos[qadr["gripper"]]))
    return deg

def _lookat_quat(pos, target, up=(0.0, 0.0, 1.0)):
    pos = np.array(pos, float); target = np.array(target, float)
    z = pos - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2; w = 0.25 * s
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s
            q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s
            q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s
            q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return [w] + q

def build_actuated_model(scene_path):
    """Load a v1.1 variant scene; fix front cam; add dynamic wrist cam."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(scene_path))
    # v1.1 frontcam xyaxes renders near-black (verified by eye, Stage 1.4;
    # the MjSpec element keeps its xyaxes orientation type, so a quat
    # assignment on it is ignored at compile -- probed). Add a quat-typed
    # twin at the same position/fovy with a look-at orientation toward the
    # workspace centre and render "front" from it. Documented fix.
    fc = spec.camera("frontcam")
    spec.worldbody.add_camera(
        name="frontfix", pos=list(fc.pos),
        quat=_lookat_quat(list(fc.pos), [0.30, 0.0, 0.05]), fovy=float(fc.fovy))
    # Wrist camera: the v1.1 scenes have none. Added as a worldbody camera
    # whose pose is set per render from the gripperframe site (see
    # render_views): eye behind/above the wrist, looking past the fingers.
    spec.worldbody.add_camera(name="wristcam", pos=[0, 0, 1],
                              quat=[1, 0, 0, 0], fovy=60)
    model = spec.compile()
    return model

def prepare_model(model):
    """Force welds off; enable finger contacts only; return info dict."""
    import mujoco
    info = {"welds_forced_inactive": 0, "finger_geoms": []}
    # welds: scene equality welds ship active=false; keep them off.
    for eq in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
        if nm and nm.startswith("weld_"):
            info["welds_forced_inactive"] += 1
    finger_bids = set()
    for nm in FINGER_BODIES:
        finger_bids.add(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm))
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid in finger_bids and int(model.geom_group[gid]) == 3:
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 1
            model.geom_friction[gid] = FINGER_FRICTION
            info["finger_geoms"].append(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid))
        elif bid not in finger_bids:
            pass  # leave scene geoms untouched
    # Non-finger ARM geoms: v1 setting (no contacts). Bodies of the arm:
    arm_bodies = {"base", "shoulder", "upper_arm", "lower_arm", "wrist"}
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bnm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bnm in arm_bodies:
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0
    return info

def _set_wrist_cam(model, data, mujoco):
    """Pose the dynamic wrist camera: eye behind/above the gripper along
    the wrist->gripper axis, target past the gripperframe site (fingers)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wristcam")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    site = np.array(data.site_xpos[sid]); gp = np.array(data.xpos[gid])
    fwd = site - gp; fwd /= np.linalg.norm(fwd)
    eye = gp - fwd * 0.06 + np.array([0.0, 0.0, 0.05])
    tgt = site + fwd * 0.10
    z = eye - tgt; z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    data.cam_xpos[cid] = eye; data.cam_xmat[cid] = R.flatten()

class PolicyService:
    def __init__(self, ckpt_path):
        self.proc = subprocess.Popen(
            [FL_PY, str(Path(__file__).parent / "policy_service.py"), ckpt_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "POLICY_THREADS": "4", "CUDA_VISIBLE_DEVICES": ""})
        self.ready = None
        for _ in range(200):  # skip library warning lines on stdout
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
    ap.add_argument("--variant", default="nominal")  # nominal | 0..4
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--checkpoint-path", default="")
    ap.add_argument("--policy", choices=["service", "gt"], default="service",
                    help="gt = diagnostic: replay recorded actions through the actuated loop")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-frames", action="store_true")
    args = ap.parse_args()
    import mujoco
    import replay as v1replay  # v1.1 scoring, read-only

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

    scene_name = (f"{args.task}__ep{args.episode}.xml" if args.variant == "nominal"
                  else f"{args.task}__ep{args.episode}_v{args.variant}.xml")
    model = build_actuated_model(SCENES / scene_name)
    minfo = prepare_model(model)
    model.opt.timestep = P.TIMESTEP
    data = mujoco.MjData(model)
    qadr = {j: int(model.joint(j).qposadr[0]) for j in JOINTS}
    act_id = {j: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)) for j in JOINTS}

    gt = np.load(RECORDS / f"base__{args.task}__ep{args.episode}.npz")["gt"].astype(float)
    n_steps = len(gt) + SETTLE_STEPS
    # Variant definition actually consumed (frozen v1.1 draws):
    var_def = None
    vj = SCENES / f"{args.task}__ep{args.episode}_variants.json"
    if vj.exists() and args.variant != "nominal":
        vd = json.loads(vj.read_text())
        var_def = vd["variants"][int(args.variant)]

    # Init arm at the recorded first-frame pose (degrees -> ctrl), objects
    # rest under a 0.25 s held pre-settle (v1.1 standard).
    mujoco.mj_resetData(model, data)
    ctrl = action_deg_to_ctrl(gt[0])
    for j in JOINTS:
        data.qpos[qadr[j]] = ctrl[JOINTS.index(j)]
        data.ctrl[act_id[j]] = ctrl[JOINTS.index(j)]
    mujoco.mj_forward(model, data)
    steps_per_frame = int(round((1.0 / FPS) / P.TIMESTEP))
    for _ in range(int(0.25 / P.TIMESTEP)):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=256, width=256)
    frame_dir = out / "frames_tmp"; frame_dir.mkdir(exist_ok=True)
    svc = None
    if args.policy == "service":
        svc = PolicyService(args.checkpoint_path)
    prompt = TASK_PROMPTS[args.task]

    traj_qpos, traj_ctrl, traj_obj = [], [], {}
    obj_bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                  for i in range(model.nbody)
                  if model.body_jntnum[i] > 0 and
                  int(model.jnt_type[int(model.body_jntadr[i])]) == 0]  # free joints
    cur_ctrl = ctrl.copy()
    chunk_actions, chunk_idx = None, -1
    t0 = time.time()
    solver_bad = False
    for step in range(n_steps):
        if step < len(gt) and step % CHUNK == 0:
            chunk_idx += 1
            # render the three views at the chunk boundary
            frames = {}
            for cam, ds in (("frontfix", "front"), ("topcam", "top"), ("wristcam", "wrist")):
                if cam == "wristcam":
                    _set_wrist_cam(model, data, mujoco)
                renderer.update_scene(data, camera=cam)
                img = renderer.render()
                fp = frame_dir / f"{ds}.npy"
                np.save(fp, img)
                frames[ds] = str(fp)
                if args.save_frames and chunk_idx == 0:
                    from PIL import Image
                    Image.fromarray(img).save(out / f"frame0_{ds}.png")
            state_deg = state_deg_from_qpos(data.qpos, qadr)
            if args.policy == "gt":
                chunk_actions = gt[step:step + CHUNK]
            else:
                chunk_actions = svc.query(state_deg, prompt, args.episode, chunk_idx, frames)
        if step < len(gt):
            target = action_deg_to_ctrl(chunk_actions[min(step % CHUNK, len(chunk_actions) - 1)])
        else:
            target = cur_ctrl  # settle: hold last target
        # STS3215 slew limit on ctrl targets
        max_d = STS3215_SPEED_RAD_S / FPS
        cur_ctrl = cur_ctrl + np.clip(target - cur_ctrl, -max_d, max_d)
        for j in JOINTS:
            data.ctrl[act_id[j]] = cur_ctrl[JOINTS.index(j)]
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
            if model.neq > 0 and np.any(data.eq_active[:model.neq]):
                raise RuntimeError("weld equality became active during rollout")
        traj_qpos.append([float(data.qpos[qadr[j]]) for j in JOINTS])
        traj_ctrl.append(cur_ctrl.tolist())
        for nm in obj_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            traj_obj.setdefault(nm, []).append(np.array(data.xpos[bid]).tolist())
        if not np.all(np.isfinite(data.qpos)):
            solver_bad = True; break
    if svc: svc.close()
    renderer.close()

    mujoco.mj_forward(model, data)
    result = v1replay.evaluate(model, data, args.task, args.episode, mujoco)
    # generalised solver-failure standard (v1.1 cable standard, all tasks)
    gen_solver = bool(solver_bad)
    for nm, arr in traj_obj.items():
        a = np.array(arr[-1]) if arr else np.array([np.nan])
        if not np.all(np.isfinite(a)) or np.linalg.norm(a[:2]) > 1.0 or not (-0.05 <= a[2] <= 0.50):
            gen_solver = True
    record = {
        **key, "status": "complete", "suite": "twin-suite-v2-dev",
        "policy_mode": args.policy, "prompt": prompt,
        "horizon_steps": n_steps, "fps": FPS,
        "variant_draws": (var_def or {}).get("draws") if var_def else "nominal",
        "sim_success": bool(result["sim_success"]) and not gen_solver,
        "secondary_pass": result.get("secondary_pass"),
        "solver_failure": bool(result.get("solver_failure") or gen_solver),
        "excluded": bool(result.get("solver_failure") or gen_solver),
        "checks": result.get("checks"),
        "objects_final": result.get("objects_final"),
        "actuator_info": minfo,
        "wall_s": round(time.time() - t0, 2),
    }
    np.savez_compressed(out / f"traj_{args.task}__ep{args.episode}_{args.variant}.npz",
                        qpos=np.array(traj_qpos), ctrl=np.array(traj_ctrl),
                        **{f"obj_{k}": np.array(v) for k, v in traj_obj.items()})
    with open(rec_path, "a") as f:
        f.write(json.dumps(record) + "\n"); f.flush(); os.fsync(f.fileno())
    print(json.dumps({k: record[k] for k in
          ("task", "episode", "variant", "sim_success", "secondary_pass",
           "solver_failure", "wall_s")}, default=str))

if __name__ == "__main__":
    main()
