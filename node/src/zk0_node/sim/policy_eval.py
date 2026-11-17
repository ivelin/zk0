"""Stage 5 checkpoint eval: the shared checkpoint as policy under test.

Build slice 1 measured the harness with a scripted oracle pilot. This
module replaces the pilot with the actual shared checkpoint
(`ivelin/zk0-smolvla-fl`, the LeRobot 0.7.0-repaired revision) and
runs the preregistration's benchmark suites (SIM-1 cube -> box,
SIM-2 mug -> bin) at >=20 trials per task with per-trial variation
draws. The verdicts are computed from state logs under the same
Tier 1 definition as every other zk0 eval (object in target +
nothing damaged + legal actuation; preregistration section 2).

How the policy is driven (all of it recorded in the eval record):
- Observations are the checkpoint's own input contract: joint state
  in degrees (LeRobot SO-101 frame) and the two camera images it was
  trained with -- `observation.images.front` and
  `observation.images.top`, rendered in-sim from declared poses at
  320x240. The joint frames are assumed aligned (LeRobot calibrated
  zero == model zero, the SO-101 calibration design).
- Actions come back in degrees, pass through the package's hard
  safety clamp (spec section 4.1: out-of-range commands are clamped
  and counted), and are applied to the position actuators.
- Control runs at ~29.4 Hz (17 physics steps of 2 ms per policy
  tick), approximating the 30 fps LeRobot capture cadence the
  checkpoint's training data was recorded at.

RUNG LABEL -- read before quoting any number from here: the policy
consumes RAW rendered frames and RAW joint state. No quantization,
no latency/hold, no noise is applied to its inputs, so by the
telemetry contract's ablation ladder (contract section 6) these are
RUNG 0 measurements: the most favorable sensing conditions, valid
as an eval measurement of the checkpoint, NON-CONFORMING for any
training-data-generation claim, and not the rung-3 conditions the
preregistration's section 4.1 bar was registered under. Every
record this module writes says so in its own fields.
"""
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from ..config import load_config, zk0_home
from ..identity import load_or_create_key
from ..records import (jsonable, manifest_sha256, runtime_manifest_path,
                       utcnow_iso, write_record, write_signed_record)
from . import assets, mugscene

HARNESS_DIR = Path(__file__).parent / "harness"
DAMAGE_BOUND_N = 25.0  # PROVISIONAL — preregistration section 2

CHECKPOINT_REPO = "ivelin/zk0-smolvla-fl"
CHECKPOINT_REVISION = "d1c3198fbdb23ea24c9f6751992986af328ce4eb"

TASK_TEXT = {
    "sim1": "Pick up the green cube and place it in the box.",
    "sim2": "Pick up the mug and place it in the bin.",
}
TASK_BODY = {"sim1": "cube_green", "sim2": "mug"}
# Nominal object spawn (the certified scenes' positions).
NOMINAL_POS = {"sim1": (0.26, 0.02), "sim2": (0.24, 0.135)}
BOX1 = (0.27, -0.14, 0.06, 0.06, 0.07)   # SIM-1 box: cx, cy, hx, hy, rim

TS = 0.002
STEPS_PER_TICK = 17                     # ~29.4 Hz policy cadence
HORIZON_S = 40.0
GATE_EVERY_STEPS = 5                    # 100 Hz gate sampling
LOG_EVERY_STEPS = 25                    # 20 Hz state log
FRONT_CAM_POS = (0.55, -0.45, 0.35)
FRONT_CAM_LOOKAT = (0.22, 0.0, 0.03)
TOP_CAM_POS = (0.22, 0.0, 0.75)
IMG_H, IMG_W = 240, 320
LEGAL_EXCURSION_TOL_DEG = 1.0
LEGAL_CLAMP_FRAC_MAX = 0.05


def _import_mt():
    """Import the vendored multitask module (it reads its model dir
    from the environment at import time; the default here is only
    ever used by its scene-string helpers, never for execution)."""
    if str(HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(HARNESS_DIR))
    os.environ.setdefault("ZK0_SIM_MODEL_DIR",
                          str(assets.model_dir("realistic")))
    import multitask as MT
    return MT

def checkpoint_dir() -> Path:
    return zk0_home() / "checkpoint" / "ivelin--zk0-smolvla-fl"


def checkpoint_info(ckpt_dir=None) -> dict:
    d = Path(ckpt_dir) if ckpt_dir else checkpoint_dir()
    info = {"repo": CHECKPOINT_REPO, "revision": CHECKPOINT_REVISION,
            "staged_dir": str(d), "present": d.exists()}
    ms = d / "model.safetensors"
    if ms.exists():
        info["model_safetensors_sha256"] = assets.sha256_of(ms)
    return info


# ---------------------------------------------------------------- policy

class PolicyBundle:
    def __init__(self, policy, pre, post, device):
        self.policy = policy
        self.pre = pre
        self.post = post
        self.device = device


_BUNDLE = None


def load_policy(ckpt_dir=None) -> PolicyBundle:
    """Load the shared checkpoint once per process (GPU if present)."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    ckpt = Path(ckpt_dir) if ckpt_dir else checkpoint_dir()
    if not ckpt.exists():
        raise FileNotFoundError(
            f"shared checkpoint not staged at {ckpt} — run dev/bootstrap.sh")
    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    pre, post = make_pre_post_processors(policy.config,
                                         pretrained_path=str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    policy.eval()
    _BUNDLE = PolicyBundle(policy, pre, post, device)
    return _BUNDLE


def shared_stats(ckpt_dir=None):
    """(action mean, action std) from the checkpoint's own normalizer
    — the protocol's shared action-normalization statistics."""
    from safetensors.torch import load_file
    ckpt = Path(ckpt_dir) if ckpt_dir else checkpoint_dir()
    f = ckpt / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    s = load_file(str(f))
    return (s["action.mean"].numpy().astype(float),
            s["action.std"].numpy().astype(float))


def shared_stats_hash(ckpt_dir=None) -> str:
    ckpt = Path(ckpt_dir) if ckpt_dir else checkpoint_dir()
    f = ckpt / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    return assets.sha256_of(f)


# ---------------------------------------------------------------- scenes

def _scene_xml(task_id, obj_xy, yaw_rad, mu):
    MT = _import_mt()  # vendored certified scenes

    if task_id == "sim1":
        x, y = obj_xy
        qw, qz = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
        inner = f"""
    <body name="cube_green" pos="{x} {y} 0.0205" quat="{qw:.6f} 0 0 {qz:.6f}">
      <freejoint name="cube_green_free"/>
      <inertial pos="0 0 0" mass="0.064" diaginertia="1.7067e-5 1.7067e-5 1.7067e-5"/>
      <geom name="cube_green_geom" type="box" size="0.02 0.02 0.02" friction="1 0.005 0.0001" rgba="0.1 0.8 0.2 1"/>
    </body>""" + MT.walls_xml("box", *BOX1[:2], BOX1[2], BOX1[3], 0.005,
                              BOX1[4], "0.75 0.75 0.78 1")
        asset_xml = ""
    elif task_id == "sim2":
        asset_xml = mugscene.mug_assets_xml()
        inner = mugscene.mug_body_xml(obj_xy, yaw_rad, mu) + MT.walls_xml(
            "bin", mugscene.BIN[0], mugscene.BIN[1], mugscene.BIN[2],
            mugscene.BIN[3], 0.005, mugscene.BIN[4], "0.62 0.66 0.70 1")
    else:
        raise ValueError(f"unknown task {task_id!r}")
    return f"""<mujoco model="policy_{task_id}">
  <include file="so101.xml"/>
  <asset>
    {asset_xml}
  </asset>
  <worldbody>
    <light pos="0.2 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" friction="1 0.005 0.0001" rgba="0.25 0.28 0.33 1"/>
    {inner}
    <camera name="front" pos="{' '.join(str(v) for v in FRONT_CAM_POS)}"/>
    <camera name="top" pos="{' '.join(str(v) for v in TOP_CAM_POS)}"/>
    <camera name="main" pos="0.62 -0.52 0.45"/>
  </worldbody>
</mujoco>"""


def _patch_gripper_friction(model_dir: Path, mu: float) -> dict:
    """Set the staged arm copy's gripper friction to mu (SIM-2 pair
    draws). Only the two gripper default-class geoms carry the 0.35
    value as an attribute; the count is asserted so a silent partial
    patch is impossible."""
    p = model_dir / "so101.xml"
    # Restore from the pristine copy first: trials share one work
    # model, so a previous trial's patch must never leak into this
    # one (a 0.12 trial followed by a 0.35 draw would otherwise run
    # at 0.12 silently — or crash the needle count).
    pristine = model_dir / "so101.pristine.xml"
    text = (pristine if pristine.exists() else p).read_text()
    needle = 'friction="0.35 0.005 0.0001"'
    n = text.count(needle)
    if n != 2:
        raise RuntimeError(f"gripper friction patch: expected 2 "
                           f"occurrences of {needle} in the pristine "
                           f"arm model, found {n}")
    if mu == 0.35:
        if p.read_text() != text:
            p.write_text(text)
        return {"patched": False, "occurrences": n}
    p.write_text(text.replace(needle, f'friction="{mu} 0.005 0.0001"'))
    return {"patched": True, "occurrences": n, "mu": mu}


# ---------------------------------------------------------------- episode

def _img_tensor(renderer, m, d, cam_name):
    renderer.update_scene(d, camera=cam_name)
    img = renderer.render()
    return np.ascontiguousarray(img)


def run_episode(task_id, draw, bundle: PolicyBundle, work_model: Path,
                log_out: Path | None) -> dict:
    """One policy episode. Returns the trial result; optionally writes
    the trial's state log (gate rows + 20 Hz states) to log_out."""
    import mujoco
    import torch

    if str(HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(HARNESS_DIR))
    from grasp_gate import GateRecorder, evaluate_cycle
    MT = _import_mt()

    body = TASK_BODY[task_id]
    mu = draw.get("mu", 0.35)
    friction_patch = {"patched": False}
    if task_id == "sim2":
        friction_patch = _patch_gripper_friction(work_model, mu)
    xml = _scene_xml(task_id, (draw["x"], draw["y"]), draw["yaw_rad"], mu)
    scene_path = work_model / f"scene_policy_{task_id}.xml"
    scene_path.write_text(xml)

    m = mujoco.MjModel.from_xml_path(str(scene_path))
    m.opt.timestep = TS
    MT.aim_camera(m, m.cam("front").id, np.array(FRONT_CAM_POS),
                  np.array(FRONT_CAM_LOOKAT))
    site_id = m.site("gripperframe").id
    qadr = [m.jnt(n).qposadr[0] for n in MT.JOINTS]
    jnt_range_deg = np.degrees(np.array([m.jnt(n).range for n in MT.JOINTS]))
    gate = GateRecorder(m, [body], site_id)
    d = mujoco.MjData(m)
    # Start pose: model zero with the gripper open (the sim's canonical
    # home — the same start every certified sim run uses).
    d.qpos[qadr[5]] = MT.GRIP_OPEN
    mujoco.mj_forward(m, d)

    r_front = mujoco.Renderer(m, height=IMG_H, width=IMG_W)
    r_top = mujoco.Renderer(m, height=IMG_H, width=IMG_W)

    if hasattr(bundle.policy, "reset"):
        bundle.policy.reset()

    log = {"t": [], "tcp": [], "q": [],
           "bodies": {body: []}, "quats": {body: []}}
    bqadr = gate.bqadr[body]
    max_ticks = int(HORIZON_S / (TS * STEPS_PER_TICK)) + 1
    ctrl = np.zeros(6)
    ctrl[5] = MT.GRIP_OPEN
    n_queries = 0
    clamp_events = 0
    oob_events = 0
    max_excursion_deg = 0.0
    termination = "horizon"
    contained_ticks = 0
    step = 0
    for tick in range(max_ticks):
        mujoco.mj_forward(m, d)
        state_deg = np.degrees(d.qpos[qadr]).astype(np.float32)
        obs = {
            "observation.state": torch.from_numpy(state_deg),
            "observation.images.front": torch.from_numpy(
                _img_tensor(r_front, m, d, "front")
                .transpose(2, 0, 1).astype(np.float32) / 255.0),
            "observation.images.top": torch.from_numpy(
                _img_tensor(r_top, m, d, "top")
                .transpose(2, 0, 1).astype(np.float32) / 255.0),
            "task": TASK_TEXT[task_id],
        }
        with torch.no_grad():
            action = bundle.post(
                bundle.policy.select_action(bundle.pre(obs)))
        action_deg = action.detach().cpu().numpy().reshape(-1).astype(float)
        n_queries += 1
        # Hard safety layer (spec section 4.1): clamp + count.
        exc = np.maximum(jnt_range_deg[:, 0] - action_deg,
                         action_deg - jnt_range_deg[:, 1])
        max_excursion_deg = max(max_excursion_deg, float(exc.max()))
        if (exc > 0).any():
            clamp_events += 1
        if (exc > LEGAL_EXCURSION_TOL_DEG).any():
            oob_events += 1
        applied_deg = np.clip(action_deg, jnt_range_deg[:, 0],
                              jnt_range_deg[:, 1])
        ctrl = np.radians(applied_deg)
        for _ in range(STEPS_PER_TICK):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
            step += 1
            if step % GATE_EVERY_STEPS == 0:
                mujoco.mj_forward(m, d)
                gate.sample(d, ctrl)
            if step % LOG_EVERY_STEPS == 0:
                mujoco.mj_forward(m, d)
                log["t"].append(d.time)
                log["tcp"].append(d.site_xpos[site_id].copy())
                log["q"].append(d.qpos[qadr].copy())
                log["bodies"][body].append(d.qpos[bqadr:bqadr + 3].copy())
                log["quats"][body].append(d.qpos[bqadr + 3:bqadr + 7].copy())
        # Early exits, judged from state (never from a success flag).
        pos = d.qpos[bqadr:bqadr + 3]
        if abs(pos[0]) > 0.6 or abs(pos[1]) > 0.6 or pos[2] < -0.02:
            termination = "object_lost"
            break
        if _contained_now(task_id, pos) and d.qpos[qadr[5]] > 1.5 \
                and d.site_xpos[site_id][2] > 0.10:
            contained_ticks += 1
            if contained_ticks >= 15:
                termination = "contained_and_released"
                break
        else:
            contained_ticks = 0

    for b in log["bodies"]:
        log["bodies"][b] = np.array(log["bodies"][b])
        log["quats"][b] = np.array(log["quats"][b])
    log["t"] = np.array(log["t"])
    log["tcp"] = np.array(log["tcp"])
    log["q"] = np.array(log["q"])

    rows = gate.rows
    max_single = max((r[body]["max_single"] for r in rows), default=0.0)
    settled_mm = float(MT.settled(log, body)) if len(log["t"]) else 999.0
    final_pos = log["bodies"][body][-1] if len(log["t"]) else \
        np.array(d.qpos[bqadr:bqadr + 3])
    containment = _containment(task_id, final_pos, settled_mm)
    damage_pass = bool(max_single < DAMAGE_BOUND_N)
    clamp_frac = clamp_events / max(n_queries, 1)
    legal_pass = bool(oob_events == 0 and clamp_frac <= LEGAL_CLAMP_FRAC_MAX)
    tier1_pass = bool(containment["pass"] and damage_pass and legal_pass)

    tier2 = _tier2(rows, body, gate)
    result = {
        "draw": {k: (round(float(v), 5) if isinstance(v, float) else v)
                 for k, v in draw.items()},
        "termination": termination,
        "duration_s": round(float(log["t"][-1]) if len(log["t"]) else 0.0, 2),
        "policy_queries": n_queries,
        "final_object_pos": [round(float(v), 4) for v in final_pos],
        "settled_mm": round(settled_mm, 2),
        "tier1": {
            "pass": tier1_pass,
            "containment": containment,
            "damage": {"max_single_contact_N_whole_run":
                       round(float(max_single), 2),
                       "bound_N": DAMAGE_BOUND_N,
                       "bound_status": "PROVISIONAL",
                       "pass": damage_pass},
            "legal_actuation": {
                "pass": legal_pass,
                "raw_out_of_range_ticks": oob_events,
                "clamp_ticks": clamp_events,
                "clamp_frac": round(clamp_frac, 4),
                "max_excursion_deg": round(max_excursion_deg, 3),
                "rule": f"requested actions must stay within the joint "
                        f"ranges (>{LEGAL_EXCURSION_TOL_DEG} deg excursion "
                        f"fails) and clamped ticks must stay "
                        f"<= {LEGAL_CLAMP_FRAC_MAX:.0%} of queries; applied "
                        f"commands are always clamped by the safety layer",
            },
        },
        "tier2": tier2,
        "gripper_friction_patch": friction_patch,
    }
    if log_out is not None:
        state_log = {
            "log_type": "sim_state_log/0.2",
            "task": task_id.upper().replace("SIM", "SIM-"),
            "policy": "shared checkpoint (see suite eval record)",
            "events": {"termination": termination},
            "t": log["t"], "tcp": log["tcp"], "q": log["q"],
            "bodies": log["bodies"], "quats": log["quats"],
            "gate_rows": rows,
        }
        write_record(state_log, Path(log_out))
        result["state_log_sha256"] = assets.sha256_of(Path(log_out))
    return result


def _contained_now(task_id, pos) -> bool:
    if task_id == "sim1":
        cx, cy, hx, hy, rim = BOX1
        return abs(pos[0] - cx) < hx and abs(pos[1] - cy) < hy \
            and pos[2] + 0.02 < rim
    c = mugscene.contained(pos, 0.0)
    return c["inside_xy"] and c["below_rim"]


def _containment(task_id, final_pos, settled_mm) -> dict:
    if task_id == "sim1":
        cx, cy, hx, hy, rim = BOX1
        x, y, z = (float(v) for v in final_pos)
        inside = abs(x - cx) < hx and abs(y - cy) < hy
        below = z + 0.02 < rim
        settled_ok = settled_mm <= 2.0
        return {"inside_xy": bool(inside), "below_rim": bool(below),
                "settled": bool(settled_ok),
                "pass": bool(inside and below and settled_ok)}
    return mugscene.contained(final_pos, settled_mm)


def _tier2(rows, body, gate) -> dict:
    """Tier 2 (PROVISIONAL) from a detected grasp cycle, if the policy
    produced one: first sustained gripper closure, then re-opening.
    No detected cycle -> Tier 2 is not evaluable for the trial (the
    policy never grasped); that is recorded, not smoothed over."""
    from grasp_gate import evaluate_cycle

    close_end = None
    open_end = None
    for r in rows:
        if close_end is None and r["grip_q"] < 1.0:
            close_end = r["t"]
        elif close_end is not None and r["grip_q"] > 1.5:
            open_end = r["t"]
            break
    if close_end is None or open_end is None:
        return {"status": "PROVISIONAL (SSOT decision 53)",
                "evaluable": False,
                "reason": "no complete grasp cycle detected (gripper "
                          "never closed below 1.0 rad and re-opened)"}
    window = [r for r in rows if close_end <= r["t"] <= open_end]
    close_cmd = min(r["grip_cmd"] for r in window)
    g = evaluate_cycle(rows, body, close_end, open_end, close_cmd,
                       gate.mass[body], gate.jnt_range, gate.force_limit)
    return {"status": "PROVISIONAL (SSOT decision 53 — not "
                      "human-validated; does not gate anything)",
            "evaluable": True,
            "gate_pass": bool(g["pass"]),
            "failed_gates": [k for k, v in g["gates"].items()
                             if not v["pass"]],
            "gates": jsonable(g["gates"]),
            "window_s": g["window_s"]}


# ---------------------------------------------------------------- suite

def _wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4),
            round(min(1.0, centre + half), 4)]


def run_suite(task_id, trials=20, seed=20261117, outdir=None, cfg=None,
              ckpt_dir=None) -> dict:
    """Run one benchmark suite (Stage 5). Returns the signed record."""
    if task_id not in TASK_TEXT:
        raise ValueError(f"unknown task {task_id!r} ({sorted(TASK_TEXT)})")
    cfg = cfg or load_config()
    bundle = load_policy(ckpt_dir)
    ckpt = checkpoint_info(ckpt_dir)

    stamp = utcnow_iso().replace(":", "").replace("-", "")
    outdir = Path(outdir) if outdir else (
        zk0_home() / "records" / f"stage5-{task_id}-{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Private copy of the staged arm model (+ mug meshes for SIM-2):
    # scenes are written into it; the asset cache stays clean.
    work_model = outdir / "model"
    shutil.copytree(assets.model_dir("realistic"), work_model)
    # Pristine arm-model copy for the per-trial friction patch.
    (work_model / "so101.pristine.xml").write_text(
        (work_model / "so101.xml").read_text())
    if task_id == "sim2":
        # The arm model compiles with meshdir="assets" — the mug
        # meshes must land in the model copy's assets/ subdir.
        mug_dest = work_model / "assets"
        mug_dest.mkdir(exist_ok=True)
        for f in assets.mug_dir().iterdir():
            if f.is_file():
                # Meshes resolve via the arm model's meshdir="assets";
                # the texture resolves from the model root. Copy to
                # both so every reference resolves.
                shutil.copy2(f, mug_dest / f.name)
                shutil.copy2(f, work_model / f.name)

    rng = np.random.default_rng(seed)
    mus = None
    if task_id == "sim2":
        mus = [0.35 if i % 2 == 0 else 0.12 for i in range(trials)]
        rng.shuffle(mus)
    nx, ny = NOMINAL_POS[task_id]
    results = []
    for i in range(trials):
        draw = {"trial": i, "seed": int(seed),
                "x": float(nx + rng.uniform(-0.01, 0.01)),
                "y": float(ny + rng.uniform(-0.01, 0.01)),
                "yaw_rad": float(rng.uniform(0, 2 * math.pi))
                if task_id == "sim1" else float(rng.uniform(-math.pi, math.pi))}
        if mus is not None:
            draw["mu"] = float(mus[i])
        log_path = outdir / f"trial_{i:02d}_state_log.json"
        res = run_episode(task_id, draw, bundle, work_model, log_path)
        res["trial"] = i
        results.append(res)
        print(f"[{task_id} trial {i + 1}/{trials}] tier1="
              f"{'PASS' if res['tier1']['pass'] else 'FAIL'} "
              f"({res['termination']}, {res['duration_s']} s)")

    k = sum(1 for r in results if r["tier1"]["pass"])
    record = {
        "record_type": "checkpoint_eval_suite",
        "schema": "zk0-node-eval/0.2",
        "created_utc": utcnow_iso(),
        "stage": "Stage 5 — checkpoint eval in sim",
        "node": {"node_id": cfg["node_id"], "package": "zk0 Node",
                 "package_version": "0.1.0",
                 "runtime_manifest_sha256": manifest_sha256(
                     runtime_manifest_path())},
        "mode": "sim",
        "checkpoint": ckpt,
        "task": {
            "id": task_id.upper().replace("SIM", "SIM-"),
            "text_given_to_policy": TASK_TEXT[task_id],
            "suite": "zk0 Node Stage 5 suite (preregistration section 2 "
                     "task set; variation draws per trial: object "
                     "position +/-10 mm, yaw"
                     + (", pair friction mu in {0.35, 0.12}" if task_id == "sim2" else "")
                     + ")",
            "trials": trials,
            "seed": int(seed),
        },
        "policy": {
            "kind": "learned_policy",
            "id": CHECKPOINT_REPO,
            "device": bundle.device,
            "control_rate_hz": round(1.0 / (TS * STEPS_PER_TICK), 1),
            "start_pose": "model zero, gripper open (the sim's canonical "
                          "home)",
            "cameras": {
                "front": {"pos": list(FRONT_CAM_POS),
                          "lookat": list(FRONT_CAM_LOOKAT),
                          "resolution": [IMG_W, IMG_H]},
                "top": {"pos": list(TOP_CAM_POS),
                        "note": "identity orientation — looks straight "
                                "down the -Z axis",
                        "resolution": [IMG_W, IMG_H]},
            },
        },
        "ablation_rung": 0,
        "rung_note": "RUNG 0: the policy consumes raw rendered frames "
                     "and raw joint state — no quantization, latency, "
                     "or noise applied. Valid as an eval measurement "
                     "of the checkpoint under the most favorable "
                     "sensing conditions; NON-CONFORMING for "
                     "training-data generation (contract section 6), "
                     "and not the rung-3 conditions the preregistration "
                     "section 4.1 bar was registered under.",
        "marks": {
            "provenance": "n/a — eval measurement of the shared "
                          "checkpoint in sim; nothing trained, nothing "
                          "contributed",
            "generation": "n/a — the policy under test is a learned "
                          "checkpoint, not a generated trajectory",
            "envelope": "n/a — trajectory admission is Stage 7 work",
        },
        "results": {
            "tier1": {"successes": k, "trials": trials,
                      "rate": round(k / trials, 4),
                      "ci95_wilson": _wilson(k, trials)},
            "tier2": {"status": "PROVISIONAL (SSOT decision 53)",
                      "evaluable_trials": sum(
                          1 for r in results if r["tier2"].get("evaluable")),
                      "gate_pass_trials": sum(
                          1 for r in results
                          if r["tier2"].get("gate_pass"))},
            "bar_context": "preregistration section 4.1 sim bar: "
                           "Tier 1 >= 60% over >=20 trials AT RUNG 3 "
                           "(PROPOSED-GUESS, signed off 2026-11-17). "
                           "This suite ran at rung 0 — the comparison "
                           "is indicative, not the registered test.",
            "per_trial": results,
        },
    }
    key = load_or_create_key()
    write_signed_record(record, outdir / "eval_record.json", key)
    record["outdir"] = str(outdir)
    return record
