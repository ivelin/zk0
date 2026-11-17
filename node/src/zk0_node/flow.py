"""Stages 7-11 of the task flow (spec section 3) + the full-loop
dry run. Build slice 4.

Stage 7 — trajectory admission (SSOT decision 54 as code): a
candidate trajectory is admitted as training data only if the
CONTROLLER that produced it completes the task (Tier 1) at every
draw of the declared physical-variance envelope (decision-52
fields: friction by coating, mass, servo class, fragility as the
damage bound). Admission tests the controller, not the file — the
recorded episode is the controller's nominal-condition execution,
and the inferability caveat of spec section 5.1 applies: a consumer
of the admitted dataset gets the trajectory plus its envelope
marks, never a claim that the file itself was re-tested.

Stage 8 — local fine-tune benchmark: the shared checkpoint is
fine-tuned LOCALLY on the admitted dataset (action expert only,
the federation's trainable subset, through the FL client's own
fit path so the local procedure and the round procedure are the
same code), then measured on the Stage 5 suite machinery.

Stages 9-11 — contribution through a local-network round (spec
section 2.3), paired comparison, and the win/no-win declaration
generated FROM the records per preregistration section 4.2. The
declaration text is machine-generated from the numbers; nothing
in it is hand-written.

Marks discipline (spec section 5.5): provenance, generation and
envelope are three independent axes, recorded side by side on
every record this module issues. They are never merged into a
single composite mark.

PREREGISTRATION STATUS (carried on every record this module
issues): the benchmark preregistration was founder-signed
2026-11-17, but its section 5 mis-scope condition was triggered
in substance for the base checkpoint by the slice-2 measurement
(0/20 and 0/20 at rung 0; verifier ruling: the runner is faithful,
so the result stands as a policy measurement). The founder's
re-scope / re-registration decision is PENDING. This slice builds
the machinery and runs it on a NEW sim-trained policy derived
from that checkpoint; it makes no registered benchmark claim for
the base checkpoint.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config, zk0_home
from .identity import load_or_create_key
from .records import (manifest_sha256, runtime_manifest_path,
                      utcnow_iso, verify_record, write_signed_record)
from .sim import assets

PREREG_STATUS = {
    "registration": "founder-signed 2026-11-17 (benchmark "
                    "preregistration, zk0 Node package design)",
    "section_5_status": "MIS-SCOPE CONDITION TRIGGERED IN SUBSTANCE "
                        "for the base checkpoint ivelin/zk0-smolvla-fl: "
                        "slice-2 Stage 5 measurement 0/20 (SIM-1) and "
                        "0/20 (SIM-2) Tier 1 at rung 0, runner ruled "
                        "faithful by the independent verifier",
    "founder_decision": "PENDING — re-scope / re-registration of the "
                        "benchmark set is the founder's decision, "
                        "not taken as of this record",
    "this_run": "machinery exercised on a NEW sim-trained policy "
                "derived from the base checkpoint; no registered "
                "benchmark claim is made for the base checkpoint",
}

PARITY_TOLERANCE = 0.05     # preregistration section 4.2: -5pp
DAMAGE_BOUND_N = 25.0       # PROVISIONAL — preregistration section 2

# The Stage 8 fine-tune budget (declared before the run; recorded
# in the fine-tune record and used identically for the Stage 9
# round's client config, so the local benchmark and the
# contribution are the same training procedure).
FINETUNE_BUDGET = {"steps": 400, "lr": 5e-5, "batch_size": 4,
                   "proximal_mu": 0.1,
                   "aggregator": "fedprox + dynamic weighting "
                                 "(the incumbent, spec section 5.3)"}

ADMISSION_SEED = 20261117


def marks_block(provenance: str, generation: str, envelope) -> dict:
    """The three independent mark axes (spec section 5.5), side by
    side, never merged."""
    return {"provenance": provenance, "generation": generation,
            "envelope": envelope}


# ------------------------------------------------------------- envelope

def envelope_declaration() -> dict:
    """The declared variance envelope (decision-52 fields), fixed
    BEFORE any admission run. Draws sample corners + interior,
    never nominal alone."""
    return {
        "envelope_id": "zk0-admission-envelope/0.1",
        "declared_before_runs": True,
        "fields": {
            "friction_pair_mu": {
                "levels": [0.12, 0.35],
                "tag": "DECLARED — the Stage 5 suite's pair-draw set "
                       "(decision-52 friction-by-coating axis; 0.12 "
                       "is the low-grip coating class, 0.35 nominal)",
                "applied_as": "both sides of the contact pair set "
                              "to the drawn mu",
            },
            "mass_kg": {
                "sim1": {"nominal": 0.064,
                         "factors": [0.85, 1.0, 1.15],
                         "tag": "nominal DECLARED (certified scene "
                                "value); factors are this envelope's "
                                "declared class spread"},
                "sim2": {"nominal": 0.118,
                         "factors": [0.9, 1.0, 1.1],
                         "tag": "nominal MEASURED (ycb-025-mug "
                                "params.json); factors are this "
                                "envelope's declared class spread"},
            },
            "servo_class": {
                "sim1": "variant position servo (the package's "
                        "default profile)",
                "sim2": "BAM-fitted STS3215 (M6 model, LeRobot "
                        "firmware P=16, 7.4 V) — the only actuation "
                        "the certified hardened law consumes (its "
                        "duty signal is a BAM actuator register)",
                "draws": "none — decision 52's per-unit spread slot "
                         "carries no values in the certified "
                         "implementation, so there is nothing "
                         "certified to sample; the class itself is "
                         "fixed per task and stated here",
            },
            "fragility": "not sampled — enters through the Tier 1 "
                         "damage bound (max single contact < "
                         "25 N, PROVISIONAL, preregistration "
                         "section 2)",
        },
        "draw_rule": "4 corners (mu x mass-factor extremes) + 1 "
                     "centre (mu midpoint, nominal mass) + 2 seeded "
                     "interior draws = 7 draws per candidate; "
                     "admission requires Tier 1 at EVERY draw "
                     "(decision 54)",
        "draw_seed": ADMISSION_SEED,
    }


def envelope_draws(task_id: str, seed: int = ADMISSION_SEED) -> list:
    """The 7 envelope draws for one task (deterministic in seed)."""
    mass = {"sim1": (0.064, 0.85, 1.15),
            "sim2": (0.118, 0.9, 1.1)}[task_id]
    nominal, lo_f, hi_f = mass
    rng = np.random.default_rng(seed)
    draws = [
        {"draw_id": "corner-mu_lo-mass_lo", "mu": 0.12,
         "mass_kg": round(nominal * lo_f, 5)},
        {"draw_id": "corner-mu_lo-mass_hi", "mu": 0.12,
         "mass_kg": round(nominal * hi_f, 5)},
        {"draw_id": "corner-mu_hi-mass_lo", "mu": 0.35,
         "mass_kg": round(nominal * lo_f, 5)},
        {"draw_id": "corner-mu_hi-mass_hi", "mu": 0.35,
         "mass_kg": round(nominal * hi_f, 5)},
        {"draw_id": "centre", "mu": 0.235,
         "mass_kg": round(nominal * 1.0, 5)},
    ]
    for i in (1, 2):
        draws.append({
            "draw_id": f"interior-{i}",
            "mu": round(float(rng.uniform(0.12, 0.35)), 4),
            "mass_kg": round(nominal * float(rng.uniform(lo_f, hi_f)), 5)})
    return draws


# ------------------------------------------------------------- SIM-1 runs

def _sim1_plan(MT, obj_xy, place_xy):
    """The certified scripted SIM-1 plan (multitask pick_place with
    the task1 constants demodata uses), planned on the multitask
    model; the joint-space segments transfer to the policy scene."""
    P = MT.Pilot("task1", bodies=["cube_green"])
    MT.pick_place(P, obj_xy, 0.02, -0.01, MT.GRIP_CLOSE_CUBE, 0.135,
                  place_xy, 0.028, tag="t1", body="cube_green")
    P.home()
    P.dwell(1.2)
    return P


def _sim1_execute(PE, MT, segs, t_total, obj_xy, mu, mass_kg,
                  work_model, record=False):
    """Execute a SIM-1 plan in the Stage 5 policy scene under one
    envelope draw; score Tier 1 with the package's definitions.

    Physics/cadence mirror policy_eval.run_episode (TS 0.002,
    100 Hz gate sampling, 20 Hz state log). The recording variant
    runs at 480 Hz for an exact 30 fps capture (demodata's
    established recording discipline) and renders front/top.
    Returns (tier1_dict, episode_dict_or_None)."""
    import mujoco
    from grasp_gate import GateRecorder

    xml = PE._scene_xml("sim1", obj_xy, 0.0, mu)
    ratio = mass_kg / 0.064
    xml = xml.replace('mass="0.064"', f'mass="{mass_kg}"')
    xml = xml.replace('diaginertia="1.7067e-5 1.7067e-5 1.7067e-5"',
                      'diaginertia="{} {} {}"'.format(
                          *[f"{1.7067e-5 * ratio:.6g}"] * 3))
    xml = xml.replace(
        '<geom name="cube_green_geom" type="box" size="0.02 0.02 0.02" '
        'friction="1 0.005 0.0001"',
        '<geom name="cube_green_geom" type="box" size="0.02 0.02 0.02" '
        f'friction="{mu} 0.005 0.0001"')
    PE._patch_gripper_friction(work_model, mu)
    scene_path = work_model / "scene_admission_sim1.xml"
    scene_path.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(scene_path))
    body = "cube_green"
    site_id = m.site("gripperframe").id
    qadr = [m.jnt(n).qposadr[0] for n in MT.JOINTS]
    jnt_range = np.array([m.jnt(n).range for n in MT.JOINTS])
    gate = GateRecorder(m, [body], site_id)
    d = mujoco.MjData(m)
    d.qpos[qadr[5]] = MT.GRIP_OPEN
    mujoco.mj_forward(m, d)

    if record:
        ts = 1.0 / 480.0
        gate_every, log_every, frame_every = 5, 24, 16
        MT.aim_camera(m, m.cam("front").id, np.array(PE.FRONT_CAM_POS),
                      np.array(PE.FRONT_CAM_LOOKAT))
        r_front = mujoco.Renderer(m, height=PE.IMG_H, width=PE.IMG_W)
        r_top = mujoco.Renderer(m, height=PE.IMG_H, width=PE.IMG_W)
        ep = {"state": [], "action": [], "front": [], "top": []}
    else:
        ts = PE.TS
        gate_every, log_every, frame_every = 5, 25, None
        ep = None
    m.opt.timestep = ts

    log = {"t": [], "bodies": {body: []}, "quats": {body: []}}
    bqadr = gate.bqadr[body]
    n_steps = int(round(t_total / ts))
    oob_samples = 0
    clamp_samples = 0
    gate_samples = 0
    tol_oob = math.radians(1.0)          # policy_eval's excursion tol
    deadband = 5e-5                      # certified rematch amendment
    for i in range(n_steps + 1):
        t = i * ts
        ctrl = MT.target_q(segs, t)
        if i % gate_every == 0:
            mujoco.mj_forward(m, d)
            gate.sample(d, ctrl)
            gate_samples += 1
            exc = np.maximum(jnt_range[:, 0] - ctrl,
                             ctrl - jnt_range[:, 1])
            if (exc > tol_oob).any():
                oob_samples += 1
            if (exc > deadband).any():
                clamp_samples += 1
        if i % log_every == 0:
            mujoco.mj_forward(m, d)
            log["t"].append(d.time)
            log["bodies"][body].append(d.qpos[bqadr:bqadr + 3].copy())
            log["quats"][body].append(d.qpos[bqadr + 3:bqadr + 7].copy())
        if record and i % frame_every == 0:
            mujoco.mj_forward(m, d)
            ep["state"].append(np.degrees(d.qpos[qadr]).astype(np.float32))
            ep["action"].append(np.degrees(ctrl).astype(np.float32))
            ep["front"].append(PE._img_tensor(r_front, m, d, "front"))
            ep["top"].append(PE._img_tensor(r_top, m, d, "top"))
        if i < n_steps:
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)

    log["bodies"][body] = np.array(log["bodies"][body])
    log["quats"][body] = np.array(log["quats"][body])
    log["t"] = np.array(log["t"])
    rows = gate.rows
    max_single = max((r[body]["max_single"] for r in rows), default=0.0)
    # MT.settled returns metres; Tier 1's settled bound is 2 mm
    # (the certified rematch checker's convention).
    settled_mm = float(MT.settled(log, body)) * 1000.0 \
        if len(log["t"]) else 999.0
    final_pos = log["bodies"][body][-1]
    containment = PE._containment("sim1", final_pos, settled_mm)
    clamp_frac = clamp_samples / max(gate_samples, 1)
    legal = bool(oob_samples == 0 and clamp_frac
                 <= PE.LEGAL_CLAMP_FRAC_MAX)
    tier1 = {
        "pass": bool(containment["pass"] and max_single
                     < DAMAGE_BOUND_N and legal),
        "containment": containment,
        "damage": {"max_single_contact_N_whole_run":
                   round(float(max_single), 2),
                   "bound_N": DAMAGE_BOUND_N,
                   "bound_status": "PROVISIONAL", "pass":
                   bool(max_single < DAMAGE_BOUND_N)},
        "legal_actuation": {
            "pass": legal, "oob_samples": oob_samples,
            "clamp_samples": clamp_samples,
            "clamp_frac": round(clamp_frac, 4),
            "counting_rule": "per 100 Hz gate sample of the plan's "
                             "commanded targets: out-of-range iff "
                             "excursion > 1.0 deg (policy_eval rule); "
                             "clamp sample iff excursion > 5e-5 rad "
                             "(the certified rematch amendment's "
                             "deadband — the model's own GRIP_OPEN "
                             "constant sits ~8e-7 rad above the "
                             "rounded joint bound)"},
        "settled_mm": round(settled_mm, 2),
        "final_object_pos": [round(float(v), 4) for v in final_pos],
    }
    return tier1, ep


# ------------------------------------------------------------- SIM-2 runs

def _run_hardened_subprocess(strategy, aim_dx, mu, mass, rung, record,
                             out_dir, model_dir):
    """One vendored-hardened-pilot run in a subprocess (the
    machinery reads its configuration from the environment at
    import time). Returns the runner's result dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ZK0_HARDENED_MODEL_DIR"] = str(model_dir)
    cmd = [sys.executable, "-m", "zk0_node.sim.hardened_run",
           "--strategy", strategy, "--aim-dx", str(aim_dx),
           "--mu", str(mu), "--rung", str(rung),
           "--out", str(out_dir)]
    if mass is not None:
        cmd += ["--mass", str(mass)]
    if record:
        cmd += ["--record"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=env, timeout=1200)
    if proc.returncode != 0 or "HARDENED_RUN_DONE" not in proc.stdout:
        raise RuntimeError(
            f"hardened run failed ({strategy} aim {aim_dx} mu {mu} "
            f"mass {mass}): rc={proc.returncode}\n{proc.stdout[-2000:]}\n"
            f"{proc.stderr[-2000:]}")
    result = json.loads((out_dir / "result.json").read_text())
    result["subprocess_wall_clock_s"] = round(time.time() - t0, 2)
    # The runner's private model copy is scratch; the shared work
    # model dir (ZK0_HARDENED_MODEL_DIR) is the caller's.
    shutil.rmtree(out_dir / "model", ignore_errors=True)
    return result


# ------------------------------------------------------------- Stage 7

def _episode_features():
    return {
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


def _write_admitted_dataset(episodes, root, node_tag):
    """Write the admitted episodes as one LeRobot dataset (the
    checkpoint's feature shapes; per-episode task text). episodes:
    [{task_id, state_deg, action_deg, front, top}]. Returns the
    dataset identity dict."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .sim import policy_eval as PE
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=f"zk0/{node_tag}-admitted",
                               fps=30, features=_episode_features(),
                               root=root, use_videos=True)
    total = 0
    for ep in episodes:
        n = len(ep["state_deg"])
        for i in range(n):
            ds.add_frame({
                "observation.state": ep["state_deg"][i],
                "observation.images.front": ep["front"][i],
                "observation.images.top": ep["top"][i],
                "action": ep["action_deg"][i],
                "task": PE.TASK_TEXT[ep["task_id"]],
            })
            total += 1
        ds.save_episode()
    ds.finalize()  # LeRobot 0.7: flushes episode metadata
    return {"repo_id": f"zk0/{node_tag}-admitted", "root": str(root),
            "episodes": len(episodes), "frames": total, "fps": 30}


def run_admission(cfg=None, out_root=None, stamp=None) -> dict:
    """Stage 7: generate candidates, run the envelope draws, admit
    what survives, write the admitted dataset + signed record."""
    from .sim import policy_eval as PE

    cfg = cfg or load_config()
    key = load_or_create_key()
    stamp = stamp or utcnow_iso().replace(":", "").replace("-", "")
    out_root = Path(out_root) if out_root else (
        zk0_home() / "records" / f"stage7-admission-{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)
    envelope = envelope_declaration()
    t_start = time.time()

    # Private model copy for planning + execution (assets stay
    # clean), with the pristine arm copy the friction patch needs.
    work_model = out_root / "model-work"
    if work_model.exists():
        shutil.rmtree(work_model)
    shutil.copytree(assets.model_dir("realistic"), work_model)
    (work_model / "so101.pristine.xml").write_text(
        (work_model / "so101.xml").read_text())
    os.environ["ZK0_SIM_MODEL_DIR"] = str(work_model)
    if str(PE.HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(PE.HARNESS_DIR))
    import multitask as MT

    candidates = []
    admitted_episodes = []   # dataset payloads, in candidate order
    sidecar = []

    # ---- SIM-1 candidates: the certified scripted pilot (the
    # package's SIM-1 generator since slice 2's demodata). The
    # noise-hardened realistic law is mug-config-only in the
    # certified record (V-025/V-026 scope) — see the deviation note
    # in the record.
    rng = np.random.default_rng(4100)
    sim1_specs = []
    for i in range(4):
        obj_xy = (0.26 + float(rng.uniform(-0.006, 0.006)),
                  0.02 + float(rng.uniform(-0.006, 0.006)))
        place_xy = (0.27 + float(rng.uniform(-0.006, 0.006)),
                    -0.14 + float(rng.uniform(-0.006, 0.006)))
        sim1_specs.append((f"sim1-cand-{i + 1}", obj_xy, place_xy))
    for cid, obj_xy, place_xy in sim1_specs:
        P = _sim1_plan(MT, obj_xy, place_xy)
        draws = envelope_draws("sim1")
        draw_results = []
        for dr in draws:
            t0 = time.time()
            tier1, _ = _sim1_execute(PE, MT, P.segs, P.t, obj_xy,
                                     dr["mu"], dr["mass_kg"],
                                     work_model, record=False)
            draw_results.append({**dr, "tier1": tier1,
                                 "wall_clock_s": round(time.time() - t0, 2)})
        # Nominal-condition recording run (the dataset episode if
        # admitted): nominal physics, 480 Hz exact-30fps capture.
        tier1_nom, ep = _sim1_execute(PE, MT, P.segs, P.t, obj_xy,
                                      0.35, 0.064, work_model,
                                      record=True)
        failed = [d["draw_id"] for d in draw_results
                  if not d["tier1"]["pass"]]
        cand = {
            "candidate_id": cid, "task": "SIM-1",
            "generator": {
                "kind": "scripted pilot (certified multitask "
                        "pick_place plan, task1 constants)",
                "plan_obj_xy": [round(v, 4) for v in obj_xy],
                "plan_place_xy": [round(v, 4) for v in place_xy],
                "plan_duration_s": round(float(P.t), 2)},
            "marks": marks_block(
                "sim-trained",
                "ORACLE-GUIDED",
                {"envelope_id": envelope["envelope_id"],
                 "status": "pending draws"}),
            "nominal_run": {"mu": 0.35, "mass_kg": 0.064,
                            "tier1": tier1_nom,
                            "recording": "480 Hz physics, exact "
                                         "30 fps capture (demodata "
                                         "discipline); scored with "
                                         "a 96 Hz gate sampling",
                            "frames": int(len(ep["state"]))},
            "draws": draw_results,
            "admitted": not failed,
            "exclusion_reasons": (
                [] if not failed else
                [f"Tier 1 failed at draw(s): {', '.join(failed)}"]),
        }
        candidates.append(cand)
        if not failed:
            admitted_episodes.append({
                "task_id": "sim1", "state_deg": ep["state"],
                "action_deg": ep["action"], "front": ep["front"],
                "top": ep["top"]})
            sidecar.append({"episode_index": len(sidecar),
                            "candidate_id": cid, "task": "SIM-1",
                            "marks": cand["marks"]})
            cand["marks"]["envelope"]["status"] = \
                f"ADMITTED — Tier 1 at all {len(draws)} draws"
        else:
            cand["marks"]["envelope"]["status"] = \
                "NOT ADMITTED — excluded from the dataset"

    # ---- SIM-2 candidates: the certified noise-hardened realistic
    # pilot (vendored, sim/hardened), rung 3.
    sim2_model = out_root / "model-hardened"
    if sim2_model.exists():
        shutil.rmtree(sim2_model)
    shutil.copytree(assets.model_dir("realistic"), sim2_model)
    sim2_specs = [("sim2-cand-1", "rim", -0.003),
                  ("sim2-cand-2", "rim", 0.0),
                  ("sim2-cand-3", "rim", 0.003),
                  ("sim2-cand-4", "pinch", -0.003),
                  ("sim2-cand-5", "pinch", 0.003)]
    for cid, strategy, aim_dx in sim2_specs:
        draws = envelope_draws("sim2")
        draw_results = []
        for dr in draws:
            res = _run_hardened_subprocess(
                strategy, aim_dx, dr["mu"], dr["mass_kg"], 3, False,
                out_root / "runs" / cid / dr["draw_id"], sim2_model)
            draw_results.append({
                **dr,
                "tier1": {**res["tier1"],
                          "pass": res["tier1"]["pass_"]},
                "tier2": res["tier2"],
                "controller_summary": res["controller_summary"],
                "wall_clock_s": res["subprocess_wall_clock_s"]})
        nom = _run_hardened_subprocess(
            strategy, aim_dx, 0.35, None, 3, True,
            out_root / "runs" / cid / "nominal", sim2_model)
        npz = np.load(out_root / "runs" / cid / "nominal"
                      / "episode.npz")
        failed = [d["draw_id"] for d in draw_results
                  if not d["tier1"]["pass"]]
        cand = {
            "candidate_id": cid, "task": "SIM-2",
            "generator": {
                "kind": "certified noise-hardened realistic pilot "
                        "(vendored V-026 machinery, sim/hardened)",
                "strategy": strategy, "aim_dx_m": aim_dx,
                "controller": nom["controller"],
                "telemetry_rung": 3},
            "marks": marks_block(
                "sim-trained",
                "ORACLE-GUIDED",
                {"envelope_id": envelope["envelope_id"],
                 "status": "pending draws"}),
            "marks_note": "generation axis: the arm plan is "
                          "ground-truth-guided (targets the settle-"
                          "probe position), so the trajectory is "
                          "ORACLE-GUIDED under the decision-55 "
                          "mapping; the gripper force law — the "
                          "part admission stresses at rung 3 — is "
                          "the TELEMETRY-ONLY hardened realistic "
                          "controller (consumes only the delivered "
                          "{q, duty} sample). Recorded as one axis "
                          "value plus this note, never merged.",
            "nominal_run": {
                "mu": 0.35, "mass_kg": 0.118,
                "tier1": {**nom["tier1"],
                          "pass": nom["tier1"]["pass_"]},
                "tier2": nom["tier2"],
                "controller_summary": nom["controller_summary"],
                "recording": "the pilot's own 30 fps recording "
                             "(state/action/front/top) in the "
                             "certified scene; camera poses are "
                             "the certified harness's (front "
                             "(0.52,-0.02,0.34), top (0.24,0,0.85)) "
                             "— NOT the Stage 5 suite's poses; a "
                             "declared distribution caveat for the "
                             "Stage 8 benchmark",
                "frames": int(len(npz["state_rad"]))},
            "draws": draw_results,
            "admitted": not failed,
            "exclusion_reasons": (
                [] if not failed else
                [f"Tier 1 failed at draw(s): {', '.join(failed)}"]),
        }
        candidates.append(cand)
        if not failed:
            admitted_episodes.append({
                "task_id": "sim2",
                "state_deg": np.degrees(npz["state_rad"]
                                        ).astype(np.float32),
                "action_deg": np.degrees(npz["action_rad"]
                                         ).astype(np.float32),
                "front": npz["cam_front"], "top": npz["cam_top"]})
            sidecar.append({"episode_index": len(sidecar),
                            "candidate_id": cid, "task": "SIM-2",
                            "marks": cand["marks"]})
            cand["marks"]["envelope"]["status"] = \
                f"ADMITTED — Tier 1 at all {len(draws)} draws"
        else:
            cand["marks"]["envelope"]["status"] = \
                "NOT ADMITTED — excluded from the dataset"

    # ---- dataset + summary
    dataset = None
    if admitted_episodes:
        ds_root = zk0_home() / "datasets" / f"admitted-{stamp}"
        dataset = _write_admitted_dataset(
            admitted_episodes, ds_root, cfg["node_id"][:8])
        (ds_root / "episodes.json").write_text(json.dumps(
            {"envelope_id": envelope["envelope_id"],
             "episodes": sidecar}, indent=1))
        dataset["episodes_sidecar"] = str(ds_root / "episodes.json")

    per_draw = {}
    for cand in candidates:
        for d in cand["draws"]:
            e = per_draw.setdefault(
                f"{cand['task']}/{d['draw_id']}",
                {"task": cand["task"], "draw_id": d["draw_id"],
                 "mu": d["mu"], "mass_kg": d["mass_kg"],
                 "passed": 0, "candidates": 0})
            e["candidates"] += 1
            e["passed"] += 1 if d["tier1"]["pass"] else 0
    for e in per_draw.values():
        e["pass_rate"] = round(e["passed"] / e["candidates"], 4)

    record = {
        "record_type": "trajectory_admission",
        "schema": "zk0-node-flow/0.1",
        "created_utc": utcnow_iso(),
        "stage": "Stage 7 — trajectory admission (SSOT decision 54)",
        "node": {"node_id": cfg["node_id"],
                 "runtime_manifest_sha256": manifest_sha256(
                     runtime_manifest_path())},
        "preregistration_status": PREREG_STATUS,
        "telemetry_rung": 3,
        "rung_note": "admission runs the Tier 1 checker at telemetry "
                     "rung 3 (contract section 6 ladder) against "
                     "the controller: SIM-2's controller consumes "
                     "the ladder-degraded {q, duty} sample; SIM-1's "
                     "scripted plan consumes no run-time telemetry "
                     "(open loop), so for SIM-1 the draws vary the "
                     "physics and the ladder has no input path to "
                     "degrade — recorded plainly rather than "
                     "dressed as a rung-3 sensing test",
        "envelope": envelope,
        "candidates": candidates,
        "summary": {
            "candidates": len(candidates),
            "admitted": sum(1 for c in candidates if c["admitted"]),
            "excluded": sum(1 for c in candidates
                            if not c["admitted"]),
            "per_task": {
                t: {"candidates": sum(1 for c in candidates
                                      if c["task"] == t),
                    "admitted": sum(1 for c in candidates
                                    if c["task"] == t and c["admitted"])}
                for t in ("SIM-1", "SIM-2")},
            "per_draw_pass_rates": sorted(per_draw.values(),
                                          key=lambda e: (e["task"],
                                                         e["draw_id"])),
        },
        "dataset": dataset,
        "deviations": [
            "SIM-1 candidate generation: the brief names the "
            "certified noise-hardened realistic pilot for both "
            "tasks, but that law exists in the certified record "
            "only for the mug configs (V-025/V-026 scope — rim-band "
            "and wall-pinch strategies on the YCB mug). No certified "
            "hardened cube controller exists. SIM-1 candidates are "
            "therefore generated by the package's certified "
            "scripted pilot (the SIM-1 generator since slice 2, "
            "ORACLE-GUIDED), and admission stresses it over the "
            "same envelope draws. Extending the hardened law to "
            "the cube is new controller work, not a vendoring.",
            "SIM-2 training episodes are recorded in the certified "
            "harness scene (its own camera poses), not the Stage 5 "
            "suite scene — the certified pilot exists only there. "
            "The Stage 8 benchmark therefore measures, among other "
            "things, transfer across that declared gap.",
        ],
        "wall_clock_s": round(time.time() - t_start, 2),
    }
    out = out_root / "admission_record.json"
    write_signed_record(record, out, key)
    record["path"] = str(out)
    return record


# ------------------------------------------------------------- Stage 8

def run_finetune(dataset_root, steps=None, lr=None, proximal_mu=None,
                 out_dir=None, cfg=None) -> dict:
    """Stage 8 (training half): fine-tune the shared checkpoint
    LOCALLY on the admitted dataset through the FL client's own
    fit path (action-expert subset only; identical procedure to a
    round contribution). Writes the fine-tuned checkpoint dir +
    signed fine-tune record."""
    import torch

    from .fl.client import Zk0NumPyClient
    from .fl.common import (arrays_hash, full_model_sha256,
                            get_subset, save_updated_checkpoint)
    from .sim.policy_eval import checkpoint_dir, load_policy

    cfg = cfg or load_config()
    key = load_or_create_key()
    steps = FINETUNE_BUDGET["steps"] if steps is None else steps
    lr = FINETUNE_BUDGET["lr"] if lr is None else lr
    proximal_mu = (FINETUNE_BUDGET["proximal_mu"]
                   if proximal_mu is None else proximal_mu)
    stamp = utcnow_iso().replace(":", "").replace("-", "")
    out_dir = Path(out_dir) if out_dir else (
        zk0_home() / "records" / f"stage8-finetune-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_policy()
    bundle.policy.float()
    client = Zk0NumPyClient(bundle.policy, bundle.pre,
                            str(dataset_root), cfg["node_id"], key,
                            steps=steps, lr=lr,
                            batch_size=FINETUNE_BUDGET["batch_size"])
    initial = get_subset(bundle.policy)
    before_hash = arrays_hash(initial)
    device = ("cuda:" + torch.cuda.get_device_name(0)
              if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    updated, n_examples, metrics = client.fit(
        initial, {"steps": steps, "lr": lr,
                  "proximal_mu": proximal_mu})
    wall = round(time.time() - t0, 2)
    model_dir = out_dir / "model"
    save_updated_checkpoint(bundle.policy, checkpoint_dir(), model_dir)
    record = {
        "record_type": "local_finetune",
        "schema": "zk0-node-flow/0.1",
        "created_utc": utcnow_iso(),
        "stage": "Stage 8 — local fine-tune (the local-only "
                 "baseline; benchmark suites are separate Stage 5 "
                 "records)",
        "node": {"node_id": cfg["node_id"],
                 "runtime_manifest_sha256": manifest_sha256(
                     runtime_manifest_path())},
        "preregistration_status": PREREG_STATUS,
        "dataset_root": str(dataset_root),
        "budget": {"steps": steps, "lr": lr,
                   "batch_size": FINETUNE_BUDGET["batch_size"],
                   "proximal_mu": proximal_mu,
                   "trainable": "action-expert subset only "
                                "(155 tensors; VLM backbone frozen)",
                   "dataset_examples_frames": n_examples,
                   "wall_clock_s": wall, "hardware": device},
        "training_metrics": {k: metrics[k] for k in
                             ("train_loss", "update_hash",
                              "update_sig") if k in metrics},
        "before": {"action_expert_subset_sha256": before_hash,
                   "model_safetensors_sha256": full_model_sha256(
                       checkpoint_dir() / "model.safetensors")},
        "after": {"action_expert_subset_sha256": arrays_hash(updated),
                  "model_safetensors_sha256": full_model_sha256(
                      model_dir / "model.safetensors")},
        "model_dir": str(model_dir),
        "marks": marks_block(
            "sim-trained",
            "n/a — a trained model, not a generated trajectory; "
            "its training data's marks are in the Stage 7 admission "
            "record",
            "trained only on trajectories admitted under "
            "zk0-admission-envelope/0.1 (see Stage 7 record)"),
    }
    out = out_dir / "finetune_record.json"
    write_signed_record(record, out, key)
    record["path"] = str(out)
    return record


# ---------------------------------------------------- Stages 10 and 11

def _suite_rate(suite_record: dict) -> dict:
    t1 = suite_record["results"]["tier1"]
    return {"task": suite_record["task"]["id"],
            "successes": t1["successes"], "trials": t1["trials"],
            "rate": t1["rate"],
            "record_content_sha256":
                suite_record.get("content_sha256")}


def build_comparison(pre: dict, benchmark: dict, post: dict) -> dict:
    """The paired comparison body (pure): per task, pre-contribution
    (Stage 5 on the base checkpoint) vs Stage 8 benchmark (local
    fine-tune) vs post-contribution (Stage 10), with the section
    4.2 deltas."""
    tasks = {}
    for task in ("SIM-1", "SIM-2"):
        p, b, a = (_suite_rate(pre[task]),
                   _suite_rate(benchmark[task]),
                   _suite_rate(post[task]))
        tasks[task] = {
            "pre_contribution": p, "stage8_benchmark": b,
            "post_contribution": a,
            "parity": {
                "rule": "post >= benchmark - 5pp (preregistration "
                        "section 4.2)",
                "margin_pp": round((a["rate"] - b["rate"]) * 100
                                   + PARITY_TOLERANCE * 100, 2),
                "pass": (a["rate"] - b["rate"]) >= -PARITY_TOLERANCE
                        - 1e-12},
            "self_regression": {
                "rule": "post >= pre (the contribution must not "
                        "lower the shared model below its "
                        "pre-contribution rate)",
                "delta_pp": round((a["rate"] - p["rate"]) * 100, 2),
                "pass": a["rate"] >= p["rate"] - 1e-12},
        }
    return {"tasks": tasks,
            "fleet_regression": {
                "rule": "no task on the standing regression suite "
                        "regresses beyond 5pp (preregistration "
                        "section 4.2)",
                "status": "NOT EVALUABLE — no standing regression "
                          "suite baseline exists; per preregis"
                          "tration section 6 this first loop "
                          "ESTABLISHES the baseline (the "
                          "pre-contribution rates above are its "
                          "first entries). The criterion is "
                          "disclosed as not evaluable, never "
                          "silently passed."}}


def run_comparison(pre_paths: dict, benchmark_paths: dict,
                   post_paths: dict, round_record_path=None,
                   out_dir=None, cfg=None) -> dict:
    """Stage 10: write the signed paired-comparison record from
    the six suite records (+ the round record, when given)."""
    cfg = cfg or load_config()
    key = load_or_create_key()
    stamp = utcnow_iso().replace(":", "").replace("-", "")
    out_dir = Path(out_dir) if out_dir else (
        zk0_home() / "records" / f"stage10-comparison-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def load(paths):
        return {t: json.loads(Path(p).read_text())
                for t, p in paths.items()}

    body = build_comparison(load(pre_paths), load(benchmark_paths),
                            load(post_paths))
    record = {
        "record_type": "paired_comparison",
        "schema": "zk0-node-flow/0.1",
        "created_utc": utcnow_iso(),
        "stage": "Stage 10 — paired comparison (same suite, same "
                 "seeds: Stage 5 machinery, 20 trials per task)",
        "node": {"node_id": cfg["node_id"],
                 "runtime_manifest_sha256": manifest_sha256(
                     runtime_manifest_path())},
        "preregistration_status": PREREG_STATUS,
        "inputs": {"pre_contribution_records":
                   {t: str(p) for t, p in pre_paths.items()},
                   "stage8_benchmark_records":
                   {t: str(p) for t, p in benchmark_paths.items()},
                   "post_contribution_records":
                   {t: str(p) for t, p in post_paths.items()},
                   "round_record": (str(round_record_path)
                                    if round_record_path else None)},
        **body,
    }
    out = out_dir / "comparison_record.json"
    write_signed_record(record, out, key)
    record["path"] = str(out)
    return record


def declaration_verdict(comparison: dict,
                        round_record: dict | None) -> dict:
    """The section 4.2 verdict (pure): every criterion's state,
    computed from the comparison + round records only."""
    tasks = comparison["tasks"]
    parity = all(t["parity"]["pass"] for t in tasks.values())
    self_reg = all(t["self_regression"]["pass"]
                   for t in tasks.values())
    accepted = None
    if round_record is not None:
        parts = round_record.get("participants") or []
        accepted_ids = set()
        for rnd in parts:
            accepted_ids.update(rnd.get("accepted_node_ids") or [])
        accepted = bool(accepted_ids)
    criteria = {
        "contribution_accepted": {
            "pass": accepted,
            "note": "the round record lists at least one accepted, "
                    "signature-verified update"
                    if accepted else
                    "no round record supplied, or no accepted "
                    "update in it"},
        "parity_with_tolerance": {
            "pass": parity,
            "note": "post-contribution shared model within -5pp "
                    "of the Stage 8 local benchmark on every task"},
        "no_self_regression": {
            "pass": self_reg,
            "note": "post-contribution rate at or above the "
                    "pre-contribution rate on every task"},
        "no_fleet_regression": {
            "pass": None,
            "note": comparison["fleet_regression"]["status"].removeprefix(
                "NOT EVALUABLE — ")},
    }
    win = bool(accepted and parity and self_reg)
    return {"criteria": criteria, "win": win}


def render_declaration_text(comparison: dict, verdict: dict) -> str:
    """The declaration text — generated from the records' numbers
    by this template; never hand-written."""
    lines = []
    lines.append("zk0 Node v0.1.0 — Stage 11 win/no-win declaration "
                 "(local-network dry run)")
    lines.append("")
    lines.append(f"VERDICT: {'WIN' if verdict['win'] else 'NO WIN'} "
                 "(local dry-run scope — see scope label below)")
    lines.append("")
    lines.append("Measured Tier 1 rates (Stage 5 suite machinery, "
                 "20 trials per task, same seeds throughout):")
    for task, t in comparison["tasks"].items():
        p, b, a = (t["pre_contribution"], t["stage8_benchmark"],
                   t["post_contribution"])
        lines.append(
            f"- {task}: pre-contribution {p['successes']}/"
            f"{p['trials']} ({p['rate']:.0%}); Stage 8 local "
            f"benchmark {b['successes']}/{b['trials']} "
            f"({b['rate']:.0%}); post-contribution {a['successes']}/"
            f"{a['trials']} ({a['rate']:.0%}). Parity margin "
            f"{t['parity']['margin_pp']:+.1f}pp (tolerance -5pp): "
            f"{'PASS' if t['parity']['pass'] else 'FAIL'}. "
            f"Self-regression delta {t['self_regression']['delta_pp']:+.1f}pp: "
            f"{'PASS' if t['self_regression']['pass'] else 'FAIL'}.")
    lines.append("")
    lines.append("Criteria (preregistration section 4.2):")
    for name, c in verdict["criteria"].items():
        state = ("PASS" if c["pass"] else
                 "NOT EVALUABLE" if c["pass"] is None else "FAIL")
        lines.append(f"- {name}: {state} — {c['note']}")
    lines.append("")
    lines.append("Scope label: this declaration covers a LOCAL "
                 "network dry run (spec section 2.3). The round's "
                 "second participant contributed an identity/no-op "
                 "update and is labeled as a dry-run participant "
                 "in the round record. Local records stay local; "
                 "nothing here enters a hosted transparency log.")
    lines.append("Per spec section 5.4, a Stage 11 declaration "
                 "does NOT establish the network-level claim "
                 "(shared model beats local-only by +10pp on the "
                 "registered set); it establishes only the "
                 "contribution-level criteria above.")
    lines.append("Preregistration status: founder-signed "
                 "2026-11-17; the base checkpoint triggered the "
                 "section 5 mis-scope condition in substance "
                 "(slice-2 verifier ruling); the founder's "
                 "re-registration decision is PENDING. This "
                 "declaration is machinery output over a "
                 "sim-trained policy derived from that checkpoint, "
                 "not a registered benchmark claim.")
    return "\n".join(lines)


def run_declaration(comparison_path, round_record_path=None,
                    out_dir=None, cfg=None) -> dict:
    """Stage 11: generate the declaration FROM the records and
    write it as a signed record (text embedded verbatim)."""
    cfg = cfg or load_config()
    key = load_or_create_key()
    stamp = utcnow_iso().replace(":", "").replace("-", "")
    out_dir = Path(out_dir) if out_dir else (
        zk0_home() / "records" / f"stage11-declaration-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = json.loads(Path(comparison_path).read_text())
    round_record = (json.loads(Path(round_record_path).read_text())
                    if round_record_path else None)
    verdict = declaration_verdict(comparison, round_record)
    text = render_declaration_text(comparison, verdict)
    record = {
        "record_type": "win_declaration",
        "schema": "zk0-node-flow/0.1",
        "created_utc": utcnow_iso(),
        "stage": "Stage 11 — win/no-win declaration (generated "
                 "from the Stage 10 comparison + round records; "
                 "the text below is machine-generated from their "
                 "numbers, never hand-written)",
        "node": {"node_id": cfg["node_id"],
                 "runtime_manifest_sha256": manifest_sha256(
                     runtime_manifest_path())},
        "preregistration_status": PREREG_STATUS,
        "comparison_record": str(comparison_path),
        "comparison_content_sha256": comparison.get("content_sha256"),
        "round_record": (str(round_record_path)
                         if round_record_path else None),
        "verdict": verdict,
        "declaration_text": text,
    }
    out = out_dir / "declaration_record.json"
    write_signed_record(record, out, key)
    record["path"] = str(out)
    return record


# ------------------------------------------------------------- dry run

def _driver_cli(args, home, log_path, timeout=3600):
    """Run one `zk0` CLI command as a subprocess under a node home
    (process isolation per stage: the policy bundle cache, the
    vendored-module import discipline and the FL server all assume
    one role per process). Returns (rc, stdout)."""
    env = dict(os.environ)
    env["ZK0_HOME"] = str(home)
    proc = subprocess.run(
        [sys.executable, "-m", "zk0_node.cli", *args],
        capture_output=True, text=True, env=env, timeout=timeout)
    Path(log_path).write_text(proc.stdout + "\n--- stderr ---\n"
                              + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"zk0 {' '.join(args)} failed rc={proc.returncode} "
            f"(home {home}); log: {log_path}\n"
            f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    return proc.stdout


def _json_last_line(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no JSON line in CLI output:\n{stdout[-1000:]}")


def _suite_records(home: Path) -> set:
    return {str(p) for p in home.glob(
        "records/stage5-*/eval_record.json")}


def _new_suite_record(home: Path, before: set) -> str:
    new = _suite_records(home) - before
    if len(new) != 1:
        raise RuntimeError(f"expected exactly 1 new suite record, "
                           f"found {sorted(new)}")
    return new.pop()


def run_dry_run(workdir, cfg=None) -> dict:
    """The full-loop dry run (spec section 3, stages 5-11 in one
    scripted pass): fresh node homes, checkpoint eval, admission,
    local fine-tune + benchmark, a local-network contribution
    round with a labeled identity/no-op second participant,
    paired comparison, declaration — and a signed chain record
    over every link, each verified with the package's own record
    verification (the `zk0 verify` rule)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logs = workdir / "logs"
    logs.mkdir(exist_ok=True)
    main_home = zk0_home()
    home_a = workdir / "node-a"
    home_b = workdir / "node-b"
    for home in (home_a, home_b):
        home.mkdir(parents=True, exist_ok=True)
        # Dev-machine convenience, stated plainly (slice-2
        # precedent): the fresh homes share the main home's staged
        # assets + checkpoint via symlinks. On separate machines
        # each node stages its own; nothing in the machinery
        # depends on the sharing.
        for name in ("assets", "checkpoint"):
            link = home / name
            if not link.exists():
                link.symlink_to(main_home / name)

    stages = []
    chain_paths = []

    def stage(name, fn):
        t0 = time.time()
        print(f"[dry-run] {name} ...", flush=True)
        detail = fn()
        stages.append({"stage": name,
                       "wall_clock_s": round(time.time() - t0, 2),
                       "detail": detail})
        print(f"[dry-run] {name}: done in "
              f"{stages[-1]['wall_clock_s']} s", flush=True)
        return detail

    t_all = time.time()
    status = "complete"

    def gate(home, tag):
        _driver_cli(["gate", "run"], home, logs / f"gate-{tag}.log")
        atts = sorted(home.glob("records/attestations/gate-*.json"))
        if not atts:
            raise RuntimeError(f"no attestation written for {tag}")
        chain_paths.append(str(atts[-1]))
        return {"attestation": str(atts[-1])}

    stage("setup + join gate (node A)", lambda: gate(home_a, "a"))
    stage("setup + join gate (node B, dry-run participant)",
          lambda: gate(home_b, "b"))

    def eval_suite(home, task, ckpt_dir=None):
        before = _suite_records(home)
        args = ["eval", "checkpoint", "--task", task, "--trials", "20"]
        if ckpt_dir:
            args += ["--ckpt-dir", str(ckpt_dir)]
        _driver_cli(args, home, logs / f"eval-{task}-{len(stages)}.log")
        path = _new_suite_record(home, before)
        chain_paths.append(path)
        rec = json.loads(Path(path).read_text())
        t1 = rec["results"]["tier1"]
        return {"record": path, "tier1": f"{t1['successes']}/"
                                        f"{t1['trials']}"}

    pre = {}
    pre["SIM-1"] = stage("Stage 5 — checkpoint eval SIM-1",
                         lambda: eval_suite(home_a, "sim1"))["record"]
    pre["SIM-2"] = stage("Stage 5 — checkpoint eval SIM-2",
                         lambda: eval_suite(home_a, "sim2"))["record"]

    adm = stage("Stage 7 — trajectory admission", lambda: _json_last_line(
        _driver_cli(["flow", "admit"], home_a, logs / "admit.log")))
    admission = json.loads(Path(adm["record"]).read_text())
    chain_paths.append(adm["record"])
    dataset_root = (admission.get("dataset") or {}).get("root")

    bench = {}
    post = {}
    comparison_path = declaration_path = None
    if not dataset_root:
        status = "aborted_no_admitted_trajectories"
        stages.append({"stage": "Stages 8-11",
                       "wall_clock_s": 0.0,
                       "detail": "not run: admission admitted no "
                                 "trajectories, so there is no "
                                 "dataset to fine-tune on. The "
                                 "chain ends here, honestly."})
    else:
        ft = stage("Stage 8 — local fine-tune", lambda: _json_last_line(
            _driver_cli(["flow", "finetune", "--dataset", dataset_root],
                        home_a, logs / "finetune.log")))
        chain_paths.append(ft["record"])
        ft_model = ft["model_dir"]
        bench["SIM-1"] = stage(
            "Stage 8 — benchmark eval SIM-1 (fine-tuned model)",
            lambda: eval_suite(home_a, "sim1", ft_model))["record"]
        bench["SIM-2"] = stage(
            "Stage 8 — benchmark eval SIM-2 (fine-tuned model)",
            lambda: eval_suite(home_a, "sim2", ft_model))["record"]

        round_info = {}

        def do_round():
            port, enroll_port = 8917, 8918
            env = dict(os.environ)
            env["ZK0_HOME"] = str(home_a)
            server_log = open(logs / "round-server.log", "w")
            server = subprocess.Popen(
                [sys.executable, "-m", "zk0_node.cli", "fl",
                 "start-network", "--port", str(port),
                 "--enroll-port", str(enroll_port),
                 "--steps", str(FINETUNE_BUDGET["steps"]),
                 "--lr", str(FINETUNE_BUDGET["lr"]),
                 "--mu", str(FINETUNE_BUDGET["proximal_mu"]),
                 "--min-clients", "2"],
                stdout=server_log, stderr=subprocess.STDOUT, env=env)
            try:
                token_file = None
                deadline = time.time() + 600
                while time.time() < deadline:
                    hits = sorted(home_a.glob(
                        "fl/networks/*/join-token.txt"))
                    if hits:
                        token_file = hits[-1]
                        break
                    if server.poll() is not None:
                        raise RuntimeError(
                            "aggregator exited before issuing a "
                            "join token; see round-server.log")
                    time.sleep(3)
                if token_file is None:
                    raise RuntimeError("no join token after 600 s")
                token = token_file.read_text().strip()
                net_id = token_file.parent.name
                _driver_cli(
                    ["fl", "join", "--endpoint", f"127.0.0.1:{port}",
                     "--enroll-url", f"http://127.0.0.1:{enroll_port}",
                     "--token", token, "--dataset", dataset_root,
                     "--steps", str(FINETUNE_BUDGET["steps"])],
                    home_a, logs / "round-join-a.log")
                _driver_cli(
                    ["fl", "join", "--endpoint", f"127.0.0.1:{port}",
                     "--enroll-url", f"http://127.0.0.1:{enroll_port}",
                     "--token", token, "--noop-dry-run"],
                    home_b, logs / "round-join-b.log")
                deadline = time.time() + 1200
                while server.poll() is None and time.time() < deadline:
                    time.sleep(5)
                if server.poll() is None:
                    server.kill()
                    raise RuntimeError(
                        "aggregator did not exit within 1200 s of "
                        "the joins completing")
            finally:
                if server.poll() is None:
                    server.kill()
                server_log.close()
            rounds = sorted(home_a.glob(
                "records/fl-rounds/round-*.json"))
            parts_a = sorted(home_a.glob(
                "records/fl-rounds/participation-*.json"))
            parts_b = sorted(home_b.glob(
                "records/fl-rounds/participation-*.json"))
            if not rounds or not parts_a or not parts_b:
                raise RuntimeError(
                    "round artifacts missing (round/participation "
                    "records) — see round logs")
            round_info.update({
                "network_id": net_id,
                "round_record": str(rounds[-1]),
                "participation_a": str(parts_a[-1]),
                "participation_b": str(parts_b[-1])})
            chain_paths.extend([str(rounds[-1]), str(parts_a[-1]),
                                str(parts_b[-1])])
            part_a = json.loads(parts_a[-1].read_text())
            round_info["received_model_dir"] = \
                part_a["received_model_dir"]
            return round_info

        stage("Stage 9 — local-network contribution round "
              "(node B: identity/no-op dry-run participant)",
              do_round)
        recv_model = Path(round_info["received_model_dir"])
        # The received dir holds model.safetensors + the base
        # checkpoint's config files (run_client assembles it); the
        # suite loads it as a checkpoint dir.
        post["SIM-1"] = stage(
            "Stage 10 — post-contribution eval SIM-1",
            lambda: eval_suite(home_a, "sim1", recv_model))["record"]
        post["SIM-2"] = stage(
            "Stage 10 — post-contribution eval SIM-2",
            lambda: eval_suite(home_a, "sim2", recv_model))["record"]

        cmp_out = stage("Stage 10 — paired comparison",
                        lambda: _json_last_line(_driver_cli(
                            ["flow", "compare",
                             "--pre-sim1", pre["SIM-1"],
                             "--pre-sim2", pre["SIM-2"],
                             "--bench-sim1", bench["SIM-1"],
                             "--bench-sim2", bench["SIM-2"],
                             "--post-sim1", post["SIM-1"],
                             "--post-sim2", post["SIM-2"],
                             "--round-record",
                             round_info["round_record"]],
                            home_a, logs / "compare.log")))
        comparison_path = cmp_out["record"]
        chain_paths.append(comparison_path)
        dec_out = stage("Stage 11 — declaration",
                        lambda: _json_last_line(_driver_cli(
                            ["flow", "declare",
                             "--comparison", comparison_path,
                             "--round-record",
                             round_info["round_record"]],
                            home_a, logs / "declare.log")))
        declaration_path = dec_out["record"]
        chain_paths.append(declaration_path)

    # ---- verify the whole chain with the package's own rule
    links = []
    for p in chain_paths:
        rec = json.loads(Path(p).read_text())
        v = verify_record(rec)
        links.append({"record": p, "record_type":
                      rec.get("record_type"),
                      "content_sha256": rec.get("content_sha256"),
                      "hash_ok": v["hash_ok"],
                      "signature_present": v["signature_present"],
                      "signature_ok": v["signature_ok"]})
    all_ok = all(l["hash_ok"] and l["signature_ok"] for l in links)

    os.environ["ZK0_HOME"] = str(home_a)
    cfg_a = load_config()
    record = {
        "record_type": "full_loop_dry_run",
        "schema": "zk0-node-flow/0.1",
        "created_utc": utcnow_iso(),
        "stage": "Stages 5-11 — full-loop dry run (one scripted "
                 "pass from fresh node homes)",
        "node": {"node_id": cfg_a["node_id"],
                 "runtime_manifest_sha256": manifest_sha256(
                     runtime_manifest_path())},
        "preregistration_status": PREREG_STATUS,
        "status": status,
        "workdir": str(workdir),
        "node_homes": {"operator": str(home_a),
                       "dry_run_participant": str(home_b)},
        "stages": stages,
        "chain": links,
        "chain_all_verified": all_ok,
        "declaration_record": declaration_path,
        "total_wall_clock_s": round(time.time() - t_all, 2),
        "note": "time-to-declaration (total_wall_clock_s) is a "
                "preregistration section 6 output of the first "
                "loop. Every link above was verified with the "
                "package's record verification (content hash + "
                "Ed25519 signature — the `zk0 verify` rule).",
    }
    out = home_a / "records" / "full-loop-dry-run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_signed_record(record, out, load_or_create_key())
    shutil.copy2(out, workdir / "chain_record.json")
    record["path"] = str(out)
    return record
