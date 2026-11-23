# Twin-Suite v2 — Independent Verification: Rollout Harness + Grasp Blocker

**Date:** 2026-11-22 · **Verifier:** Muse subagent, separate context, raw artifacts only
**Subject:** `rollout-harness-v2.md` (Stage 1.4 part 1), DGX
`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness/`
**Governing:** twin-suite-v2-preregistration-20261122.md §4

## Verdicts

- **(i) Harness mechanics: CONFIRMED**
- **(ii) Grasp-blocker diagnosis: CONFIRMED**

No probe was found to have been run incorrectly in a way that affects
either verdict. Two minor measurement-definition differences (jaw
centroid, §4) and one loose probe-numbering point (§4) are noted; none
changes a conclusion.

## 1. Harness mechanics

- **Two-process split is real.** `sim_runner.py` runs under the sim venv;
  `policy_service.py` is spawned as a subprocess under the FL venv python
  (`zk0-study001-fl-20261121/venv/bin/python`), JSON-lines over a pipe,
  one request per 50-action chunk, frames passed as `.npy` files.
  Versions re-measured on the DGX by the verifier: sim venv MuJoCo
  **4.0.1**, FL venv MuJoCo **3.8.1** — exactly as reported. Policy service
  pins CPU (`CUDA_VISIBLE_DEVICES=""`, thread cap), smoke ran `nice -n 10`
  under the job guard.
- **Input contract matches the Study 001 eval harness** (code-read of both
  `sim_runner.py`/`policy_service.py` and
  `run/eval-traj-20261122/scripts/eval_traj.py`): CAM_MAP zip order
  front/top/wrist → camera1/camera2/camera3; renders 256×256; state [6]
  in degrees (sim qpos converted rad→deg, gripper via the fixed
  0.10 + (deg/52.3)·1.35 mapping and its inverse); verbatim per-task
  prompt (block_stack record carries `"Stack the colouful blocks on top
  of each other"`, dataset spelling preserved); A5 rename-step drop
  applied identically; noise seeded `episode*1000+chunk` in both code
  bases; 20 fps; horizon = recorded length + 20 steps (1.0 s settle) —
  smoke records show 296 = 276+20 (ep411) and 289 = 269+20 (ep453).
- **v1.1 imported read-only; no v1.1 modification.** `sim_runner.py`
  inserts the v1.1 workdir on `sys.path` and imports `twin_params` and
  `replay.evaluate()`; it writes only to its own `--out` dir. File check:
  no file under the v1.1 workdir `zk0-study001-twin-20261122` (code or
  scenes) has an mtime after 2026-11-22 20:00 — the sole later file is
  the v1.1 completion-report doc itself (20:00:53), which predates the
  v2 harness build (harness dir 21:44, smoke 22:02). Scene header still
  carries the frozen `param_hash=a4eb3f8c3611894d`.
- **Welds asserted inactive.** Runner forces scene welds off and raises
  if any `eq_active` becomes true at any physics substep; records carry
  `welds_forced_inactive: 8`. Verifier independently loaded the ep411
  scene: `neq = 8`, all `eq_active` false at init.
- **Solver-failure detection on.** v1.1 `evaluate()` result OR'd with a
  generalised check (non-finite qpos, object radial > 1.0 m, z outside
  [−0.05, 0.50] ⇒ excluded, never success) — present in code and in the
  record fields.

## 2. Smoke evidence (`smoke/base_final/rollouts.jsonl`, read directly)

4 rollouts, exactly as reported:

| Episode | Variant | sim_success | secondary | solver_failure | horizon | wall_s |
|---|---|---|---|---|---|---|
| 411 | nominal | False | True | False | 296 | 52.56 |
| 411 | v0 | False | False | False | 296 | 59.56 |
| 453 | nominal | False | True | False | 289 | 75.46 |
| 453 | v0 | False | True | False | 289 | 79.22 |

Secondary 3/4, solver failures 0/4 — matches the builder's table.
**Sample frames inspected by eye by the verifier** (copied off the DGX):
front (fixed camera) shows the yellow arm, both cubes (green/purple) and
the enclosure; top shows arm, mat and both cubes from above; wrist shows
both cubes centred ahead of the gripper with the arm filling the lower
frame. All sane, correctly exposed, 256×256.
**Kill-and-resume drill:** `smoke/guard-smoke4/guard/command.out` begins
with `[rollout] already recorded, skipping {…ep411, nominal…}` before
producing the ep411-v0 record — the completed record survived the kill
and was skipped on relaunch, as reported. The canonical `base_final` run
(guard-smoke6) shows all 4 records in order ending `SMOKE_DONE`.

## 3. Actuation claims

Vendored model `zk0-sim-v01-20261115/sim/model/so101_new_calib.xml`,
read directly by the verifier: `sts3215` default class
`kp="998.22" kv="2.731" forcerange="-2.94 2.94"`, joint damping 0.60,
frictionloss 0.052, armature 0.028; all six `<position>` actuators carry
the override `forcerange="-3.35 3.35"`. Exactly as cited.
**Ctrl-tracking, re-derived** from the actuated GT diagnostic trajectory
(`smoke/gt411/traj_block_stack__ep411_nominal.npz`, qpos vs ctrl):
mean |qpos − ctrl| = **0.0052 rad**, max **0.091 rad** — matches the
claimed ~0.005 / 0.09. Actuation is not the failure mode. The GT rollout
record itself: FAIL primary, secondary PASS, solver_failure False.

## 4. THE BLOCKER — re-run by the verifier

Probes were copied to a scratch dir on the DGX (`/tmp/verify-v2`,
removed afterwards) and re-run CPU-only, `nice -n 10`, MuJoCo 4.0.1.
The v1.1 workdir was not written.

- **Recorded-pose grasp probe (shipped `grasp_probe.py`):** at the
  recorded pose the fingers register **no cube contact at all**; closing
  to the recorded minimum (deg 2.7) changes nothing; after the recorded
  lift the cube is at z = 0.0149 m, `z_max` 0.0149 m, contacts `[world]`
  only. **No pinch-lift.** (This reproduces the builder's probe-#4
  finding; the builder's probe numbering in §5 of the report does not
  map 1:1 onto the three shipped script filenames — the five reported
  probes correspond to these scripts plus in-session variants whose
  logs `grasp_probe{,2..6}.log` are all on disk and mutually
  consistent. Noted as a documentation looseness, not an error.)
- **Scripted IK-centred grasp (shipped `grasp_probe3.py`, lowest
  descent, site target = cube + 0.005 m):** reproduced the builder's
  `grasp_probe6.log` line for line. IK error 0.0014–0.0019 m (centring
  is genuine). On descent the **fixed** jaw touches the cube
  (contacts `[gripper, world]`); on close the cube is **shoved from
  x 0.2399 → 0.2256 (−13.5 mm; builder: −14 mm)**; after the lift
  waypoint it rests at [0.2207, 0.0261, 0.0149], contacts `[world]`.
  **The moving jaw (`moving_jaw_so101_v1`) never appears in the cube's
  contact set at any step of any re-run probe, and the cube never
  lifts.** The higher-descent logs on disk (probe4/5) show the same
  pattern with even less contact.
- **Close-sweep log on disk (`grasp_probe2.log`)** corroborates: gripper
  qpos follows ctrl fully to −0.05 rad while the cube is swept sideways
  (y 0.030 → 0.010) and never lifts.
- **Jaw-arc geometry, re-measured by the verifier** at the recorded
  frame-170 arm pose, gripper joint 1.45 → −0.10 rad: the moving jaw's
  geom centroid moves z **0.104 → 0.075 m** and x **0.200 → 0.182 m**.
  The builder reported z 0.106 → 0.066, x 0.209 → 0.188 (a mesh-vertex
  centroid rather than a geom-origin mean — the absolute values differ
  by definition, the motion does not): in both measurements the jaw
  closes in a **downward/backward arc of ~3–4 cm in z**, not a lateral
  pinch at cube height. Root-cause geometry confirmed in substance.

**Probe-correctness check:** the probes enable exactly the finger
collision geoms the harness enables (contacts with both jaw bodies do
register in the recorded-pose probes, so contact enabling works), close
through and past the harness's closed endpoint (to −0.05 rad; joint
limit −0.1745), centre via IK with ≤2 mm error, and use the scene's own
cube (30 mm, 20 g, friction 0.6) with finger friction (1.0, 0.005,
0.0001). No weld or assistance is active in any probe. No incorrect
probe run was found; the blocker is **not** a probe artefact.

**Blocker verdict: CONFIRMED.** Without the weld, no probe — recorded
or optimally centred — pinch-lifts the 30 mm cube; the moving jaw does
not form an opposing contact, and closing sweeps or misses the cube.
Gate runs on grasp-dependent tasks would measure the finger model, not
the policy. The builder's recommendation (finger-pad collision geoms
as a versioned suite change, acceptance = this probe battery must
lift and hold) is the right shape of fix; no fix was attempted here.

## 5. Camera findings

Re-rendered by the verifier from the frozen ep411 scene:

- **v1.1 `frontcam` renders near-black:** mean pixel **43.0/255**,
  max 71 — the builder's 43/255 exactly. The harness's `frontfix`
  render of the same scene: mean 66.3, max 255, and the saved frame
  (inspected, §2) shows the workspace correctly.
- **No wrist camera in the v1.1 scenes:** the raw scene's camera list
  is exactly `[topcam, frontcam]` — confirmed from the compiled model.
- **Replacement poses as claimed:** compiled harness model cameras are
  `[topcam, frontcam, frontfix, wristcam]`; `frontfix` pos
  [0.75, −0.55, 0.35] and fovy 55 are **identical** to `frontcam`'s
  (only the orientation type differs, quat look-at toward
  [0.30, 0, 0.05]); `wristcam` is a worldbody camera (fovy 60) whose
  pose is set per render from the gripperframe site (eye 6 cm behind
  the gripper body along the wrist→gripper axis + 5 cm up, target
  10 cm past the site) — code matches the report's description, and
  the saved wrist frame is consistent with it.

## 6. Caveats

- The jaw-centroid absolute numbers are definition-dependent (§4);
  the sealed claim should be the arc direction/magnitude class, not
  the third decimal.
- Probe-to-script numbering in the builder report is loose (§4);
  all underlying logs are on disk and reproduce.
- FL / C-exp control checkpoints were not run by the builder and were
  not run here — correct restraint given the confirmed blocker.
