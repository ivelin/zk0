# Twin-Suite v2 — Stage 1.4 (part 1): Closed-Loop Policy Rollout Harness (MuJoCo)

**Date:** 2026-11-22 · **Builder:** Muse subagent (Stage 1.4 part 1) ·
**Governing docs:** twin-suite-v2-preregistration-20261122.md §4 (frozen,
commit bb11d20), sim-gate-design-memo-20261122.md, builder1-v11-harness.md.
**Code:** DGX `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/harness/`
(`sim_runner.py`, `policy_service.py`, `run_smoke.sh`, grasp probes).
No v1.1 file was modified (frozen, param_hash a4eb3f8c3611894d).

## Verdict up front

The harness works end-to-end (actuated closed-loop rollouts, three-camera
renders, v1.1 mechanical scoring, incremental records, kill-and-resume —
all smoke-verified below). **But Stage 1 has a BLOCKER: physical grasping
with the vendored SO-101 model's fingers does not work** — in five
separate probes the 30 mm cube was never pinch-lifted (details in §5).
Per the build brief this is reported as a blocker with evidence, not
papered over: no weld or grasp assistance was added. Gate runs on
grasp-dependent tasks should not proceed until the finger geometry /
grasp model question is decided (options in §5.3).

## 1. Architecture — venv resolution (from evidence)

Probed on the DGX (2026-11-22):

| venv | mujoco | torch | lerobot |
|---|---|---|---|
| sim venv `zk0-sim-v01-20261115/.venv` | **4.0.1** | ✗ | ✗ |
| FL venv `zk0-study001-fl-20261121/venv` | 3.8.1 | 2.15.0+cu132 | ✓ |

The FL venv *does* contain both stacks, so a single-process design was
possible — and was **rejected**: its MuJoCo is 3.8.1, a different engine
version than the v1.1 suite's 4.0.1, and silently changing the physics
engine version would break continuity with the v1.1 scoring/physics the
instrument is certified against. **Chosen: two-process design.**
`sim_runner.py` runs in the sim venv (MuJoCo 4.0.1, imports the v1.1
scoring code read-only); `policy_service.py` runs in the FL venv
(SmolVLAPolicy + pre/post processors) as a subprocess speaking JSON-lines
over a pipe — one request per 50-action chunk, frames passed as `.npy`
files in a shared tmp dir. Chunk-rate IPC cost is negligible (measured
8.3 s CPU inference per chunk vs milliseconds of IPC).

**Device choice (recorded): CPU inference for both policy and sim.**
CUDA is explicitly hidden from the policy service (`CUDA_VISIBLE_DEVICES=""`)
so DGX training is never disturbed; the whole smoke ran `nice -n 10`,
CPU-only, under `launch_dgx_job.sh` + job guard (logonly mode, JOB_LOCK=0,
telemetry mirror set on-box). Per-rollout wall time ≈ 100 s (service load
~35 s + 6 chunks × ~8.3 s + ~1 s sim).

## 2. Policy input contract (mirrors the Study 001 eval harness)

From `run/eval-traj-20261122/scripts/eval_traj.py` and the base config:

- Cameras: dataset front/top/wrist → base config keys
  camera1/camera2/camera3 (the eval harness's CAM_MAP zip order);
  renders 256×256 (base config declares [3,256,256] per camera).
- State: `observation.state` [6] in **degrees** (dataset units), current
  joint positions read from the sim each chunk (gripper via the fixed
  mapping in §3.3).
- Prompt: the dataset task string verbatim — block_stack:
  `"Stack the colouful blocks on top of each other"` (dataset spelling
  preserved, meta/tasks.parquet).
- Pre/post-processing: `make_pre_post_processors` with the A5 rename-step
  drop applied identically to the eval harness; noise seeded per
  (episode, chunk) as `episode*1000+chunk`, exactly as eval_traj.py.
  One deviation, forced by CPU: the base checkpoint's processor config
  declares device `cuda`; the service pins processors to CPU (override +
  post-hoc step pin). No other contract change.
- Loop: 20 fps (dataset rate, read from meta/info.json via v1.1
  twin_params). One policy query per 50-action chunk; each action held
  0.05 s = 100 physics substeps at the v1.1 timestep 0.0005 s.
  Horizon = recorded episode length + 20 steps (1.0 s settle, last ctrl
  held), final state read at settle end — pre-registration §4.

## 3. Actuated arm

### 3.1 Actuators already exist in the vendored model
`so101_new_calib.xml` (the model every v1.1 scene includes) already
defines six `<position>` actuators, one per joint — v1.1 replay never
used them (it set qpos kinematically and grasped via welds). The v2
harness drives the arm **only** through `data.ctrl`. Tracking in the
actuated GT diagnostic: mean |qpos − ctrl| = 0.005 rad, max 0.09 rad.

### 3.2 STS3215 limits (citations)
From the vendored model's `sts3215` default class (so101_new_calib.xml,
the recorded sim-side limit set): position kp = 998.22, kv = 2.731
(derived in the model file's own comment from a servo proportional gain
of 16), class forcerange **±2.94 N·m** (= the STS3215's 30 kg·cm stall
torque), actuator-level forcerange override **±3.35 N·m**, joint damping
0.60, frictionloss 0.052, armature 0.028. The harness keeps the model's
values untouched. **Speed:** no speed limit is recorded in the twin/sim
records; the harness additionally slew-limits ctrl targets at
6.16 rad/s (Feetech STS3215 rated 0.17 s/60° at 7.4 V) — flagged as a
harness instrument parameter (datasheet-sourced, not record-sourced),
to be frozen or amended in the suite manifest.

### 3.3 Gripper calibration (instrument parameter)
Dataset gripper.pos: LOW = closed, HIGH = open (v1.1 calibration finding,
replay.closedness_trace). Fixed global mapping onto the v1.1 visual
endpoints: joint_rad = 0.10 + (deg/52.3)·1.35 (52.3 = dataset global max,
meta/stats.json); inverse for the state vector. Set on the development
subset only, per pre-registration §4.

### 3.4 Cameras — two instrument fixes (documented, dev-subset scope)
- **Front:** the v1.1 scene `frontcam` (xyaxes orientation) renders
  near-black (verified by eye; mean pixel 43/255 of a mostly empty
  frame, image contentless). MjSpec keeps the element's xyaxes
  orientation type, so assigning a quat is silently ignored (probed).
  The harness adds a quat-typed camera `frontfix` at the **same
  position and fovy**, look-at toward the workspace centre (0.30, 0, 0.05).
  The fixed front view shows arm, both cubes and enclosure correctly.
- **Wrist:** the v1.1 scenes have **no wrist camera at all**. The harness
  adds `wristcam` as a worldbody camera whose pose is set per render
  from the gripperframe site: eye 6 cm behind the gripper along the
  wrist→gripper axis + 5 cm up, target 10 cm past the site. Verified by
  eye: on approach the target cube is centred ahead of the gripper; at
  grasp the gripper fills the lower frame (comparable in spirit to the
  dataset wrist view). Placement is an instrument parameter for Stage 1
  tuning, not a certified viewpoint match.

## 4. Variants and scoring

- Variant consumption: the runner loads the frozen v1.1 variant XMLs
  directly (`<task>__ep<id>_v<k>.xml`; nominal = `_v0`-equivalent base
  XML) and records the variant's draws from the frozen
  `<episode>_variants.json` (seed 20261122, param_hash a4eb3f8c…) into
  the rollout record — the identical scene inputs a Drake port must
  consume.
- Scoring: v1.1 `replay.evaluate()` imported read-only (contacts_with
  criteria, secondary thresholds 3.0 cm / ring 2.0 cm, cable solver
  detection), plus a generalised solver-failure check (any tracked
  object non-finite, radial > 1.0 m, or z outside [−0.05, 0.50] ⇒
  excluded, counted, never scored as success) per the v1.1 standard.
- The scene weld equalities are forced inactive and **asserted inactive
  at every physics step** (a weld activating aborts the rollout).

## 5. GRASP FINDING — Stage 1 BLOCKER

Setup common to all probes: welds off; arm-geom contacts off except the
two gripper-end collision geoms (fixed jaw on body `gripper`, moving jaw
on `moving_jaw_so101_v1`), friction set to the parameter table's finger
value (1.0, 0.005, 0.0001) (physics-parameters.md: "SO-101 gripper
fingers/pads"); cube 30 mm / 20 g / friction 0.6 (scene values).

| # | Probe | Result |
|---|---|---|
| 1 | Actuated GT replay, ep411 nominal (recorded actions through ctrl) | Cube z_max 0.027 m (a push, not a lift); ends unstacked at its start pose. Primary FAIL, secondary PASS. Tracking mean err 0.005 rad — actuation is not the problem. |
| 2 | Recorded grasp pose (frame 130) + close + recorded lift | Fingers touch the cube, closing **drags it along the mat** (xy 0.245→0.198); z never exceeds 0.0151 m; left behind. |
| 3 | Close-sweep at frame 130 (ctrl 0.162 → −0.05 rad) | Gripper qpos follows ctrl fully; cube is **swept sideways** (y 0.030→0.010 at −0.05) instead of being pinched; never lifts. |
| 4 | Recorded hold pose (frame 190) | Fingers **never contact** the scene cube — recorded FK pose vs image-placed cube offset ≈ 7.4 cm in xy (the known FK↔image frame disagreement; gripperframe site z also goes to −0.016 m, below the mat, at frame 121). |
| 5 | Scripted centred grasp (Jacobian IK on gripperframe site, 3 descent heights: site z = cube+0.045 / +0.020 / +0.005) | At the two higher descents the jaw closes in air (jaw mesh lowest point is only 3.4 mm above the site; fixed finger 7 mm below it — site-targeted descent is easy to get wrong). At the lowest descent the fixed side touches the cube and closing **shoves it −14 mm in x**; the **moving jaw never registers a cube contact in any scripted probe**, and the cube never lifts. |

Root-cause geometry (measured, frame-170 pose): as the gripper joint
goes 1.45 → −0.10 rad, the moving-jaw mesh centroid travels
z 0.106 → 0.066 m and x 0.209 → 0.188 m — **the jaw closes in a
downward/backward arc, not a lateral pinch**; the fixed and moving
meshes are already at ~0 mm minimum distance at every angle (they meet
at the root, not at opposing faces at object height). There are no
opposing pinch faces at the 30 mm cube's height in this vendored model:
closing sweeps the object away or misses it.

### 5.1 What this is not
Not an actuation failure (tracking 0.005 rad), not a friction-value
failure (a 20 g cube needs ≥0.33 N of pinch normal force at μ=0.6 — the
pinch itself never forms), and not the weld in disguise (no assistance
of any kind was added or will be added silently).

### 5.2 Consequence
Closed-loop gate rollouts on grasp-dependent tasks cannot produce
meaningful success signal with this finger model: even a perfect policy
cannot pinch-lift the cube. Known-ordering controls (§3.4 of the
pre-registration) would fail for instrument reasons, not policy reasons.

### 5.3 Options for the founder/verifier decision (not taken here)
(a) Repair the finger geometry in the vendored model (true opposing
jaw faces at object height) as a **new suite version** with full
re-validation — a scene/model change, not a harness tweak;
(b) add simple box/capsule finger-pad collision geoms parented to the
jaw bodies (declared scene-template change, same versioning rules);
(c) restrict v2 gate claims to non-grasp outcome classes until (a)/(b).
Any of these is a minor/major suite change under §11 change control and
needs its own verification. **Recommendation: (b) then re-run this
grasp probe battery as the acceptance test** (scripted centred grasp
must lift the cube and hold it 1 s).

## 6. Full-path smoke (miniature, per standing rule)

Checkpoint: base `lerobot/smolvla_base` at the pinned snapshot
`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` (local HF cache, matches
pre-registration §6). Episodes: block_stack 411 + 453 (development
subset), nominal + variant v0. Results (canonical run, `smoke/base_final/`, guard rundir
`smoke/guard-smoke6`, SMOKE_DONE):

| Episode | Variant | sim_success | secondary | solver_failure | wall |
|---|---|---|---|---|---|
| 411 | nominal | False | True | False | 52.6 s |
| 411 | v0 | False | False | False | 59.6 s |
| 453 | nominal | False | True | False | 75.5 s |
| 453 | v0 | False | True | False | 79.2 s |

Base fails every rollout, as expected (its sealed floor is 0/100);
horizons 296 / 289 steps = recorded lengths (276 / 269) + 20 settle
steps. Variant v0's draws are the frozen empty-jitter draws, recorded
verbatim in each record. (Wall times are rollout-internal and varied
with the training chain's load on the box; total guarded run ~25 min
for 4 rollouts under contention, ~100 s/rollout uncontended.)

- Renders sane: sample frames saved per rollout (`frame0_front/top/wrist.png`
  in the smoke dir) and inspected by eye — front (fixed) shows arm +
  both cubes + enclosure; top shows arm, mat, cubes; wrist shows the
  green cube ahead of the gripper on approach.
- Actions move the arm: ep411 nominal trajectory qpos std per joint up
  to 0.13 rad (shoulder_lift range −1.32…0.13 rad); cubes unmoved —
  base fails the task, as expected. Smoke success = mechanics, and the
  mechanics pass.
- Scoring executed on every rollout (v1.1 evaluate()); records persisted
  incrementally to `rollouts.jsonl` (fsync per record) + trajectory npz.
- **Kill-and-resume drill:** the smoke was killed mid-rollout-2
  (pkill of runner + service); the completed ep411-nominal record
  survived intact, and the relaunch printed
  `already recorded, skipping {…ep411, nominal…}` (guard-smoke4
  command.out) and resumed with the remaining rollouts — the completed
  rollout was neither lost nor re-run. Recorded honestly: my
  overlapping relaunches during the drill let one in-flight rollout
  also append to the scratch dir (`smoke/base/`), which therefore
  holds one duplicate ep411-v0 line; the drill dir is scratch and is
  not cited. The canonical smoke above was run once, clean, into a
  fresh dir (`smoke/base_final/`) with the fixed front camera, after
  one launch (guard-smoke5) was correctly refused by the guard's
  preflight (load1 20.47 > 14 from the training chain; box temp
  81 °C — the guard's breaker FIRE during the drill was log-only
  WOULD-KILL, per the founder's 2026-11-22 decision).

## 7. Per-rollout record format

One JSON object per line in `<out>/rollouts.jsonl`, appended + fsync'd
at rollout completion:
`task, episode, variant ("nominal"|"0".."4"), checkpoint_id, status,
suite, policy_mode, prompt, horizon_steps, fps, variant_draws (from the
frozen v1.1 variant file; "nominal" for the nominal scene), sim_success,
secondary_pass, solver_failure, excluded, checks (v1.1 mechanical
detail), objects_final (per-object xyz), actuator_info (welds forced
inactive count, finger geoms enabled), wall_s`.
Trajectory per rollout in `traj_<task>__ep<id>_<variant>.npz`:
per-policy-step `qpos` [6], `ctrl` [6], and `obj_<body>` xyz for every
free-joint body.

## 8. Known limitations / next steps

- Policy service restarts per rollout (~35 s load each) — batching
  multiple rollouts per service instance is a trivial optimisation for
  gate runs, deliberately not built before the grasp decision.
- Wrist-camera placement and the STS3215 slew limit are unfrozen
  instrument parameters (development-subset tuning only, §3.2/§3.4).
- The FL (3-camera repaired) and C-exp control checkpoints were not run:
  controls are meaningless on grasp tasks until §5 is resolved, and
  running them now would burn the control set on a known-broken
  instrument. Cable_clip (no pinch grasp of a rigid object; v1.1
  certified scenes) is the one task family where control runs could
  proceed — flagged for the parent/founder, not run here.
- Noted from standing context (2026-11-22 21:44 CT): a spec-only scene
  path for validator privacy is **not** part of this Stage 1.4 part-1
  build and was not built here.
