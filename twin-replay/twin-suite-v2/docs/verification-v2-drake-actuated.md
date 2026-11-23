# Twin-Suite v2 — Independent Verification: Drake Actuated Rollout Build (Stage 1.2 Part 2a)

**Verdict: CERTIFIED-WITH-FINDINGS**

**Date:** 2026-11-23 · **Verifier context:** independent, separate from the
builder; verified from raw artifacts only (code, model files, result
JSONs, smoke records) plus verifier re-runs in a DGX scratch copy
(`drake_actuated_verify/`, CPU-only, `nice -n 19`, via
`launch_dgx_job.sh` + job guard logonly, rundir
`runs/verify-drake-actuated2`, guard exit 0; mirror FAIL rc=255 as in
the build — logged, no override used). No DGX training disturbed.
**Subject:** `drake-actuated-v2.md`. **Artifact class:** pre-reg §9
(b)/(c) — code + environments.
**Scope:** the actuated-Drake build's claims (actuator/pad parity,
mesh exclusion, grasp battery + divergence, base smoke). The §3.2
certification ruling (policy-outcome agreement on control checkpoints)
belongs to the controls battery, not to this verification.

---

## 1. Actuator parity — CONFIRMED

- MuJoCo vendored model (`zk0-sim-v01-20261115/sim/model/so101_new_calib.xml`),
  read directly: `sts3215` class carries `<position kp="998.22"
  kv="2.731" forcerange="-2.94 2.94"/>` and `<joint damping="0.60"
  frictionloss="0.052" armature="0.028"/>`; the six actuator-level
  `<position class="sts3215" … forcerange="-3.35 3.35">` entries
  (lines 157–162) override the class range. The builder's table is
  exact, including its choice of the actuator-level ±3.35 N·m.
- Drake `actuated_port.py`: `KP_STS3215, KV_STS3215, EFFORT_STS3215 =
  998.22, 2.731, 3.35`, applied to every `JointActuator` via
  `set_effort_limit` + `PdControllerGains(p=…, d=…)`; the recorded
  `actuator_info` in `results/padpos_drake.json` and in the smoke
  record shows all six joints at those values. (The URDF's own joint
  `effort="10"` limits are superseded by this code, as in MuJoCo the
  class value is superseded by the actuator value.)
- Slew limit: `sim_runner_drake.py` has `STS3215_SPEED_RAD_S = 6.16`
  and clips ctrl targets by `6.16 / FPS` per step — the identical
  constant and mechanism as the MuJoCo harness `sim_runner.py`.
- Damping/friction absence is real: `grep -c damping` on
  `so101_drake.urdf` = **0**; the URDF carries transmissions and
  effort/velocity limits only, and no code in `drake_actuated/` sets
  joint damping, friction, or armature equivalents. The §1 logged
  difference (MuJoCo 0.60 / 0.052 / 0.028 with no Drake counterpart)
  stands as an open instrument difference, exactly as reported.

## 2. Pad parity — CONFIRMED

- Dimension-by-dimension comparison of the shipped MuJoCo padded model
  (`scenes-v2/arm/so101_new_calib_padded.xml`, md5
  `0c3f0d1535dc6ea50c6b8ef527a58cf3` — re-verified; `sph_tip` count 0)
  against the Drake tables in `actuated_port.py`
  (`FIXED_JAW_GEOMS` / `MOVING_JAW_GEOMS`): every one of
  `fixed_jaw_box1..7`, `moving_jaw_box1..3`, `finger_pad_fixed`
  (centre (−0.0105, 0, −0.080)) and `finger_pad_moving` (centre
  (−0.009, −0.048, 0.01875), euler −0.13) matches in position, with
  Drake `Box` full sizes exactly 2× the MuJoCo half-sizes.
- Capsule conversion as logged: MuJoCo `fixed_jaw_box3` capsule
  size (radius 0.0011, half-length 0.002) → Drake
  `Capsule(0.0011, 0.004)` (radius, full length). Correct, not
  approximate.
- World-position parity, recomputed from the stored records
  (`results/padpos_drake.json` vs `results/padpos_mujoco.json`,
  frame-130 pose): fixed pad Δ ≈ 0.0015 mm, moving pad Δ ≈ 0.0018 mm,
  gripperframe site Δ ≈ 0.0017 mm. The claimed **0.002 mm** agreement
  holds for all three.
- Tip spheres absent in Drake too: the Drake `finger_geoms` list in
  every result record contains exactly the 10 jaw geoms + 2 pads, no
  spheres — consistent with the shipped MuJoCo model (see the
  fingerpad delta verification for the spheres' trial-and-revert
  history).

## 3. Mesh exclusion — CONFIRMED (reproduced); coverage assessed

- **Reproduction by the verifier:** in the scratch copy,
  `DK_FINGER_MESH=1 … grasp_probes_drake.py --padpos` fails at
  `plant.Finalize()` with `RuntimeError: Collision filter declaration
  references geometry id 165 which has not been registered with this
  filter system` — the claimed mechanism, in the claimed place (the
  builder's "164/165" range; this run produced 165). With the default
  `DK_FINGER_MESH=0` the meshes are skipped and the exclusion strings
  are recorded.
- **Recording:** the exclusion appears in every rollout/probe record
  checked — all three `results/probe_results_*.json` files
  (`mesh_failed`), the padpos record, and the smoke
  `rollouts.jsonl` (`actuator_info.finger_mesh_failed`, all three
  mesh names with the RuntimeError text).
- **What has no Drake counterpart:** the MuJoCo padded model's three
  gripper-part collision meshes — `fixed_jaw_mesh`
  (wrist_roll_follower gripper part) and `moving_jaw_mesh0/1`
  (moving-jaw gripper parts). These are the bulk hulls behind/around
  the fingers. (The original full finger meshes are visual-only in
  both engines' shipped configurations, so they are not a gap.)
- **Coverage:** the surfaces the acceptance battery exercises — the
  two authored pad faces and the jaw boxes that form the pinch — are
  all present at parity (§2), and the battery's contact sets confirm
  pinch contact forms on them in the diagnostic variant (§4). The
  missing part-mesh bulk could matter for incidental contacts a
  policy rollout might make (cube or objects against the finger
  bulk/back, finger bodies brushing fixtures), where Drake will
  under-report contact versus MuJoCo. It does not affect the pinch
  faces themselves. This is a real, bounded instrument difference —
  disclosed in every record, as required.

## 4. THE DIVERGENCE — CONFIRMED by verifier re-run (decisive item)

Verifier re-ran the full Drake battery in the scratch copy
(guard rundir `verify-drake-actuated2`), scripts unmodified:

**Battery as specified (hydro, approach waypoint, timescale ×1):**

| Height | Shove | Grip | Lift / hold z | Hold contacts | Verdict |
|---|---|---|---|---|---|
| +0.005 | 0.1 mm | 0.050 | 0.015 / 0.015 | [world] | FAIL |
| **0.000** | **0.1 mm** | **0.050** | **0.015 / 0.015** | **[world]** | **FAIL** |
| −0.005 | 0.1 mm | 0.050 | 0.015 / 0.015 | [world] | FAIL |
| site-centred 0.000 | 0.0 mm | 0.050 | 0.015 / 0.015 | [world] | FAIL |

At 0.000 m the recorded pinch block gives site z = **0.0346 m** vs the
0.015 m target — the descent stalls ≈20 mm short, exactly as reported;
close contacts at 0.000 are `[gripper, world]` (the fixed finger on
the mat), the gripper reaches the 0.05 rad close target unstalled
(fingers close in air — no cube between them), and the cube never
lifts. Site→pinch-midpoint at stall: 24.2 mm (builder: 24.2 mm).
Minor numeric note: the −0.005 m shove re-ran at 0.1 mm vs the
builder's 0.3 mm; every other number matches to the reported
precision, and no class or mechanism is affected.

**Diagnostic variant (labeled: `PROBE_SKIP_APPROACH=1`,
`PROBE_TIMESCALE=3`):**

| Height | Shove | Grip stall | Lift / hold z | Hold contacts | Verdict |
|---|---|---|---|---|---|
| +0.005 | 6.9 mm | 0.225 | 0.084 / 0.083 | [gripper, moving_jaw] | **PASS** |
| 0.000 | 6.8 mm | 0.231 | 0.084 / 0.084 | [gripper, moving_jaw] | **PASS** |
| −0.005 | 7.2 mm | 0.239 | 0.083 / 0.083 | [gripper, moving_jaw] | **PASS** |
| site-centred | 0.2 mm | 0.231 | 0.083 / 0.083 | [gripper, moving_jaw] | **PASS** |

Pinch-lift + 2 s hold by finger contact alone at all heights —
digit-for-digit the builder's diagnostic table. The capability exists
in Drake; the battery-as-specified transit path is what fails.

**MuJoCo comparison numbers — CONFIRMED from the fingerpad
verification record** (`verification-v2-fingerpad.md` §2, itself an
independent re-run): padded MuJoCo battery at xoff −0.020 — +0.005
FAIL (lift 0.076, dropped), **0.000 PASS** (stall 0.095, lift 0.082,
hold 0.081), **−0.005 PASS** (stall 0.098, lift 0.097, hold 0.101),
hold contacts `[gripper, moving_jaw_so101_v1]` at both passes. The
builder's comparison table and the stall contrast (Drake diagnostic
0.225–0.239 vs MuJoCo 0.095–0.098) are accurate.

**Mechanism assessment (evidence):** the failure is localised to the
approach→descent transit — after the approach waypoint the fixed
finger tip lands on the mat and the descent stalls with
`[gripper, world]` contact; the ×3-timescale-with-approach run in the
builder's records fails identically (static lock, not timing), and a
different transit (direct descent) reaches the target and pinches.
Geometry (§2), actuation (§1), and pinch capability (diagnostic) are
each independently shown not to be the cause. The builder's
diagnosis — hydroelastic contact × executor-path × effort-limited PD
— is the explanation the evidence supports.

## 5. Smoke — record CONFIRMED; renders FINDING (F1, major)

- Record (`smoke/base_drake/rollouts.jsonl`), read directly: base @
  ep411 nominal — `sim_success` **false**, `secondary_pass` **true**,
  `solver_failure` **false**, horizon 296, wall **51.35 s** (≈51 s as
  reported), cubes final at their start poses
  (0.2391, 0.0296, 0.015) / (0.0661, −0.0231, 0.015). Outcome classes
  match the MuJoCo smoke as reported.
- IPC contract — code-read CONFIRMED: `sim_runner_drake.py` drives
  the **same** `harness/policy_service.py` via a subprocess
  JSON-lines pipe, `CHUNK = 50`, frames as `.npy` paths in a shared
  tmp dir, same prompt/state mapping/slew/record format (fsync'd
  `rollouts.jsonl` + `traj_*.npz`, skip-if-recorded).
- **F1 — the renders are blank.** All six image artifacts in the
  smoke directory — `frame0_{front,top,wrist}.png` and
  `frames_tmp/{front,top,wrist}.npy` — are a single flat colour
  (RGB 204, 229, 255; exactly 1 unique colour per image; the three
  PNGs are byte-identical, md5 `fb13c9cbf5…`). The camera *poses* in
  code are the claimed matched viewpoints (front (0.75, −0.55, 0.35)
  fovy 55 → (0.30, 0, 0.05); top (0.3, 0, 0.85) fovy 50; wrist
  site-relative fovy 60, via `RenderColorImage`), but the renderer
  returned background only — no scene geometry in any frame,
  including the last chunk's frames actually sent to the policy
  service. Consequences: (i) the builder's "renders from the three
  matched viewpoints / sample frames produced by the Drake renderer"
  is true of the code path and false of the images; (ii) in this
  smoke the policy was effectively blind, so the outcome-class
  agreement with MuJoCo is weak evidence (base fails and the cubes
  stay unmoved under either input). The smoke still establishes the
  IPC loop, scoring, and record mechanics end-to-end. **The renderer
  must be fixed and a smoke re-run with non-blank frames verified
  before any Stage 2 Drake policy rollout is trusted** — a rollout
  battery run this way would measure a blind policy, not the
  checkpoint.

## 6. No pin/weld; no v1.1 file modified — CONFIRMED in substance, wording finding (F2)

- **Grasp pin/weld: absent, confirmed.** No code in
  `drake_actuated/` pins an object to the gripper or sets object
  poses during dynamics: `sim_runner_drake.py` contains no
  `SetFreeBodyPose`/`set_free_pose` call at all; `grasp_probes_drake.py`
  calls `set_free_pose` only in `reset_arm()` to restore initial
  poses between probes. The v1 pose-pin defect class (F3 of the
  part-1 verification) is retired for the actuated harness.
- **F2 — "no weld anywhere" is literally false, and the report's
  phrasing should be narrowed.** `actuated_port.py` uses
  `plant.WeldFrames` for (a) the arm base to world, (b) static
  fixture bodies to world, and (c) the (default-inactive) mesh
  carrier body. These are standard static scene construction, none
  of them touches a graspable object, and none affects any reported
  result — but a grep-level "no weld anywhere in drake_actuated/"
  claim does not survive contact with the code. The accurate claim
  is: no grasp weld, no pose pin, no object-pose setting during a
  rollout.
- **No v1.1 file modified: CONFIRMED.** No file under the v1.1
  `code/`, `scenes/`, or `validation/` trees has an mtime after
  2026-11-23 02:00 (the actuated build's files are 02:18–02:30);
  the padded model md5 is unchanged (§2).

## 7. Compute discipline — CONFIRMED as reported

Builder batches ran via `launch_dgx_job.sh` + guard (rundirs
`runs/drake-actuated1…5`); the guard logs record the mirror target
set and `MIRROR FAIL rc=255` (Muse VM unreachable from the DGX on
that path), telemetry on-box, and **no** `JOB_GUARD_ALLOW_NO_MIRROR`
override — the verifier's own guard run reproduced the identical
mirror failure. The mirror path remains a Stage 2 blocker per
pre-reg §12 — unchanged, correctly reported.

---

## Findings summary

- **F1 (major, new):** Drake smoke renders are blank (single-colour,
  byte-identical across viewpoints) in both the saved sample frames
  and the frames fed to the policy. Camera-pose code is correct;
  image content is absent. Fix + verified re-smoke required before
  Stage 2 Drake rollouts. (§5)
- **F2 (minor, wording):** "no weld anywhere" overstates — WeldFrames
  is used for base/fixtures/carrier; the grasp-relevant claim (no
  grasp weld/pin, no in-rollout pose setting) is confirmed. (§6)
- **F3 (minor, numeric):** −0.005 m battery shove re-ran 0.1 mm vs
  reported 0.3 mm; no class/mechanism impact. (§4)
- Builder-disclosed open items endorsed: mesh-bulk absence (§3),
  joint-damping gap (§1), guard mirror (§7).

## What the divergence means for the §3.2 standard (evidence only)

The §3.2 standard is cross-engine *policy-outcome* agreement. The
evidence here shows: (1) the two engines' instruments agree on arm,
actuators, and gripper geometry to measurement precision; (2) the
pinch capability exists in both engines; but (3) under the *same
executor protocol*, a grasp approach that succeeds in MuJoCo fails
in Drake for a contact-model × transit-path reason that is global to
that approach geometry, not per-episode. Policy rollouts whose
approaches resemble the battery's will therefore be exposed to the
same lock in Drake, so a Stage 2 Drake battery run as-is would
conflate engine/contact divergence with checkpoint quality — its
outcome agreement (or disagreement) with MuJoCo could not be read as
the §3.2 quantity until the protocol question the builder poses
(diagnostic-variant protocol for Drake, a global executor-path
change, or a contact-configuration revisit) is decided at suite
level, and until F1's blank renders are fixed. That decision, and
the certification ruling, belong to the controls battery / founder —
not to the builder and not to this verification.

## Verdict

**CERTIFIED-WITH-FINDINGS.** Items 1, 2, 3, and 6 are confirmed
(actuator and pad parity to the reported precision; the mesh
RuntimeError reproduced verbatim; no grasp pin/weld; no v1.1 file
touched). Item 4 — the decisive divergence — is confirmed by the
verifier's own re-runs, digit-for-digit in both the
battery-as-specified FAIL (stall ≈20 mm short, site z 0.0346, fingers
closing in air on `[gripper, world]` contact) and the diagnostic
pinch-lift PASS, against MuJoCo numbers confirmed from the fingerpad
verification record. Item 5's smoke record and IPC contract are
confirmed, with the major new finding F1 (blank renders) that must
be closed before Stage 2 Drake policy rollouts are trusted. Nothing
found overturns the build report's central, honestly-stated
conclusion: the actuated capability is built, and grasp parity under
the battery as specified is **not** established.

---

## Delta verification — F1/F2 renderer fix (2026-11-23)

**Verdict (delta scope): CERTIFIED**

**Date:** 2026-11-23 · **Verifier context:** independent, separate from
both the original builder and the fix builder; verified from raw
artifacts only (code, the pre-fix verifier scratch copy
`drake_actuated_verify/`, evidence PNG/`.npy` files, smoke records)
plus verifier re-renders on the DGX (CPU-only, `nice -n 19`, scene
builds in `/tmp`; no DGX training disturbed). **Scope:** only the
2026-11-23 fix note appended to `drake-actuated-v2.md` (findings F1
and F2 of the verification above). The original verdict and text
above are unaltered; the §3.2 / grasp-divergence questions are out of
scope for this delta and unchanged by it.

### 1. Root cause — CONFIRMED, derived independently

- The pre-fix code survives verbatim in the verifier scratch copy
  (`drake_actuated_verify/sim_runner_drake.py`): `_lookat_R` built
  `z = -fwd`, `x = cross(up, z)`, `y = cross(z, x)` — i.e. the
  camera's +z axis pointed *away* from the target. Its own docstring
  states the correct Drake fact ("Drake camera frame: +z out of the
  screen (backwards), +x right, +y down") and then builds the
  OpenGL/MuJoCo rotation anyway. Under Drake's x-right / y-down /
  z-forward convention that points every camera away from the scene.
- The fixed implementation was derived, not trusted: with
  `z = fwd` (target − position, normalised), `x = normalize(cross(z,
  up))`, `y = cross(z, x)`, the matrix is a proper rotation
  (det = +1, orthonormal to machine precision, verifier-computed for
  front/top/axis cases); `x × y = z` holds, so the frame is
  right-handed in Drake's convention. Spot checks: the front
  camera's z column equals the normalised (0.75, −0.55, 0.35) →
  (0.30, 0, 0.05) direction (−0.583, 0.713, −0.389); the top view
  gives columns x = +X, y = −Y (image down), z = −Z (forward, down),
  i.e. image right = +x and image up = +y — the claimed v1.1 topcam
  orientation. A camera looking along +X with up +Z gets image
  right = −Y, as required.
- **Depth claim reproduced exactly.** Verifier re-render of the
  ep411 front view with the *exact* pre-fix `_lookat_R` (copied from
  the scratch copy): colour image = exactly 1 unique colour,
  (204, 229, 255); depth image finite-pixel fraction = **0.0000**.
  With the fixed rotation: depth finite fraction 0.7794 (min depth
  0.404 m), colour unique colours > 1,200 in the verifier's
  default-pose build. The renderer, SceneGraph registration and
  camera positions were never the problem; the orientation was.

### 2. Frame content — CONFIRMED by direct inspection + measurement

- The verifier opened and looked at all six evidence PNGs
  (`render_fix_evidence/drake_{frame0,mid130}_{front,top,wrist}.png`).
  Frame 0: yellow arm entering from the left (front/top), beige
  enclosure walls, dark mat, green cube on the mat in all three
  views, purple cube visible as a small patch beside/behind the
  gripper in front and top views, wrist view dominated by the yellow
  wrist/gripper with the green cube above it. Frame 130 front: the
  arm has descended with the gripper at the green cube, as claimed.
  Frame-130 wrist shows arm + mat only — consistent with the fix
  note's own table for that cell (no cube claimed there).
- Verifier-measured unique-colour counts on the PNG files match
  `render_proof.json` **exactly, all six cells**: frame 0 front
  1,097 / top 858 / wrist 1,286; frame 130 front 1,252 / top 1,074 /
  wrist 1,168 (claimed range ~850–1,300; pre-fix = 1). The `.npy`
  twins give identical counts.
- Side-by-side composites (`sidebyside_{front,top,wrist}.png`)
  inspected: compositions match the MuJoCo side in all three views —
  arm from the left in front/top with cubes in the same relative
  positions, wrist view dominated by the yellow/black gripper with
  the green cube above centre. Residual differences are the
  disclosed ones (flat Drake lighting vs MuJoCo gradient/shadows;
  clear colour above the horizon in front view).

### 3. Visual clones + no collision change — CONFIRMED

- Full diff of `actuated_port.py` and `sim_runner_drake.py` against
  the pre-fix verifier scratch copy: the *only* changes are
  (a) `_lookat_R` (above), (b) `RegisterVisualGeometry`-only
  additions in `actuated_port.py` — the `.obj` clones registered at
  the inspector-cached pose `_insp0.GetPoseInFrame(_gid)` with the
  original's perception diffuse colour, the originals stripped via
  `RemoveRole` of the illustration + perception roles only, and
  floor/mat/wall visual twins registered at **identical
  RigidTransforms and shapes** to their collision counterparts
  (floor Box(4, 4, 0.05) at z = −0.026; mat Box at (mx, my, −st/2);
  walls Box(bx, by, 0.06) at z = 0.03) in the claimed colours —
  (c) the `frame_content` record statistics in `sim_runner_drake.py`,
  and (d) comments/docstrings. **No `RegisterCollisionGeometry`
  call, friction property, contact parameter, gain, or geometry
  table differs between the verified build and the fixed build.**
  `grasp_probes_drake.py` is byte-identical to the pre-fix copy.
- Clone count verified three ways: `render_proof.json`
  `arm_visual_obj_clones` = 17 with `arm_visual_strip_failed: []`;
  the fixed smoke record's `actuator_info.arm_visual_obj_clones` =
  17; the verifier's own fresh `DrakeScene` build printed
  `clones: 17, strip_failed: []`.

### 4. Re-smoke — CONFIRMED

- `smoke/base_drake_fixed/rollouts.jsonl`, read directly:
  `sim_success` **false**, `secondary_pass` **true**,
  `solver_failure` **false**, wall **54.5 s**, cubes final at their
  start poses (0.2391, 0.0296, 0.015) / (0.0661, −0.0231, 0.015) —
  the claimed FAIL / secondary PASS / solver False at ~54 s.
  Guard rundir `runs/drake-renderfix3` exit code 0, with the render
  proof and re-smoke output in `guard/command.out` matching the
  records (mirror FAIL rc=255 as before, no override — unchanged
  Stage 2 blocker, correctly carried over).
- `frame_content` is present in the record with the claimed values:
  chunk 0 unique colours front 1,072 / top 873 / wrist 1,272;
  minimum non-clear fraction over all chunks front 0.7812 / top
  1.0 / wrist 0.5815. The verifier independently measured the
  policy-fed final-chunk `.npy` frames in
  `smoke/base_drake_fixed/frames_tmp/`: front **1,266** / top
  **1,095** / wrist **1,332** unique colours — exactly the fix
  note's numbers; none is blank.
- The blind original is retained unchanged: `smoke/base_drake/`
  mtimes 2026-11-23 02:30–02:31 (pre-fix), its three frame-0 PNGs
  still md5-identical to each other (`fb13c9cb…`), and all six of
  its image artifacts still measure exactly 1 unique colour.

### 5. F2 wording — CONFIRMED accurate

- Grep of `drake_actuated/*.py`: `WeldFrames` occurs exactly three
  times, all in `actuated_port.py` — (a) arm base body to world
  (line 248), (b) the finger-mesh carrier body to its finger body
  (line 369; inside the default-inactive `DK_FINGER_MESH` path),
  (c) static fixture bodies to world via `static_body` (line 419).
  No weld touches a graspable object.
- `sim_runner_drake.py` contains **no** `SetFreeBodyPose` /
  `set_free_pose` call at all. `set_free_pose` is defined in
  `actuated_port.py` and called only in `grasp_probes_drake.py`'s
  `reset_arm()` to restore initial poses between probes; the one
  `SetFreeBodyPose` in `actuated_port.py` sets initial object poses
  at scene construction. The appended wording — no grasp pin/weld,
  no object-pose setting during a rollout — matches the code.

### 6. No v1.1 file modified — CONFIRMED

`find` over the v1.1 `code/`, `scenes/` and `validation/` trees for
files with mtime after 2026-11-23 02:00 returns **0 files** (the fix
build's own files are 02:51–03:04, all inside `drake_actuated/`).

### Delta verdict

**CERTIFIED.** Every load-bearing claim in the F1/F2 fix note was
reproduced from raw artifacts or by the verifier's own re-render:
the root cause and the corrected rotation (derived and depth-tested
in both orientations), the frame content (inspected visually and
measured to the exact claimed counts), the 17 visual-only clones
with a collision-geometry diff of zero, the sighted re-smoke record
with non-blank policy-fed frames, the narrowed F2 wording, and the
untouched v1.1 trees and blind-smoke evidence. F1 and F2 are closed.
This delta certification covers the renderer fix only; it does not
re-open or alter the original findings, the grasp-battery
divergence, or the guard-mirror Stage 2 blocker recorded above.
