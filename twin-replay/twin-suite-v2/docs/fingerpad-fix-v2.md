# Twin-Suite v2 — Stage 1 Fix: Finger-Pad Collision Geometry (SO-101 gripper, MuJoCo)

**Date:** 2026-11-22 · **Builder:** Muse subagent (finger-pad fix) ·
**Governing:** twin-suite-v2-preregistration-20261122.md §4; Amendment 2 §A2.2
(solvability executors run in the gate twin) ·
**Responds to:** rollout-harness-v2.md §5 (grasp blocker) and
verification-v2-rollout-harness.md §4 (blocker CONFIRMED: jaw closes in a
downward/backward arc, moving jaw never contacts the cube, closing shoves
the cube ~14 mm, no pinch-lift without the weld).
**Code (DGX):** `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/`
— `scenes-v2/arm/so101_new_calib_padded.xml` (+ `scenes-v2/arm/assets/`),
`scenes-v2/block_stack__ep{411,453}{,_v0}.xml`,
`harness/sim_runner_padded.py`, `harness/fingerpad_probes.py`,
`harness/grasp_probe{,2,3}_padded.py`, `harness/scripted_pass.py`,
`harness/run_smoke_padded.sh`.
**No v1.1 file and no vendored arm file was modified.** The v2 harness's
original `grasp_probe.py` had its import line briefly flipped by a sed
error during this build and was restored byte-for-byte the same session
(verified: all original harness files again import `sim_runner`).

## Verdict up front

**The grasp is fixed for a fingers-down, face-centred approach — and only
for that approach.** With the padded model, a scripted executor that
centres the cube between the pad faces, pre-closes to 0.7 rad, and closes
to a 0.05 rad servo target (stalling at ≈0.10–0.12 rad on the cube)
**pinch-lifts the 30 mm cube and holds it ≥2 s by finger contact alone**
at 2 of 3 descent heights (details §4). The moving jaw is in the cube's
contact set in every successful phase, with measured pad normal forces
of 2.5–10.7 N per contact point.

Two honest limits, both evidenced below:

1. **The original site-centred probes still fail.** Re-running the
   shipped probe battery as-is (recorded pose; IK site-centred grasps;
   close-sweep at the recorded pose) does **not** produce a pinch-lift:
   the gripperframe site is ≈20 mm off the pinch centre and at the
   recorded wrist orientation the finger faces are diagonal, so closing
   still sweeps the cube (−13 mm) and the moving jaw still never
   contacts. The pads give the fingers opposing faces; they do not move
   the site or change the recorded approach geometry.
2. **The full scripted stack pass is NOT yet proven.** The executor
   grasps, lifts and carries cube_a, and twice placed it in contact
   with cube_b's top, but the place site found by IK requires
   wrist_flex ≈107°, beyond the model's ±95° joint limit, so the
   actuated arm stalls ≈25 mm short of the place target and the cube
   topples on release. Final v1.1 mechanical score: **FAIL**
   (secondary PASS). Recorded as A2.2 evidence in §5: grasp
   solvability proven, full-pass solvability not proven by this
   executor.

## 1. Pad sourcing (honest lineage)

- The vendored `so101_new_calib.xml` (onshape-to-robot export of the
  TheRobotStudio SO-ARM100) carries only the two full finger **meshes**
  as collision geoms. The project's own DGX sim assets also contain the
  **MuJoCo Menagerie edition of the same TheRobotStudio model**
  (`zk0-sim-v01-20261115/menagerie/robotstudio_so101/so101.xml`), which
  ships **dedicated gripper collision geoms the vendored export
  dropped**: `fixed_jaw_box1..7`, tip capsules/spheres,
  `moving_jaw_box1..3`, and split gripper-part collision meshes
  (`wrist_roll_follower_so101_gripper_part0_v1.stl`,
  `moving_jaw_so101_gripper_part{0,1}_v1.stl`). Body frames, geom
  positions and joint ranges in the Menagerie file match the vendored
  file element-for-element for both finger bodies, so the geoms were
  ported **verbatim** (pos/size), with only the class changed
  (`collision_gripper*` → the vendored file's `collision` class,
  group 3) so the v2 harness's finger-contact enabling applies.
- Real-finger dimensions cross-checked from the source STLs (mm):
  moving jaw mesh extent 22.3 × 92.0 × 48.0; its gripper part (the
  finger beyond the motor) spans y −82.1…−22.7 (≈59 mm long);
  fixed-follower gripper part z 37.9…105.7 in its mesh frame.
  Project's measured jaw opening 97.7 mm (project records) is
  consistent with the open gap the padded model presents
  (41 mm between pad faces at the cube's height at 1.2 rad — the
  faces are part-way along the fingers, not at the tips).
- **Collision-responsibility change (declared):** in the padded copy,
  the two original full-mesh collision geoms on the finger bodies are
  set `group=2 contype=0 conaffinity=0` (visual only). Rationale,
  measured: with the full meshes enabled, the largest cube contact
  force at grasp was an unnamed mesh contact (9.57 N) on a **diagonal
  hull surface** — the wedge that ejects the cube — while the pad
  faces carried 2.5–10.7 N. The Menagerie model itself does not use
  the full follower/moving meshes as gripper collision (it uses the
  boxes + part meshes), so this follows the upstream design.

### Fix note — 2026-11-22 (F1/F2, responding to verification-v2-fingerpad.md)

This note is appended, dated, and does not silently edit the text above;
where it corrects §1, the correction below supersedes the original
wording.

**F1 — tip spheres omitted from the port (disclosed; resolution:
disclosure-only after a measured battery change, spheres NOT ported
in the shipped copy).** The Menagerie source also carries six 0.75 mm
tip spheres — `fixed_jaw_sph_tip1..3` (size 0.00075; pos
(−0.0081, 0, −0.101), (−0.0081, 0.0035, −0.0975),
(−0.0081, −0.0035, −0.0975)) and `moving_jaw_sph_tip1..3` (size
0.00075; pos (−0.012, −0.078, 0.0192), (−0.012, −0.0745, 0.0225),
(−0.012, −0.0745, 0.0155)) — which the original port omitted without
disclosing it. (Also for the record: the upstream part-mesh geoms are
unnamed upstream (`collision_gripper_mesh`) and were given names in
the copy — `fixed_jaw_mesh`, `moving_jaw_mesh0/1`.) On 2026-11-22
the six spheres were ported verbatim (pos/size, class `collision` as
for the boxes) into a trial copy and the decisive battery subset was
re-run on the DGX (CPU-only, `nice -n 10`, scripts unmodified;
`harness/fingerpad_probes.py` xoff −0.020 and
`harness/grasp_probe3_padded.py`):

| Case | Verified (no spheres) | With spheres | Class |
|---|---|---|---|
| +0.005 m | shove 11.0 mm, grip 0.096, lift 0.076, hold 0.015 — FAIL (dropped) | shove 10.3 mm, grip 0.116, lift 0.079, hold 0.077 — PASS | **changed** |
| 0.000 m | shove 12.5 mm, grip 0.095, lift 0.082, hold 0.081, contacts [gripper, moving_jaw] — PASS | shove 11.5 mm, grip 0.115, lift 0.083, hold 0.082, contacts [gripper, moving_jaw] — PASS | same |
| −0.005 m | shove 13.9 mm, grip 0.098, lift 0.097, hold 0.101 — PASS | shove 13.4 mm, grip 0.114, lift 0.086, hold 0.086 — PASS | same |
| Probe 3 (site-centred) | −13.3 mm shove, no lift | −13.3 mm shove (x 0.2407 → 0.2274), contacts [gripper, world] only, z 0.0149 — no lift | same |

The spheres change an outcome class (+0.005 m flips FAIL → PASS),
shift the grip stall systematically (+0.018…+0.021 rad at every
height), and move the −0.005 m hold by −15 mm. Under the fix pass's
pre-stated rule (any outcome-class change or material number move →
revert), the spheres were **reverted**: the shipped padded copy is
byte-identical to the verified model (md5
`0c3f0d1535dc6ea50c6b8ef527a58cf3`, `sph_tip` count 0), and a
post-revert re-run of `fingerpad_probes.py` reproduced the verified
numbers exactly (11.0 / 12.5 / 13.9 mm; 0.076 / 0.082 / 0.097 m lift;
hold 0.081 / 0.101 m; FAIL / PASS / PASS). The omission therefore
stands as a disclosed lineage gap only: the certified battery was run
on a model whose Menagerie port comprises the jaw boxes/capsule and
part meshes listed in §1, **without** the six tip spheres. No other
geometry, parameter value, v1.1 file, or vendored file was touched
(vendored `so101_new_calib.xml` md5 `64092b46…` unchanged).

**F2 — "element-for-element" corrected.** The §1 sentence claiming
that "body frames, geom positions and joint ranges in the Menagerie
file match the vendored file element-for-element for both finger
bodies" is overstated and is superseded by this note. What matches
element-for-element is the **jaw collision geoms** (pos/size of
`fixed_jaw_box1..7`, the tip capsule, and `moving_jaw_box1..3`) and
the joint ranges. The padded copy is a copy of the **vendored** model
and follows the vendored frames where the two sources differ — in
particular the `gripperframe` site pos is the vendored
(−0.0079, −0.000218121, −0.0981274), **not** Menagerie's
(0.012, −0.000218, −0.098127); several body quats also differ in
representation between the two files. This was the correct choice
(copy of the vendored model) and has no functional impact; only the
original sentence was inaccurate.

## 2. Authored pads (instrument parameters, v2 parameter-table lineage)

Two box pads consolidate the upstream stepped faces into flat opposing
faces. Every dimension labelled per the audit discipline:

| Pad | Geom (body-local) | Source of each parameter |
|---|---|---|
| `finger_pad_fixed` (body `gripper`) | box, centre (−0.0105, 0, −0.080), half-size (0.0025, 0.010, 0.023) | Face plane local x = −0.008 = upstream `fixed_jaw_box4` face **[MEASURED from upstream geoms]**; z extent −0.057…−0.103 = upstream boxes 4–7 span **[MEASURED]**; y half 0.010 = upstream box7 half-width **[MEASURED]**; thickness 5 mm **[ESTIMATE]** (consolidates the stepped boxes) |
| `finger_pad_moving` (body `moving_jaw_so101_v1`) | box, centre (−0.009, −0.048, 0.01875), half-size (0.003, 0.030, 0.006), `euler="0 0 −0.13"` | y extent −0.018…−0.078 = upstream moving mesh/box tip span **[MEASURED]**; z 0.01875 and half-thickness 0.006 = upstream boxes 2/3 **[MEASURED]**; face plane local x = −0.006 **[ESTIMATE]** (2–4 mm proud of upstream tip-box faces −0.0083…−0.0103, to present a face along the full finger, not only the tip); pre-tilt −0.13 rad about the grip axis **[MEASURED-EST]**: first guess −0.35 rad left a measured face ramp (face x drift 13 mm over 58 mm of face height at grip 0.13 rad); +0.22 rad correction made the face vertical (corner x constant 0.240/0.246 over z 0.011…0.071, re-measured) |

Friction on all finger surfaces: (1.0, 0.005, 0.0001) — the parameter
table's finger value, applied by `prepare_model` exactly as in v2
Stage 1.4. Actuators, limits, gains: **identical to the vendored
model** (nothing in the padded copy changes them).

Template regeneration: `scenes-v2/block_stack__ep{411,453}.xml` and
`_v0.xml` are byte copies of the v1.1 scenes with only the `<include>`
swapped to the padded arm copy. `sim_runner_padded.py` is
`sim_runner.py` with `SCENES` pointed at `scenes-v2/` (one line).

## 3. Why the first attempts failed (evidence trail, condensed)

- Upstream geoms alone: moving jaw still never contacted; cube shoved
  −6 mm. The upstream boxes are tip/root segments; the face area at
  cube height is too small.
- Pads without tilt correction: closing wedged the cube out
  (face ramp ≈24° in the x–z plane; cube ejected downward/sideways
  after 2–4 cm of lift, jaw qpos snapping 0.135 → 0.08 as it escaped).
- Pads with the full finger meshes still colliding: same ejection —
  the diagonal mesh hulls (9.57 N measured) overpower the flat faces.
- Site-centred approaches (any pad version): the gripperframe site
  sits ≈20 mm from the pinch centre (fixed face passes through the
  site; moving face at +18…+27 mm from it at grasp closure), so a
  site-centred descent puts the fixed finger **inside** the cube's
  footprint (descent sink 2–3 mm into the mat, +2 mm shove) and the
  moving face never arrives.

## 4. Acceptance battery — before / after

Executor for (b) (`harness/fingerpad_probes.py`, GENERATION:
ORACLE-GUIDED): 6-DOF IK (site position + gripper body aligned to
world — fingers down, pinch axis world x), site offset x −0.020 m
centring the cube between the measured pad faces, pre-close 0.7 rad
during descent, close to 0.05 rad target, lift in 1 cm Cartesian
stages with grip ctrl frozen at stall −0.02 rad, 2 s hold.

### (a) Recorded-pose probe (`grasp_probe_padded.py`, frame 190)

| | Before (v1.1 fingers) | After (padded) |
|---|---|---|
| Cube contact at recorded pose | none | none |
| After recorded close + lift | z_max 0.0149 m, left behind | z_max 0.0149 m, left behind |

Unchanged — as expected: this probe's failure is the known FK↔image
pose offset (≈7.4 cm), not finger geometry. Pads do not repair pose
estimation, and this report does not claim they do.

### (b) Scripted IK-centred grasps, three heights (`fingerpad_probes.py`, xoff −0.020)

| Descent height (site z − cube z) | Before | After: shove on close | Grip stall | Lift z | Hold 2 s | Verdict |
|---|---|---|---|---|---|---|
| +0.005 m | no moving-jaw contact; shove −14 mm; no lift | 11.0 mm | 0.096 rad | 0.076 m | dropped during hold | FAIL (lifted, not held) |
| 0.000 m | same | 12.5 mm | 0.095 rad | 0.082 m | **0.081 m, contacts [gripper, moving_jaw] only** | **PASS** |
| −0.005 m | same | 13.9 mm | 0.098 rad | 0.097 m | **0.101 m, contacts [gripper, moving_jaw] only** | **PASS** |

- **Pinch-lift: PASS at 2/3 heights** — cube held ≥2 s by finger
  contact alone (no `world` contact), no weld (welds asserted
  inactive in the scripted-pass runner; probes never enable them).
- **Moving jaw in contact set: YES** in every close/lift/hold phase
  above (before: never, in any probe).
- **Shove:** the 11–14 mm displacement is measured from pre-close to
  post-close and occurs **after first contact**: at pre-close the
  moving face is ≈9 mm from the cube's face (cube displacement before
  contact ≈ 0 — nothing touches it during descent at this offset;
  descent displacement ≤3 mm), and closing then centres the cube
  against the fixed face. It is not the original failure mode
  (sweep-away with no resulting pinch), but it does **not** meet a
  strict "few mm total displacement" reading. The trade-off is
  measured: at xoff −0.010 (cube pre-seated against the fixed face)
  shove falls to 1.7–3.6 mm, but the moving face then cannot reach
  the cube (grip reaches the 0.05 target unstalled, no pinch). The
  gap geometry (fixed face at site −0.002 m, moving face at
  site +0.024 m at stall) leaves a ~9–12 mm centring travel for a
  30 mm cube unless the descent pre-seats it to within ~2 mm of the
  fixed face without the fixed tip (site +≈0.02 m in x at cube-top
  height) striking the cube's top — a placement-precision problem
  for the executor, recorded here rather than tuned around.
- Original shipped probe 3 (site-centred, recorded-orientation IK),
  re-run on the final padded model for direct comparability: still
  FAIL — cube shoved −13 mm on close, moving jaw never contacts,
  no lift. The fix is in the fingers' capability, realised only by
  an approach that uses the faces as faces.

### (c) Close-sweep at recorded pose (`grasp_probe2_padded.py`, frame 130)

Before: gripper follows ctrl to −0.05 rad, cube swept y 0.030 → 0.010
(20 mm), never lifts, moving jaw absent from contacts.
After: cube starts already in contact with **both** jaws at this pose;
closing 1.45 → −0.05 rad moves the cube ≈2–7 mm total (y 0.0197 →
0.0120, x 0.2391 → 0.2460) instead of 20 mm, and the moving jaw
registers in the contact set at the start and at 0.162 rad. No lift
(recorded pose; see (a)).

## 5. Scripted reference pass — A2.2 solvability-executor record

`harness/scripted_pass.py`, block_stack ep411 **nominal**,
GENERATION: ORACLE-GUIDED. Grasp cube_a (the ep411 green cube at
(0.239, 0.030)), lift, translate, place on cube_b, release, settle;
scored by the v1.1 mechanical criteria (`replay.evaluate`, read-only).

- Grasp: stall 0.095 rad, contacts [gripper, moving_jaw, world].
- Lift: cube_a carried to z 0.082 m, contacts [gripper, moving_jaw]
  only — **the pinch-grasp is solvable in the gate twin without any
  weld.**
- Place: IK for the place site (0.058, −0.018, 0.057) converges
  kinematically (err 1.3 mm) only with wrist_flex ≈107°, **beyond
  the model joint limit ±95°**; the actuated arm stalls ≈25 mm
  short. Twice, cube_a was set down in contact with cube_b's top
  (z 0.033–0.037, contacts include cube_b) but off-centre by
  ≈25–30 mm, and toppled on release (once pushing cube_b ≈9 mm).
- **Final score (best-documented run): sim_success FALSE,
  secondary_pass TRUE, solver_failure FALSE.** Final objects:
  cube_a (0.063, −0.050, 0.015), cube_b (0.064, −0.015, 0.015) —
  side by side, not stacked.
- **A2.2 record:** this scene's *grasp* is solvability-proven by an
  executed pass trajectory segment (grasp + lift + carry). The full
  task pass is **not** solvability-proven by this executor; the
  blocker is executor/arm-workspace (wrist_flex limit at the place
  pose in every IK branch found), not the finger geometry. A
  place approach with a tilted (non-aligned) wrist — as the recorded
  demo itself uses — is the obvious next executor iteration; it was
  not attempted in this fix's scope.

## 6. Regression smoke (padded template)

`harness/run_smoke_padded.sh` under `launch_dgx_job.sh` + job guard
(logonly, JOB_LOCK=0, mirror target set — see §7 note): base
checkpoint @ ep411 + ep453, nominal + v0, via `sim_runner_padded.py`.

| Episode | Variant | sim_success | secondary | solver_failure | wall |
|---|---|---|---|---|---|
| 411 | nominal | False | **False** | False | 61.6 s |
| 411 | v0 | False | False | False | 71.7 s |
| 453 | nominal | False | True | False | 76.3 s |
| 453 | v0 | False | True | False | 64.2 s |

Guard exit code 0, `SMOKE_PADDED_DONE`. Comparison with the unpadded
smoke (`smoke/base_final/`): success 0/4 in both (base fails every
rollout, as expected — its sealed floor is 0/100); solver failures 0/4
in both; horizons identical (296 / 289 = recorded length + 20 settle);
records, trajectory npz, sample frames and scoring all produced in
the same format (`actuator_info.finger_geoms` now lists the upstream
jaw geoms + both pads — the expected instrument change). **One
difference:** ep411 nominal secondary flipped True → False. The cubes
finish unmoved at their start poses in both runs, so the flip is in
the near-miss criterion path, most plausibly because the policy now
sees padded fingers (wrist/front renders) and the arm's contact
dynamics changed — i.e. the instrument change is visible to the
policy, which is expected and is precisely why this model is a new
suite version needing its own certification, not a silent harness
tweak. Nothing else broke: records, scoring and solver detection are
unchanged in kind.

## 7. Notes / caveats

- Guard mirror: `JOB_GUARD_MIRROR_TARGET` was set to the off-box
  precedent target (`hatch@100.64.115.122:…/guard-mirror-fingerpad`);
  the guard log records MIRROR FAIL rc=255 — ssh to that host is
  connection-refused (the Muse VM currently has no sshd listening /
  no direct tailnet address; it is reached via the runtime proxy).
  No no-mirror override was used; telemetry is in the on-box guard
  rundir `smoke/guard-padded1/guard/`. Flagging for the parent: the
  off-box mirror path itself appears broken in this environment.
- All probe/smoke work ran CPU-only, `nice -n 10`, under the job
  guard for the sustained smoke; DGX training was not disturbed
  (guard preflight/telemetry in the rundir).
- The padded model is a **new suite artifact** (v2 dev), not a
  change to the frozen v1.1 suite (param_hash a4eb3f8c3611894d
  untouched). Whether it enters the certified suite is a §11
  change-control decision for the founder/verifier, with this
  battery as the acceptance evidence.


**Erratum — 2026-11-23 (D2, from the delta pass in `verification-v2-fingerpad.md`):** in the Fix note above, the grip-stall shift with the tip spheres is **+0.016…+0.020 rad**, not "+0.018…+0.021 rad" — the cases are +0.020 rad (+0.005 m: 0.116 − 0.096), +0.020 rad (0.000 m: 0.115 − 0.095) and **+0.016 rad** (−0.005 m: 0.114 − 0.098); the systematic-shift claim itself stands.
