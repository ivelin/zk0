# Twin-Suite v2 — Stage 1.2 (part 1): Drake Port + Engine-Equivalence Diagnostics

**Date:** 2026-11-22 · **Builder context:** Stage 1.2 build agent (separate
verifier required per pre-registration §9 — this report is a builder
deliverable, NOT a certification).
**Governing docs:** `twin-suite-v2-preregistration-20261122.md` §3.2 (frozen,
commit bb11d20) and `sim-gate-design-memo-20261122.md`.
**Scope delivered:** programmatic Drake port of the four precision tasks'
twin scenes (block_stack, eye_drops_to_basket, ring_insert, tool_insert) +
kinematic-replay equivalence diagnostics on the development subset,
nominal variant. This is port sanity evidence for §3.2 — it is **not** the
§3.2 certification standard, which is policy-outcome agreement on the
control checkpoints with bounds set by the §7 Bounds Amendment. No v1.1
file was modified (frozen suite, param_hash `a4eb3f8c3611894d`).

**Workdir (DGX):** `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake/`
— `drake_port.py` (builder), `drake_replay.py` (kinematic replay + v1.1
criteria), `mujoco_baseline.py` (MuJoCo re-runner), `compare.py`,
`results/` (point-contact Drake), `results_hydro/` (hydroelastic Drake),
`mujoco_results/`, per-batch guard logs under `../runs/`.

## 1. Inventory

**Scene JSONs** (`zk0-study001-twin-20261122/scenes/`, one per episode +
`<key>_variants.json` + variant XMLs): the raw JSON encodes, per episode,
Builder B object entries keyed by colour/role name with
`pose_xyz_yaw = [x, y, z, yaw_rad|null]`, per-object `confidence`,
`sigma_xy_cm` / `sigma_yaw_deg`, detection method, and an FK-anchor
cross-check block; finals are separate `<name>_final` entries. Fixtures are
either global (`scenes/fixtures.json`: eye platform/basket, tool tray/slot,
ring peg pixel positions per episode) or repeated per episode. **The poses
that actually reach a scene are not the raw JSON values**: v1.1
`build_scenes.load_episode_poses` applies the global FK-affine frame
correction, detection-pixel primacy for block_stack, and parallax offsets
(eye carton −0.014/−0.093 m, etc.). The Drake builder therefore consumes
poses through that same v1.1 adapter — the engine-independent input the
pre-registration §3.2 requires — rather than re-interpreting the JSONs.
Variant files encode K=5 draws per episode (seed 20261122): per-object
pose jitter (clipped Gaussian, σ from Builder B), mass draws from the
sourced ranges in `twin_params.MASS_RANGES`, and friction multipliers
U(0.8, 1.2); k=0 is nominal (empty draws). Colours are per-episode data
(block cube keys are `cube_<colour>`; ring carries an explicit `color`
field) and are rendered through v1.1's `COLOR_RGBA` table in both engines.

**Arm model (MuJoCo):** `zk0-sim-v01-20261115/sim/model/so101_new_calib.xml`
(onshape-to-robot SO-101, STS3215 joint classes), included into every
scene; STL assets vendored under `scenes/assets/`. In v1.1 replay the arm
is purely kinematic (qpos prescribed from the recorded gt trajectory) and
all arm geoms have contype/conaffinity = 0 — grasp is the declared weld
device only.

**Drake assets already on the box (Nov-15 eval work,
`zk0-sim-eval2-20261115/`):** a converted Drake URDF
`assets/SO101/so101_drake.urdf` (model instance `so101_new_calib`, same
six joint names), plus probe/test code that established the pattern used
here (hydroelastic-with-fallback contact, PD gains, TCP offset
(−0.0079, −0.000218121, −0.0981274) — which is exactly MuJoCo's
`gripperframe` site offset, and Drake's URDF carries a matching
`gripper_frame_link`). The port reuses the URDF as-is; no new arm
conversion was needed. Drake venv:
`zk0-sim-eval2-20261115/drake/venv` (pydrake; note: this build has **no
`Torus` shape and no `SpatialVelocity` binding** — see §2).

**FK verification (port smoke):** with both arms set to episode 411's
recorded frame-0 pose, Drake and MuJoCo agree on the gripper body pose
and the gripperframe site position to 4 decimal places (position
(0.1183, 0.0056, 0.0890), site (0.1472, 0.0085, −0.0050), identical
quaternion). The two arm models are kinematically the same model.

**MuJoCo baseline reproducibility:** the diagnostics re-ran v1.1
`replay.py` unmodified (in-process monkeypatch only to also record final
quaternions, which v1.1's stored results omit). Re-run finals are
**bit-exact** against the stored v1.1 validation results (block 411
checked to full float precision; event lists match on spot checks).

## 2. Port design

One programmatic builder (`DrakeScene` in `drake_port.py`) for all
episodes/variants/tasks — no hand-built per-episode files. It reads
poses/fixtures/draws via the v1.1 functions `load_episode_poses`,
`load_fixtures`, `make_draws`, and all geometry/mass/friction values from
`twin_params.py`, then builds a Drake `MultibodyPlant` + `SceneGraph`
(discrete, 1 ms step):

- Arm from the Nov-15 URDF, base welded to world, **arm proximity role
  removed** (mirrors MuJoCo contype=0). First implementation used
  `AssignRole(kReplace, empty)` which silently does **not** remove the
  role in this Drake build (verified: 17/17 arm geoms stayed proximity,
  and the live gripper physically shoved objects — see §4, eye 1746);
  `SceneGraph.RemoveRole` is the working API (verified 0/17 after fix).
  All results below are from the fixed build.
- Fixtures (platform, basket, board, peg, tray incl. the v1.1 slot
  recess construction) as welded rigid bodies with MuJoCo-identical
  local geometry, so contact body names match MuJoCo's exactly; mat,
  walls and floor on the world body. MuJoCo's infinite floor plane is a
  4×4×0.05 m box with top at z=−0.001 (Drake `HalfSpace` reports no
  contact data in this build).
- Objects: cubes/carton as boxes; ring as **12 capsule geoms on one
  body**, exactly MuJoCo's approximation (no Torus in this Drake);
  tool as one capsule (Drake capsule axis is local +z, registered
  rotated into MuJoCo's local +x; body frame unchanged). Ring/tool
  inertia are `UnitInertia.SolidCylinder` approximations with exact
  masses (logged approximation).
- Kinematic-arm mode: arm joint positions prescribed per frame from the
  recorded gt (degrees→radians; gripper joint from v1.1's closedness
  mapping, clipped to URDF limits). The replay harness
  (`drake_replay.py`) mirrors v1.1 `replay.py` step for step: 0.25 s
  pre-settle + build-defect check, the dwell/planar-shape-distance
  grasp state machine (same gates, same hysteresis, same weld-slot
  cap of 4), 1.0 s settle, and the v1.1 mechanical criteria
  re-implemented on Drake final poses + Drake contact pairs
  (point-pair and hydroelastic-surface contacts both read).
- **Weld realisation:** Drake has no runtime-toggleable weld here, so
  the grasp is a pose pin: while attached, the object's pose is set to
  `gripper_pose × snapshot_relpose` at each frame start and Drake's
  contact solver then runs the frame — contact resolves against the
  pin, approximating MuJoCo's soft constraint weld (solref 0.005),
  which yields to supports. (First realisation — override *after* the
  step — teleported objects through supports; z-traces in §4.)
  Known remaining gap: the pinned object carries no gripper velocity,
  so release momentum differs from MuJoCo's weld release.

**Global, logged engine-mapping choices (never per-episode):**
(i) friction — MuJoCo tuples are (sliding, torsional, rolling); Drake
`CoulombFriction` gets (static = sliding, dynamic = sliding), the only
sourced value; (ii) physics step 1 ms vs MuJoCo 0.5 ms, arm updated at
frame boundaries (20 Hz) vs per-substep interpolation — immaterial to
objects except through weld snapshots, which are taken at frame starts
in both engines; (iii) two contact-model configurations run as a
sensitivity pair: **point contact** (`kPointContactOnly`) and
**hydroelastic-with-fallback** (the Nov-15 Drake configuration:
compliant modulus 1e7, dissipation 10, 5 mm resolution hint on dynamic
objects; rigid hydroelastic on statics).

## 3. Equivalence diagnostics — results

Development subset (pre-reg Appendix A), nominal variant, identical
initial states, kinematic gt replay + 1.0 s settle, v1.1 mechanical
criteria in both engines. MuJoCo successes on the subset: block 0/4,
eye 1/4 (ep1798), ring 0/4, tool 0/4.

| Task | Outcome agreement (point / hydro) | Secondary agreement (point / hydro) | Final-pos Δ median / max, m (point) | Final-pos Δ median / max, m (hydro) |
|---|---|---|---|---|
| block_stack | 3/4 / **4/4** | 3/4 / 3/4 | 0.004 / 0.555 | 0.005 / 0.555 |
| eye_drops_to_basket | 4/4 / 4/4 | 4/4 / 4/4 | 0.165 / 0.322 | 0.152 / 0.351 |
| ring_insert | 4/4 / 4/4 | 4/4 / 4/4 | 0.014 / 0.585 | 0.010 / 0.586 |
| tool_insert | 4/4 / 4/4 | 4/4 / 4/4 | 0.115 / 0.369 | 0.161 / 0.231 |
| **Total** | **15/16 / 16/16** | **15/16 / 15/16** | — | — |

Orientation deltas (full-quaternion angle; medians): block ≈2° in both
configs (cubes stay flat); ring ≈88°, eye ≈113–123°, tool 153° (point) /
45° (hydro). Read with care: the tool capsule and the ring are
physically symmetric under end-for-end / face flips that the quaternion
angle counts as ~180°, and tipped cartons land chaotically in both
engines — orientation divergence for ring/tool overstates physical
divergence; carton orientation divergence is real (post-release tip /
roll outcomes differ).

Grasp-event agreement is tight: attach/release frames match exactly on
most episodes and within ±1–2 frames on the rest, in both configs, with
three diagnosed exceptions (§4). Drake solver-failure flags (v1.1
out-of-bounds/non-finite rule): point 5/16 (block 489, eye 1630, eye
1783, ring 195, tool 883); hydro 4/16 (block 489, eye 1630, ring 195,
tool 852). MuJoCo flags 0/16 on the same rule.

## 4. Disagreements and diagnosed causes

1. **block_stack 498 — the only outcome-class disagreement (point
   config only).** Identical grasp events (attach 59, release 205 in
   both engines). In Drake-point the placed cube settles stacked
   (within inset, contact with lower cube only → SUCCESS); in MuJoCo
   it ends beside the base cube on the mat (FAIL). Final positions
   differ by 2.4–3.0 cm — the placement is borderline against the
   11.25 mm inset half-width, and post-release drop/settle dynamics
   (contact model) decide which side it lands. Under hydroelastic the
   episode agrees (both FAIL). This is a genuine cross-engine
   divergence on a borderline placement, not a port defect.
2. **block_stack 411 — secondary-criterion disagreement (both
   configs).** Events identical; finals within ~1–2 mm. MuJoCo's two
   cubes end at exactly equal z (0.014920…) so its upper-cube
   tie-break picks cube_b; Drake's differ by 20 µm and pick cube_a.
   The secondary criterion measures the *upper* cube against a
   different recorded final in each engine and flips. A tie-break
   artifact of the criterion, not physics.
3. **Objects ending below the support surface in Drake (block 489
   cube_a, eye 1630 carton, ring 195 ring; final z −0.13…−0.22 in both
   configs).** All are episodes whose last grasp is still attached at
   trajectory end or re-grasped after a drop. Z-traces (recorded per
   episode in the Drake result JSONs) show the mechanism: the weld
   pin's relpose is snapshotted at the dwell frame; when the recorded
   gripper subsequently descends past the object's resting plane, the
   pin drives the object into/through its support — MuJoCo's soft
   weld yields under the same motion (its object stays on the
   support, and its weld relpose effectively re-seats). The pin
   approximation is exact for grasps whose snapshot and place heights
   align (block 411/453/498 finals within mm–cm) and fails this
   class. These variants are caught by the solver-failure flag and
   would be excluded + counted under the v1.1 harness rule — but in a
   *gate* harness they would silently corrupt still-attached
   outcomes, so the Drake grasp needs a contact-respecting
   realisation (e.g. a stiff compliant grasp constraint) before
   Stage 2 use. This is the port's main open defect.
4. **Loose-capsule migration (tool_insert).** In Drake point contact
   the loose tool rolls/slides ≈8 cm in −x and sinks 3 mm before the
   grasp window (traced frame-by-frame on ep683), so ep683's attach
   gates never pass in point config (MuJoCo attaches at frame 92 and
   holds the tool to the end). Hydroelastic removes the sink (rests
   at exactly z = 0.005) and slows the migration (≈5.6 cm by frame
   95) enough that ep683's attach fires at frame 92, matching
   MuJoCo; conversely hydro ep852 attaches in Drake (77/161/200)
   where MuJoCo never does. Cause: capsule-on-plane contact in Drake
   has effectively no rolling resistance at these parameters, where
   MuJoCo's contact (friction tuple incl. rolling component, patch
   of contacts along the capsule) holds the tool at its recorded
   pose. Outcome classes still agree on all four tool episodes in
   both configs, but the loose-object phase is **not** engine-
   equivalent for slender round objects.
5. **Superseded build defects (found and fixed during the port, not
   in the reported numbers):** (a) arm proximity role not removed by
   `AssignRole` — the live Drake gripper shoved objects, e.g. it
   pushed eye 1746's carton into attach-gate range (static trace
   proves the gates never pass with the carton at its recorded pose:
   min centre distance 0.0614 m vs the 0.060 m gate) and ejected
   block 498's cube_a to (0.85, 2.71, −1.09); (b) post-step weld
   override teleporting through supports (replaced by the frame-
   start pin, §2); (c) tool axis read from the wrong body-frame axis
   in an early criteria draft (body frame is MuJoCo's; only the
   capsule *geometry* is rotated).

## 5. Reading of the result (builder's, for the verifier)

- The port itself is faithful: identical inputs, FK-exact arm,
  bit-exact MuJoCo baseline, grasp state machine event-aligned, and
  outcome classes agree 15–16/16 on the development subset depending
  on contact configuration, with the single flip (498) diagnosed as
  borderline-placement contact dynamics and configuration-sensitive.
- Final-pose divergence is nevertheless large where physics is
  chaotic (drops, tips, rolls): position medians of 10–16 cm for
  eye/tool, and a class of still-attached episodes corrupts in Drake
  via the weld pin (cause 3). Per §3.2's design, certification rests
  on *policy-outcome* agreement on the control checkpoints inside a
  registered bound — these kinematic diagnostics say the instrument
  is close enough to run that test, and they name the two things
  most likely to move it: the Drake grasp realisation for
  still-attached/re-grasp episodes, and loose-capsule rolling.
- Nothing was tuned per episode. The only iterations were global:
  arm-proximity removal (a MuJoCo-semantics requirement), the weld
  pin ordering, and the two pre-declared contact configurations.

## 6. Compute discipline and time

All runs CPU-only, `nice -n 19`, launched via `launch_dgx_job.sh`
under the job guard (runs/batch1–5; one preflight refusal at load1
15.54 > 14 was honoured and relaunched after load fell). DGX training
was not disturbed. Guard mirror: the off-box mirror target was not
provisioned for this VM→DGX path, so batches ran with the guard's
logged `JOB_GUARD_ALLOW_NO_MIRROR=1` override — telemetry is on-box
in `runs/*/guard/`; flagged here as a discipline gap to close before
Stage 2 runs. Per-episode wall time: MuJoCo ≈1–2 s, Drake ≈2–6 s
(hydroelastic ring ≈6 s). **Time spent: ≈1.5 agent-hours of the
20 agent-hour box** (port + harness + diagnostics); the agreement
harness on control checkpoints (the rest of the box) is part 2 of
Stage 1.2 and not started here.

## 7. Open items for part 2 / verifier

1. Contact-respecting Drake grasp (replace the pose pin) — blocks
   trusting still-attached and re-grasp outcomes in Drake.
2. Loose-capsule rolling in Drake — decide the global stance
   (hydroelastic mitigates; neither config matches MuJoCo's hold).
3. Release-velocity inheritance for the pinned grasp (gripper
   finite-difference velocity at release) — known gap, unmeasured
   contribution to post-release divergence.
4. Independent verification per pre-reg §9(b): re-run the dev-subset
   diagnostics from the raw artifacts in `drake/results*` +
   `equiv_summary.json` files; certification verdicts are the
   verifier's, not this report's.
