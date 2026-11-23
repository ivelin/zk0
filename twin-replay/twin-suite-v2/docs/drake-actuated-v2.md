# Twin-Suite v2 — Stage 1.2 Part 2a: Drake Actuated Rollout Capability

**Date:** 2026-11-23 · **Builder context:** Stage 1.2 part 2a build agent
(separate verifier required per pre-registration §9 — this report is a
builder deliverable, NOT a certification).
**Governing docs:** `twin-suite-v2-preregistration-20261122.md` §3.2/§4;
`drake-port-v2.md` + `verification-v2-drake-port.md` (CERTIFIED-WITH-
FINDINGS; F3 = the pose-pin grasp defect); `fingerpad-fix-v2.md`;
`rollout-harness-v2.md`.
**Code (DGX):** `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake_actuated/`
— `actuated_port.py` (actuated fork of `drake/drake_port.py`; the
original and every v1.1 file are untouched — verified: no file under
the v1.1 `code/`, `scenes/`, or `validation/` trees has an mtime after
this build began), `grasp_probes_drake.py`, `sim_runner_drake.py`,
`mujoco_padpos.py` (parity diagnostic), `run_batch{,2,3}.sh`,
`results/`, `smoke/base_drake/`. Guard rundirs:
`runs/drake-actuated1…5` (batch 4 exit 1 = a renderer-registration
failure, superseded by batch 5, exit 0; all other batches exit 0).

## Verdict up front

**The Drake actuated capability is built and the policy loop runs
end-to-end in Drake** — actuated arm (no pose pin, no weld anywhere),
the MuJoCo finger-pad geometry ported with world-position parity of
**0.002 mm**, renders from the three matched viewpoints through the
same policy-service IPC contract, and a base-checkpoint smoke on
block_stack ep411 nominal that completes, scores, and lands on the
**same outcome classes as the MuJoCo smoke** (FAIL / secondary PASS /
no solver failure).

**The grasp acceptance battery, run with the exact MuJoCo protocol,
FAILS in Drake at all heights — with a diagnosed cause that is not a
geometry or actuation defect:** on the approach→descent path, the
fixed finger lands on the mat and Drake's stiff hydroelastic contact
+ the effort-limited PD cannot slide it free, so the descent stalls
≈20 mm short of the target and the fingers close in air beside the
cube. A labeled diagnostic variant (direct descent, ×3 segment times)
shows the capability itself exists in Drake: **pinch-lift + 2 s hold
PASSES at all three heights** by finger contact alone. The divergence
is reported, not tuned away: the battery-as-specified and the
diagnostic are both recorded below, and which protocol Stage 2 uses
in Drake is a founder/verifier decision.

## 1. Actuated arm — actuator parity

The Nov-15 `so101_drake.urdf` transmissions already give one Drake
`JointActuator` per joint. `actuated_port.py` sets, per actuator:

| Parameter | MuJoCo vendored model (`so101_new_calib.xml`) | Drake actuated port |
|---|---|---|
| Position gain kp | 998.22 (sts3215 class) | `PdControllerGains(p=998.22)` |
| Velocity gain kv | 2.731 (sts3215 class) | `PdControllerGains(d=2.731)` |
| Torque limit | class forcerange ±2.94 N·m; **actuator-level override ±3.35 N·m** | `effort_limit = 3.35 N·m` (the actuator-level value, matching what MuJoCo actually applies) |
| Speed limit | harness slew limit 6.16 rad/s on ctrl targets | identical slew limit in `sim_runner_drake.py` |
| Joint damping / frictionloss / armature | 0.60 / 0.052 / 0.028 (model file) | URDF carries no per-joint damping/friction in Drake form — **difference logged**: Drake arm dynamics rely on the PD + plant defaults where MuJoCo adds joint damping 0.60 |
| Control path | `data.ctrl` position targets | desired-state input port (`DesiredStateSource` LeafSystem: position targets in actuator order, zero velocity targets) |

Tracking evidence: in the probe diagnostic the arm reached the IK
descent target with site error 1.8 mm and joint positions equal to
the desired targets to 3 decimals — actuation is not the limiting
factor in any failure below. Physics step 1 ms (Drake discrete plant,
as part 1) vs MuJoCo 0.5 ms; contact model hydroelastic-with-fallback
(the part-1 configuration that agreed 16/16), global for all runs.

## 2. Finger pads — geometry parity

Transcribed verbatim (pos/size, body-local frames) from
`scenes-v2/arm/so101_new_calib_padded.xml` into Drake collision
geometry on `gripper_link` / `moving_jaw_so101_v1_link`:
`fixed_jaw_box1..7` (box3's capsule converted radius 0.0011,
half-length 0.002 → Drake `Capsule(0.0011, 0.004)`),
`moving_jaw_box1..3`, `finger_pad_fixed` (box 0.005×0.020×0.046 at
(−0.0105, 0, −0.080)), `finger_pad_moving` (box 0.006×0.060×0.012 at
(−0.009, −0.048, 0.01875), tilt −0.13 rad). MuJoCo size=half-sizes →
Drake Box full sizes (2×). Finger friction: MuJoCo tuple
(1.0, 0.005, 0.0001) → Drake Coulomb (1.0, 1.0) per the port's
standing rule. All other arm collision geoms keep proximity stripped
(MuJoCo harness: only the two finger bodies collide). **Tip spheres
stay OUT** — the same disclosure as the MuJoCo fix (they measurably
changed its battery; the shipped MuJoCo model excludes them, md5
`0c3f0d1535dc6ea50c6b8ef527a58cf3`, re-verified this build).

**Parity measurement** (both engines, recorded frame-130 pose of
ep411 set kinematically, geom centre world positions):

| Geom | MuJoCo (m) | Drake (m) | Δ |
|---|---|---|---|
| finger_pad_fixed centre | (0.220459, 0.038728, 0.025088) | (0.220459, 0.038729, 0.025088) | **0.002 mm** |
| finger_pad_moving centre | (0.242125, 0.034888, 0.046344) | (0.242125, 0.034889, 0.046345) | **0.002 mm** |
| gripperframe site | (0.231792, 0.040041, 0.010761) | (0.231792, 0.040043, 0.010761) | **0.002 mm** |

The body frames are the same frames; the transcription is exact.

**Engine-forced differences (each logged, none hidden):**
1. **Gripper-part meshes OUT in Drake** (`fixed_jaw_mesh`,
   `moving_jaw_mesh0/1`). Registering a `Mesh` on any body after the
   URDF parse breaks `Finalize()` in this Drake build —
   `RuntimeError: Collision filter declaration references geometry id
   164/165 which has not been registered with this filter system`
   (reproduced on the finger bodies and on welded carrier bodies).
   The boxes + authored pads — the grasp faces — are all present;
   the missing meshes are the bulk hulls behind the fingers. Recorded
   in every rollout record's `finger_mesh_failed`.
2. Capsule size convention (above) — converted, not approximate.
3. Part meshes would have loaded from the Nov-15 assets' `.obj` twins
   of the same STL geometry (Drake Mesh does not read `.stl` here);
   moot given (1).

## 3. Grasp acceptance battery — both engines

Protocol (identical semantics, `grasp_probes_drake.py` mirroring
`harness/fingerpad_probes.py`): IK-centred grasp of ep411's 30 mm
cube_a, site x-offset −0.020 m, pre-close 0.7 rad, close to 0.05 rad
target, staged lift, 2 s hold; PASS = held by finger contact alone
(moving jaw in contact set, no world contact, z > 0.05). MuJoCo
numbers are this build's re-run of the shipped battery (batch 1),
matching `fingerpad-fix-v2.md` exactly.

| Case | MuJoCo (shipped battery) | Drake, battery as specified | Drake, diagnostic variant* |
|---|---|---|---|
| +0.005 m | shove 11.0 mm, stall 0.096, lift 0.076, hold dropped — **FAIL** | shove 0.1 mm, grip 0.050 (no stall), no lift — **FAIL** | shove 6.9 mm, stall 0.225, lift 0.084, hold 0.083, contacts [gripper, moving_jaw] — **PASS** |
| 0.000 m | shove 12.5 mm, stall 0.095, lift 0.082, hold 0.081 — **PASS** | shove 0.1 mm, grip 0.050, no lift — **FAIL** | shove 6.8 mm, stall 0.231, lift 0.084, hold 0.084 — **PASS** |
| −0.005 m | shove 13.9 mm, stall 0.098, lift 0.097, hold 0.101 — **PASS** | shove 0.3 mm, grip 0.050, no lift — **FAIL** | shove 7.2 mm, stall 0.239, lift 0.083, hold 0.083 — **PASS** |
| Site-centred (xoff 0, 0.000 m) | shove −13.3 mm, moving jaw never contacts, no lift — **FAIL** (probe 3) | shove 0.0 mm, grip 0.050, no lift — **FAIL (same class)** | shove 0.2 mm, stall 0.231, lift/hold 0.083 — **PASS (diverges — see §4)** |

\* Diagnostic variant (labeled, global, not per-episode):
`PROBE_SKIP_APPROACH=1` (direct descent from the reset pose, skipping
the +0.13 m approach waypoint) + `PROBE_TIMESCALE=3` (all segment
durations ×3). A ×3-timescale run *with* the approach waypoint was
also run: identical FAIL at all heights — the block is not timing.

**Pinch-centre offset (model property, re-measured in Drake):**
site→pinch-midpoint distance at close = **24.2 mm** in the battery
configuration (MuJoCo: ≈24.5 mm as cited in the build brief; the
fingerpad report's gap geometry — fixed face at site −0.002 m, moving
face at site +0.024 m — is the same measurement decomposed). In the
diagnostic's stalled grasp it measures 25.9–26.1 mm. The offset is a
property of the model, reproduced across engines.

## 4. Divergences + diagnoses (not tuned away)

1. **Approach-path contact lock (the battery failure).** Traced
   waypoint by waypoint: after the approach waypoint (site z 0.138,
   on target), the descent ends with the site at z 0.0346 vs the
   0.015 target — the arm stalls ≈20 mm short with contact set
   [gripper, world]: the fixed finger's tip has landed on the mat
   and, under hydroelastic contact (stiff, μ=1.0 finger friction) +
   the 3.35 N·m effort-limited PD, it neither penetrates nor slides
   the remaining distance. MuJoCo's soft-constraint contact
   (solref 0.005) yields/slides through the same transit, so its
   descent reaches and the pinch forms. ×3 segment time does not
   change the Drake outcome (static lock, not creep). Direct descent
   from the reset pose (a different transit path that brings the
   fingers in beside the cube without the tip-first mat landing)
   reaches the target (site error 1.8 mm) and the pinch forms.
   **This is a contact-model × executor-path divergence, global to
   the approach geometry — it will affect policy rollouts whose
   approach resembles the battery's approach waypoint, and it is the
   main Stage 2 risk in Drake.**
2. **Grip stall differs where the pinch does form:** Drake stalls at
   0.225–0.239 rad vs MuJoCo 0.095–0.098 rad on the same cube. The
   Drake fingers stop ~0.13 rad more open — consistent with
   hydroelastic compliance (the pads/cube deform under load instead
   of the hard stall) plus the absent part-mesh bulk (§2.1). The
   hold is nevertheless stable ≥2 s at all heights in the diagnostic.
   Shove is smaller in Drake (6.8–7.2 mm vs 11.0–13.9 mm).
3. **Site-centred diagnostic PASS vs MuJoCo FAIL.** In the diagnostic
   variant only, the site-centred probe also lifts: with the direct
   descent and ×3 close time, the closing fingers self-centre the
   cube in Drake where MuJoCo's site-centred probes sweep it away.
   Under the battery as specified, both engines FAIL site-centred
   (the brief's expectation holds for the battery). Flagged so the
   diagnostic PASS is not misread as battery parity.
4. **Joint damping gap** (§1): the URDF/Drake side has no equivalent
   of MuJoCo's joint damping 0.60 / frictionloss 0.052 / armature
   0.028 in the port. Not implicated in any result above (tracking
   was exact), but it is an open instrument difference for rollout
   dynamics.

## 5. Policy-rollout smoke (base checkpoint, Drake)

`sim_runner_drake.py` drives the **same** `policy_service.py` (FL
venv) through the same JSON-lines IPC (one request per 50-action
chunk, frames as `.npy` in a shared tmp dir), with renders from
Drake's SceneGraph VTK renderer at the matched viewpoints: front =
the MuJoCo harness's `frontfix` solution (frontcam position
(0.75, −0.55, 0.35), fovy 55, look-at (0.30, 0, 0.05)); top = v1.1
topcam (0.3, 0, 0.85), fovy 50, looking down; wrist = the MuJoCo
harness's per-render site-relative pose (eye 6 cm behind the gripper
along wrist→gripper axis + 5 cm up, target 10 cm past the site),
fovy 60 — per-render camera poses via `QueryObject.RenderColorImage`,
so the wrist view is dynamic exactly as in MuJoCo. State [6] degrees
with the identical gripper mapping; same prompt; horizon 296 steps
(276 recorded + 20 settle); scoring by `drake_replay.evaluate` (the
part-1 Drake re-implementation of the v1.1 criteria) + the same
generalised solver-failure rule. Records in the MuJoCo harness's
format (`rollouts.jsonl` + `traj_*.npz`, fsync'd, skip-if-recorded).

| Run | sim_success | secondary | solver_failure | wall |
|---|---|---|---|---|
| Drake, base @ ep411 nominal | **False** | **True** | **False** | 51.4 s |
| MuJoCo, base @ ep411 nominal (Stage 1.4 smoke) | False | True | False | 52.6 s |

Base fails the task in Drake, as expected — **smoke success =
mechanics**: the loop ran (6 chunks through the IPC, sample frames
saved and produced by the Drake renderer, cubes unmoved at their
start poses (0.2391, 0.0296, 0.015) / (0.0661, −0.0231, 0.015)),
scoring executed, record complete. Outcome classes agree with
MuJoCo on this rollout.

## 6. Compute discipline

All sustained work ran CPU-only, `nice -n 19`, via
`launch_dgx_job.sh` + job guard (logonly, JOB_LOCK=0) in five batches
(`runs/drake-actuated1…5`); short diagnostics (scene builds, pad
parity, IK traces) ran as bounded direct commands. DGX training was
not disturbed (guard preflight/telemetry in the rundirs).
**Guard mirror:** `JOB_GUARD_MIRROR_TARGET` was set on every launch
(the precedent tailnet target); the guard log records
`MIRROR FAIL rc=255` — the Muse VM is unreachable from the DGX on
that path (same finding as the fingerpad build). **No
`JOB_GUARD_ALLOW_NO_MIRROR` override was used.** Telemetry is on-box
in `runs/drake-actuated*/guard/`. The mirror path remains a Stage 2
blocker per pre-registration §12 / verification F4 — unchanged by
this build.

## 7. What this does and does not establish

- Establishes: actuated Drake arm with STS3215-honest gains/limits;
  pad geometry at cross-engine parity (0.002 mm); no pin/weld in any
  Drake rollout path (F3's defect class is retired for the actuated
  harness — nothing in `drake_actuated/` sets object poses); the
  policy loop + scoring + records working end-to-end in Drake with a
  class-agreeing base smoke.
- Does **not** establish grasp parity under the battery as specified:
  the approach-path contact lock (§4.1) makes the Drake battery FAIL
  where MuJoCo passes 2/3. Stage 2 Drake rollouts should not be
  trusted for grasp outcomes until the founder/verifier decides how
  to treat §4.1 (e.g. accept the diagnostic-variant protocol for
  Drake, change the executor approach path globally, or revisit the
  contact configuration) — each is a suite-level decision, not a
  builder tweak.
- Time spent: ≈2.5 agent-hours (Stage 1.2 box: 20 agent-hours total;
  part 1 reported ≈1.5 h, this part ≈2.5 h).

---

## Fix note — 2026-11-23 (verification findings F1 + F2)

Appended by the F1/F2 fix build; the original text above is unedited.
Governing finding: `verification-v2-drake-actuated.md` §5 (F1, major)
and §6 (F2, minor). All runs below: CPU-only, `nice -n 19`, via
`launch_dgx_job.sh` + job guard logonly (rundir
`runs/drake-renderfix3`; guard mirror set to the precedent target and
again logging `MIRROR FAIL rc=255` — no override used). No v1.1 file
modified. Two earlier fix batches (`drake-renderfix1/2`) failed on
build defects in the fix itself and are superseded by
`drake-renderfix3`; their guard `command.out` logs are retained.

### F1 — blank renders: root cause

**The camera rotation helper used the wrong convention.**
`sim_runner_drake.py::_lookat_R` built the camera rotation with −z
toward the target — the OpenGL/MuJoCo camera convention (camera looks
along its −z). Drake's RenderEngine camera frame is x = right,
y = **down**, z = **forward** (into the scene), so every camera —
front, top, wrist — pointed exactly *away* from the scene and every
render was the `RenderEngineVtk` default clear colour
(204, 229, 255) alone: exactly 1 unique colour per frame, byte-
identical across viewpoints, in the saved PNGs and in the `.npy`
frames fed to the policy. The policy ran blind, as the verifier
found. Isolation evidence (minimal single-scene diagnostics on
ep411): the same scene rendered with the rotation flipped 180° about
the camera y-axis returns 224 unique colours, and a depth render in
the shipped orientation has 0% finite pixels — the renderer, the
SceneGraph registration, and the camera positions were never the
problem; the orientation was. (The batch-4 "renderer-registration
failure" in the build history was the separate mesh collision-filter
RuntimeError of §2.1 — a clue about registration fragility in this
Drake build, but not the cause of the blank frames.)

Two secondary content defects were found behind the primary one and
fixed in the same pass, because with the camera corrected they would
still have hidden parts of the scene:

1. **Arm invisible to VTK.** The URDF visual meshes are `.stl`, and
   `RenderEngineVtk` ignores `.stl` Mesh specifications outright
   (runtime warning: it supports only `.obj` / `.gltf`). Every visual
   `.stl` has an `.obj` twin in the same assets directory (the URDF
   collision meshes reference them).
2. **Floor / mat / walls invisible.** In `actuated_port.py` these
   statics had collision (proximity) geometry only — no visual
   geometry at all.

### F1 — the fix (all in `drake_actuated/`, global, no per-scene data)

- `sim_runner_drake.py::_lookat_R`: build the rotation in Drake's
  frame — z = forward (target − position), x = normalize(cross(z,
  up)) (image right), y = cross(z, x) (image down), world up +z with
  a (0, 1, 0) fallback for the top view, which reproduces the v1.1
  topcam's identity-quat orientation (image right = +x, image
  up = +y). This is exactly the MuJoCo harness's look-at mapped into
  Drake's camera frame.
- `actuated_port.py` (in `DrakeScene`, after the URDF parse): for
  every robot visual geometry whose shape is an `.stl` Mesh, register
  an `.obj` visual clone at the identical frame pose with the same
  diffuse colour, then strip the `.stl` original's illustration +
  perception roles so each part keeps exactly one visual geometry —
  **17 clones** on ep411 (`arm_visual_obj_clones` is recorded in
  every rollout record). Implementation note for this Drake build:
  `GeometryId` objects returned by `GetVisualGeometriesForBody` are
  unstable proxies (a cached id's value mutates and SceneGraph then
  rejects it as unregistered), so the strip re-fetches the body's
  geometry ids and a fresh inspector per geometry, matches by name,
  and strips immediately; with that, all 17 strips succeed
  (`arm_visual_strip_failed: []`).
- `actuated_port.py`: visual twins for floor / mat / walls at the
  collision geometry's poses, in the v1.1 MuJoCo scene-XML colours
  (floor rgba 0.35 0.30 0.25; mat 0.16 0.19 0.24; walls
  0.76 0.66 0.50).
- `sim_runner_drake.py`: every rollout record now carries a
  `frame_content` field — per-view unique-colour count and
  non-clear-colour fraction for chunk 0, plus the minimum
  non-clear fraction over all chunks — so a blank-frame regression
  is visible in the record itself, not only in saved samples.

### F1 — proof (ep411 nominal, `render_fix_evidence/` on the DGX)

Rendered with the fixed code at the recorded frame-0 pose and at the
mid-episode recorded grasp pose (frame 130), all three views; the
builder inspected every PNG visually. Frames show the yellow arm,
the beige enclosure walls, the dark mat, and both cubes; the
frame-130 front frame shows the arm descended to the green cube.
Colour checks use the ep411 scene colours (cube_a green
(64, 166, 76), cube_b purple (140, 64, 191), arm yellow
(255, 209, 31); tolerance 70 in RGB distance):

| Frame | View | Unique colours | Non-clear frac | Cube / arm evidence |
|---|---|---|---|---|
| frame 0 | front | 1,097 | 0.78 | green cube 46 px @ (116, 131); arm yellow 349 px @ (73, 98); mat 42,486 px |
| frame 0 | top | 858 | 1.00 | green 98 px @ (106, 118); purple 32 px @ (49, 138); arm 894 px @ (30, 127); walls 2,121 px |
| frame 0 | wrist | 1,286 | 1.00 | green 974 px @ (114, 21); arm 6,143 px @ (131, 144) |
| frame 130 | front | 1,252 | 0.78 | green 49 px @ (116, 131); arm 475 px @ (80, 88) |
| frame 130 | top | 1,074 | 1.00 | green 23 px @ (109, 121); purple 50 px @ (49, 137); arm 469 px @ (52, 123) |
| frame 130 | wrist | 1,168 | 1.00 | arm 5,560 px @ (128, 138); mat 38,788 px |

(Before the fix every cell in the first two numeric columns was
1 / 0.00.) The purple cube reads 0 px within tolerance in the front
and wrist frame-0 views because from those viewpoints it sits
behind the gripper/arm — it is visible as a small purple patch in
the front PNG and is measured directly in both top views, at the
image position its world pose (0.066, −0.023) predicts. Top-view
geometry cross-check: the green cube's measured centroid (106, 118)
sits within ~4 px of the position projected from its world pose
(0.239, 0.030) under the top camera.

Side-by-side with the MuJoCo harness at the same scene/pose
(`render_fix_evidence/sidebyside_{front,top,wrist}.png`; MuJoCo side
from a fresh harness gt-mode ep411 run, `smoke/mj_ep411_sbs/`):
compositions match in all three views — arm entering from the left
in front/top, cubes in the same relative positions, wrist view
dominated by the yellow wrist/gripper with the green cube above.
Residual differences are engine rendering, not viewpoint: Drake's
flat lighting vs MuJoCo's light gradient/shadows, and Drake's clear
colour showing above the horizon where MuJoCo's infinite floor
plane fills the frame.

### F1 — re-smoke (base @ ep411 nominal, fixed renders)

Fresh output directory `drake_actuated/smoke/base_drake_fixed/`
(the original blind smoke record in `smoke/base_drake/` is retained
untouched as evidence). Full standard record (`rollouts.jsonl` +
`traj_block_stack__ep411_nominal.npz`):

| Run | sim_success | secondary | solver_failure | wall |
|---|---|---|---|---|
| Drake, base @ ep411 nominal, **fixed renders** | **False** | **True** | **False** | 54.5 s |
| Drake, base @ ep411 nominal, original (blind) smoke | False | True | False | 51.4 s |
| MuJoCo, base @ ep411 nominal (Stage 1.4 smoke) | False | True | False | 52.6 s |

Outcome classes are unchanged — now with a sighted policy. Cubes
finish at their start poses (0.2391, 0.0296, 0.015) /
(0.0661, −0.0231, 0.015), as in both earlier smokes. The record's
`frame_content` proves the policy received real frames throughout:
chunk 0 unique colours front 1,072 / top 873 / wrist 1,272;
minimum non-clear fraction over all 6 chunks front 0.78 / top 1.00 /
wrist 0.58; and the final chunk's `.npy` frames actually fed to the
policy service measure 1,266 / 1,095 / 1,332 unique colours
(front / top / wrist). F1 is closed on this evidence; independent
verification is the verifier's, per pre-registration §9.

### F2 — wording correction ("no weld anywhere")

The §"Verdict up front" phrase "actuated arm (no pose pin, no weld
anywhere)" overstates, as verification §6 found. `actuated_port.py`
does use `plant.WeldFrames` — for (a) the arm base to the world,
(b) static fixture bodies to the world, and (c) the (default-
inactive) mesh carrier body. These are standard static scene
construction; none of them touches a graspable object. The accurate
claim, which the verification confirmed in substance, is: **no grasp
pin/weld, and no object-pose setting during a rollout** (no
`SetFreeBodyPose` / `set_free_pose` call exists in
`sim_runner_drake.py`; `grasp_probes_drake.py` uses `set_free_pose`
only in `reset_arm()` to restore initial poses between probes).
The same narrowing applies to the "No weld, no pin" line in the
`sim_runner_drake.py` module docstring.
