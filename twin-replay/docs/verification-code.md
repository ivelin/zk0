# Study 001 Twin-Replay — Independent Code Verification (V-CODE)

Verifier: V-CODE, independent context (did not write the code under audit).
Date: 2026-11-22. Spec: `docs/twin-replay-preregistration-20261122.md`,
`docs/twin-replay-amendment1-20261122.md`, `docs/twin-replay-amendment2-20261122.md`.
Artifacts audited: `code/` (all 17 .py files), `validation/param-changelog.md`,
`validation/validation_results{,_run1,_run2}.json`, `validation/run{1,2,3}.log`,
`scenes/calibration_fk_affine.json`, sampled scene XMLs.
The builders' narrative report (`docs/twin-validation-report.md`) was NOT read.
No checkpoint inference was performed or audited.

## VERDICT: NOT-CERTIFIED

The scoring/validation code does not implement the pre-registration "EXACTLY
as written" (its own claim, `code/replay.py:1-15` docstring and
`evaluate()` docstring at `replay.py:318`). The primary criteria are replaced
by geometric proxies that omit the pre-registered contact/support conditions;
the secondary position criterion is computed but never evaluated against its
pre-registered thresholds anywhere in the pipeline; the grasp device departs
from the registered 3.0 cm centre gate in five separate ways; and the
changelog's timestamps cannot be reconciled with the run/file timestamps on
the same machine. None of these defects flatters the headline result in a
way that rescues certification — with one exception (F-07, defect exclusions
removing only failing variants) the deviations are direction-neutral or
conservative — but certification under the pre-registration is about fidelity
to the registered method, and the method as coded is not the method as
registered. Individual passes are recorded per check below; several
sub-systems (variant determinism, 60% rule arithmetic, settle time, global
scope of the two calibration transforms in the replay path) do certify.

---

## 1. Criteria fidelity

### 1a. What matches

- **Settle 1.0 s** — `twin_params.py:20 SETTLE_S = 1.0`; used in
  `replay.py:291` (`int(P.SETTLE_S / P.TIMESTEP)` steps after trajectory end). MATCH.
- **≥60% variant rule** — `run_validation.py:91-92`:
  `frac = len(agreeing)/len(valid); reproduced = bool(valid) and frac >= 0.6`.
  MATCH (Amendment 1 A2). Task bar ≥80% reproduced is computed as
  `bar_80pct_met` in `run_validation.py:124`. MATCH.
- **Block inset rule (the 25% part)** — `replay.py:348-349` (evaluate-relative
  lines 32-33): `inset = (e/2)*0.75; within = abs(upper-lower) <= inset` per
  axis. The centre-within-footprint-inset-by-25% clause is implemented as
  written.
- **Tool axis 20°** — `replay.py:394-397`: angle between tool local-x axis and
  slot axis, `ang <= 20.0`. MATCH. **Ring 1 cm upper bound** —
  `replay.py:381`: `(ring_bottom - bt) <= 0.010`. MATCH (upper bound only; see F-03).

### 1b. Findings

**F-01 (MAJOR) — Contact/support conditions are never checked; `contacts_with`
is dead code.** `replay.py:322-333` defines `contacts_with(nm)`; it is never
called anywhere in `code/` (grep: single hit, the definition). Every
pre-registered criterion that is stated in contact/support terms is instead
scored by position proxies:
- block_stack ("upper cube **in contact with** lower cube's top … **no other
  support contact**"): `replay.py:350-352` substitutes
  `abs(dz) < 0.006` (face gap, a silent 6 mm tolerance not in the
  pre-registration), `upper[2] > e*1.2` (silent threshold), and
  `no_other = upper[2]-e/2 > 0.004` — a height-above-mat test, not a support
  test. A cube resting against a wall/fixture at the right height and gap
  would pass; a cube in genuine contact with a 7 mm solver gap would fail.
- eye_drops ("final resting **support is the basket** … **not the platform
  or the mat**"): `replay.py:363` substitutes
  `on_basket_floor = pc[2] < PLATFORM_HEIGHT*0.6 and pc[2] > 0.005`
  (0.005 < z < 0.054). No support/contact test at all; support by anything
  inside the footprint projection at that height passes.
- cable_clip ("clip **closed over it**"): the clip jaws are static geoms in
  `build_scenes.py` (clip_jaw_a/b) that never move; closure is unmodelled and
  unchecked. `replay.py:417` checks only segment-in-channel geometry with
  silent tolerances (±0.012 m in x and z, channel half-width + 0.002 m).

**F-02 (MAJOR) — Secondary position criterion is never enforced or even
thresholded.** The pre-registration sets 3.0 cm (block/eye/tool) and 2.0 cm
(ring, when not engaged). `replay.py:336-342` computes `planar_err_m`, and
`run_validation.py:104` copies the nominal value into the output as
`nominal_planar_err_m` — but no code anywhere compares it to 0.03/0.02.
Consequence, from `validation_results.json` itself: **all four reproduced
eye_drops episodes violate the secondary criterion by 2–4×** on their nominal
variants — ep1757: 0.0728 m; ep1778: 0.0925 m; ep1810: 0.1243 m;
ep1811: 0.1218 m (threshold 0.030 m) — yet are counted REPRODUCED with no
flag in the summary structure. (Amendment 1 says the secondary is "assessed
on the nominal variant"; assessment that cannot fail is not assessment. The
pre-registration's certification bar names only outcome agreement, so this
does not by itself change any task's certified/not-certified status — but a
reader of `validation_results.json` cannot tell the criterion was missed,
which is exactly what the bar's "no unexplained systematic failure mode"
clause is meant to surface.)

**F-03 (MODERATE) — Ring engagement tolerance exceeds the geometry it
claims to test.** `replay.py:379`:
`engaged = planar <= 0.005 + (P.RING_R_IN - P.PEG_R) + 0.002` = 0.010 m.
The physical clearance is RING_R_IN − PEG_R = 0.003 m; beyond that the peg
cannot pass through the hole without intersecting the ring wall. The extra
5 mm + 2 mm are silent tolerances. Also `replay.py:381` allows
`ring_bottom - board_top >= -0.005` — 5 mm of interpenetration below the
board surface counts as "within 1 cm of the board top".

**F-04 (MODERATE) — Tool slot test ignores slot yaw and adds a silent
z-gate; slot size is an unregistered local choice.** `replay.py:395` tests
`abs(pt-sp) <= SLOT_SIZE/2` axis-aligned in world frame while the angle test
at `replay.py:393` does use `syaw` — if a slot were yawed, centre and axis
tests would use inconsistent frames (latent: Builder B fixture yaws are null
in the current data, so this did not fire). `replay.py:396` adds
`in_tray_z = 0.02 <= pt[2] <= 0.075`, a "resting in the tray" proxy found
nowhere in the pre-registration. `SLOT_SIZE = (0.110, 0.032)` is labelled
`LOCAL-CHOICE` at `twin_params.py:78` — i.e. the target footprint itself was
never sourced from the recordings.

**F-05 (MAJOR) — Playback rate is 20 fps, not the pre-registered 30 fps.**
Pre-registration: "arm driven kinematically (qpos set per recorded frame,
30 fps)". Code: `twin_params.py:16 FPS = 20.0`, consumed at
`replay.py:203 steps_per_frame = round((1/FPS)/TIMESTEP)` = 25 substeps.
The deviation IS flagged in code (`twin_params.py:16-19`: "NOTE: prereg text
also says 30 fps … 20 is the dataset's own rate") and `extract_frames.py:17`
also uses FPS=20 — but no artifact in the workdir proves the dataset rate is
20 fps; both numbers rest on the builders' assertion. If the gt traces are
30 fps data, every replay ran the arm at 2/3 speed (longer contact dwell,
more stable grasps) — a systematic, result-relevant difference that remains
unresolved on the evidence in the workdir. This is a flagged deviation, not
a silent one; it still breaks "exactly as written".

**F-06 (MAJOR) — The grasp device departs from the registered spec in five
ways (all in `replay.py`, all beyond the declared weld device).** The
pre-registration declares a "proximity weld on close" with Builder A's
3.0 cm centre gate and 50%/30% thresholds. The coded device
(`replay.py:126-285`) instead implements: (a) gate on **planar distance to
the object's material/shape** (point-to-footprint/annulus/axis,
`shape_dist`, `replay.py:151-176`), not centre distance; (b) **per-object
centre bounds enlarged** to 0.060 m (carton), 0.075 m (tool), 0.045 m
(cable segment) at `replay.py:198-202`, vs `GRASP_DIST_M = 0.030`
(`twin_params.py:24`); (c) **dwell/snapshot attach** — attach fires on
turn-up ≥3 mm / plateau / excursion end using a relpose snapshot from the
arg-min frame (`replay.py:255-268`), not at the threshold crossing; (d)
**release-recency ineligibility** — a released object cannot re-attach until
the site has been ≥3 cm from its centre (`replay.py:185-197, 231-238`), a
rule with no counterpart in the spec, introduced to suppress an artifact
the device itself creates; (e) the header comment at `replay.py:140-148`
still describes an "APPROACH ≥3 mm" requirement that the final code does
not implement (the trackers store no excursion-start distance) — comment
and code disagree about the device's own rules. Each change is narrated in
the changelog (#2, #5), so none is hidden; collectively they are a different
grasp device from the registered one, and every validation number is
conditional on it. Additionally `twin_params.py:27-40` still documents the
**non-inverted** closedness formula and states the channel "rises during a
grasp" — the opposite of the direction the code now implements (see §2b);
the parameter file's own documentation of its central normalisation is stale.

## 2. Per-episode fudging

**In the scoring/replay path: NONE FOUND.** Grep of `replay.py`,
`build_scenes.py`, `run_validation.py` shows no episode-id conditionals, no
per-episode parameter tables, no episode-keyed thresholds. All thresholds
are global constants or per-task/per-object-class functions
(`centre_bound(nm)` is keyed by object type, not episode).

**F-07 (MODERATE) — Per-episode hand edits exist upstream, in Builder B's
pose pipeline (`code/finalize.py`, `code/patch_finalize.py`,
`code/patch2.py`), and they feed the scenes the verifier is asked to trust.**
These are outside Builder C's scoring code but inside `code/` and inside
the artifact chain:
- `finalize.py:36-39` HAND_BLOCK_FIRST: hand-picked pixels for ep411, ep530,
  ep539 (×2), ep546.
- `finalize.py:40-44` EYE_FIRST_GROUND: 11 episodes hand-picked;
  `finalize.py:46-55` EYE_FINAL_GROUND: 18 episodes hand-picked.
- `finalize.py:56-57` TOOL_GROUND: 10 episodes; TOOL_LOW: 3 episodes forced low.
- `finalize.py:89-90`: confidence overrides for exactly ep411/cube_green and
  ep513/cube_purple.
- `patch_finalize.py:31-40` CABLE_BLOCK_GROUND: hand picks for **all 20**
  cable_clip episodes (the clip_block pose — the outcome reference for the
  cable criterion — is a hand pick in every cable episode).
- `patch_finalize.py:57-63`: per-episode ring peg/board derived from the
  final-frame ring detection on the stated assumption "ring seated on peg
  (recorded demo success)" — the outcome reference is anchored to the
  assumed outcome. Defensible for successful demos, but it is an assumption
  baked into the reference, and it is per-episode.
- `patch2.py:6-9`: hand picks for ep1780, ep1783, ep1814 cartons.
Mitigations: each pick is written into the scene JSON with
`method: "visual grounding pick …"` / `"manual pick …"`, i.e. disclosed in
the data, not hidden; and the picks are pose *estimates*, not scoring
changes. But the pre-registration's honesty rule 2 ("never per-episode")
is written about parameters during validation, and the changelog (§3)
never mentions any of these — a reader of the changelog alone would
conclude no per-episode hand work exists anywhere in the pipeline.

**F-08 (MODERATE) — Changelog #3a misdescribes the block final-pose code.**
Changelog #3a says the stack pixel is "mean of the two final detections when
they agree <80 px, else the manipulated (top) cube's final detection".
`build_scenes.py` (block finals block in `load_episode_poses`) actually
selects, whenever the base cube's initial detection exists, the candidate
**nearest the base cube's initial pixel**; the mean/<80 px rule survives
only as a fallback branch when the base initial pixel is missing. The code
rule is global (not per-episode), but the changelog — the audit trail —
describes a different rule from the one that ran.

### 2b. Global calibration transforms

- **Affine frame correction: VERIFIED GLOBAL.** One matrix `_M` loaded from
  `scenes/calibration_fk_affine.json` in `build_scenes.py` (module level),
  applied to every Builder B pose via `correct_xy()` / `_affine_px()` /
  `_correct_yaw()`; no task or episode branches. The JSON records provenance,
  n=93 pairs, median residual 1.4 cm, 83% <3 cm. The tie-point source data
  exists (`scenes/fk_tiepoints.json`, produced by `code/fk_tiepoints.py`).
  PASS on globality and on having a data artifact.
- **Gripper-channel inversion: GLOBAL in code, but the evidence trail is
  narrative only.** Implemented once, in `closedness_trace`
  (`replay.py:29-48`): `closedness = clip((hi - v)/(hi - lo))` with the same
  per-episode p5/p99 normalisation for all episodes. No per-episode branch.
  However: **there is no data-driven proof artifact for the inversion** — no
  script, no output file, no transit-timing table. The entire evidence is
  prose in the `replay.py:31-42` docstring and changelog #4, citing ep411
  (transit frames ~124–225 vs low phase 117–227) and ep7 (frame 177, trace
  1°, ring "visibly pinched"). V-CODE spot-checked the traces: ep411 gt is
  276 frames with gripper p5/p99 = 1.1°/25.9° and ep7 272 frames at
  0.7°/16.4°, consistent with the quoted magnitudes, but the decisive
  transit-timing comparison (FK site position vs trace phase) is not
  reproduced by any artifact in the workdir and could not be re-derived
  from saved outputs. For a change that reverses the meaning of the grasp
  signal for all 97 episodes, "proof by docstring" is below the
  pre-registration's evidence standard (honesty rule 1: every parameter
  carries a source with method stated — the method here is stated, the
  data is not kept).

## 3. Changelog integrity

**F-09 (MAJOR) — Changelog timestamps are irreconcilable with the machine's
own timestamps.** All files are CST (DGX clock: CST, −0600).
`validation/run1.log` 18:26, `validation_results_run1.json` 18:33,
`run2.log` 18:34, `validation_results_run2.json` 18:37, `run3.log` and final
`validation_results.json` 18:39, `code/replay.py`/`build_scenes.py` last
modified 18:37, `param-changelog.md` itself written 18:41. The changelog
nevertheless dates entry #3 to ~19:00, #4 to ~19:20, #5 to ~19:50, #6 to
~20:10 and #7 ("VALIDATION COMPLETE") to ~20:30 — i.e. the final code, all
three full runs, and the changelog file itself all exist **before** the
times at which the changelog says the changes they embody were made. Either
the changelog times are wrong by ~1.5–2 h, or they record something other
than when the work happened. As an audit trail, the changelog's "When"
column cannot be relied on.

**What does cohere:** exactly three full runs are evidenced (three result
JSONs + three logs, 97 episodes / 485 variant jobs each), and the summary
numbers move consistently with the changelog's *sequence*: run1 eye 0/20,
ring 0/20 → run2 eye 4/20, ring 0/20 → run3 (final) eye 4/20, ring 1/20;
changelog #7's "ring 0/20 → 1/20, others unchanged" matches run2→run3
exactly, and #6 (peg jitter) is present in the current `build_scenes.py`
(`make_draws` fixture_jitter) and `replay.py` (peg read from sim body).
Every changelog entry #1–#6 corresponds to code that exists and is global
in scope: #1 affine (`build_scenes.py`), #2 dwell attach (superseded by #5,
as stated), #3 detection-pixel block poses + EYE_CARTON_OFFSET
(`build_scenes.py`, global constant (−0.014, −0.093)), #3b/#4 parallax
(`PARALLAX_K`, applied in `load_episode_poses`/`load_fixtures` by height
class, global formula), #4 inversion (`replay.py:29-48`), #5 grasp v3
(`replay.py` state machine), #6 peg jitter (`build_scenes.py make_draws`).

**F-10 (MODERATE) — `param_hash` does not cover the parameters that
changed.** `build_scenes.py: param_table_hash()` hashes only
`twin_params.py` + `docs/physics-parameters.md`. All three run files carry
the identical hash `5ef4c692a314e87b` despite runs 1→3 embodying changelog
#4/#5/#6 changes that live in `replay.py` and `build_scenes.py` — the two
files the hash excludes. The hash therefore cannot distinguish the three
runs' code states and gives false assurance of parameter identity.

**F-11 (MODERATE) — Changelog #6 is a change the amendments do not
authorize in that form.** Amendment 1 A2 authorises jitter of "object
initial pose … friction and mass". Entry #6 jitters the **peg + board
(the outcome reference itself)** per variant and re-reads the criterion
against the moved peg (`replay.py:373-378`). Moving the reference with the
scene is internally consistent, but it is not the registered sampling
design: the variant family no longer tests robustness to a fixed recorded
reference; it tests robustness to a reference resampled from the same
estimator's σ. It is logged and global — but it changes what a "variant
agreement" means for ring_insert, and it was introduced after run 2 in
response to run 2's miss pattern (a systematic ~3.4 cm planar-to-peg
offset), i.e. it is outcome-motivated reference motion, the pattern
Amendment 1's "never parameter-tuned to force a pass" clause guards
against. Effect was small (ring 0→1 reproduced), but the precedent is the
finding.

## 4. Exclusions (the 12 variants)

The 12 excluded variants, from `validation_results.json` build_defects:
eye_drops ×4 (ep1780 v4, ep1783 v2, ep1808 v4, ep1814 v1 — all
"carton: displaced 0.090 m / z=0.017 after pre-settle") and tool_insert ×8
(ep885 v0/v1/v3, ep888 v0/v4, ep892 v0/v4, ep903 v0 — all
"tool: displaced 0.034–0.038 m after pre-settle"). Mechanism:
`replay.py:110-124` flags a build defect if, after a 0.25 s pre-settle, any
weldable body moved >0.03 m or z < −0.02 or is NaN;
`run_validation.py:78-84` then removes flagged variants from the valid set
(outcome-blind: the flag is computed before the trajectory runs). The
episode-level exclusions (tool ep630/871/876, low initial-pose confidence)
follow the pre-registration's honesty rule 3 and are named in the output.

**F-12 (MODERATE) — The defect trigger is a silent threshold that only
approximates Amendment 1's grounds, and it is asymmetric in effect.**
Amendment 1 permits exclusion only for "spawn interpenetration, initial
instability, solver failure". The code's proxy — >3 cm displacement in
0.25 s — is nowhere in the amendments. The eye cases (exactly 0.090 m =
one platform height) look like genuine unsupported-spawn falls and are
plausibly "initial instability". The tool cases (3.4–3.8 cm, normal z) are
a capsule tool sliding/rolling just past an arbitrary line; whether that is
instability or simply a wrong spawn pose expressing itself is not
distinguishable from the record. Two structural problems: (i) **all 12
excluded variants have sim_success = False and all 12 belong to episodes
whose recorded outcome is SUCCESS** — exclusion therefore only ever removes
disagreements from denominators (e.g. tool ep885 is scored on 2 of 5
variants). No episode's reproduced status changes because of this (all
affected episodes are non-reproduced regardless), so the headline numbers
are not rescued by it — but the mechanism, as coded, can only ever help.
(ii) **Solver failure during the trajectory is never detected**: the NaN
check runs only after pre-settle. `run2.log`/`run3.log` contain MuJoCo
"Nan, Inf or huge value in QACC … simulation is unstable" warnings, and
changelog #7 itself notes chain explosions in cable runs — yet cable_clip
counts 100/100 valid variants and 20/20 reproduced, every one an agreement
between a recorded FAILURE and a sim FAILURE. A variant that exploded
mid-run is "valid" under this code; under Amendment 1 it is the canonical
excludable case. The exclusion code thus excludes on a ground the amendment
does not name (pre-settle slide) and fails to exclude on a ground it does
name (solver failure).

## 5. Determinism

**PASS.** Regenerated all 5 variants for block_stack ep411 and ring_insert
ep7 from the current code (`make_draws` + `build` in-process, nice -n 10,
CPU-only): all 10 XML files are **byte-identical** to the on-disk scenes
(SHA-256 prefixes matched), and all 10 draw records match the on-disk
`*_variants.json` exactly. `make_draws` (`build_scenes.py`) seeds via
`SeedSequence([VARIANT_SEED=20261122, crc32(task__ep), k])` — deterministic
and independent of run order/worker count. Caveat: determinism is proven
for scene *generation*; full replay determinism was not re-run end-to-end
(MuJoCo stepping is deterministic for a fixed binary/inputs, but no
replay-repeat check was performed in this audit).

Related sampling deviation, **F-13 (MINOR)**: Amendment 1 says pose jitter
is "within Builder B's per-pose quantitative error estimate". The code
draws **Gaussian** N(0, σ) samples (`build_scenes.py make_draws`), which
are unbounded — roughly a third of draws exceed 1σ by construction.
Friction (uniform ±20%) and mass (uniform in sourced ranges) match the
amendment's sampling language; pose jitter does not, in distribution shape.
Also `twin_params.py:126 JITTER = {high: (0.005, 3°), medium: (0.010, 6°)}`
is dead code — defined, documented as the tier scales, and never used;
the actual sigmas come from Builder B's per-pose fields with a silent
fallback of 3.5 cm (`make_draws`: `o.get("sigma_xy_cm") or 3.5`) for any
pose lacking one.

## Summary of verdict basis

Certified sub-items: settle time; 60%/80% arithmetic; affine transform
globality + tie-point artifact; inversion globality in code; scene
generation determinism (10/10 byte-identical); no episode conditionals in
the scoring path; three-run structure of the results files.

Blocking findings: F-01 (contact/support criteria replaced by proxies,
dead contact code), F-02 (secondary criterion never thresholded; all 4
reproduced eye episodes violate it unflagged), F-05 (20 vs 30 fps
unresolved), F-06 (grasp device is not the registered device; stale
parameter-file documentation), F-09 (changelog timestamps irreconcilable
with file/run timestamps).

Non-blocking but recorded: F-03, F-04, F-07, F-08, F-10, F-11, F-12, F-13.

Remediation path to CERTIFIED-WITH-FINDINGS or better: implement or
formally amend the contact/support clauses; threshold and report the
secondary criterion per episode; produce a kept, runnable artifact for the
gripper-inversion proof and for the dataset frame rate; correct the
changelog timestamps and F-08's rule description; extend `param_hash` to
all code that affects outcomes; move the build-defect threshold and the
peg-jitter design into a written amendment or remove them.
