# Twin-Suite v2 Stage 1 — Controls Battery, Part B (2026-11-23)

**Scope:** the three registered control checkpoints (Part A controls
manifest, CERTIFIED) run through the gate protocol on the 20
solvability-validated scenes, both engines — L1 nominal MuJoCo (300
rollouts), L2 envelope corners MuJoCo (120), L3 Drake nominal (15).
**Total: 435/435 rollouts recorded.** This is a builder's report; it
requires independent verification in a separate context before it is
cited as settled (pre-reg §9). Every gate output carries the §8 label:
**"funnel verdict — sim bar unanchored to real-arm delta (no
calibration pairs yet)."**

## Verdict up front

- **§3.4 known-ordering control: FAILS as registered.** Base fails
  every task (0/100 pooled, 0 on every task) — that half holds. But
  FL round-30 and C-exp **also score 0 on every task** under the
  primary mechanical criteria, so the ordering base < FL < C-exp does
  **not** reproduce, pooled or per task. Under the pre-reg's
  cross-modality procedure (§3.4 steps 1–5), no modality mechanism
  was **measured** in this battery that could excuse a pair, so there
  is no downgrade on this evidence: the ordering control fails, and
  no precision task's instrument is certified by this battery.
- **§3.2 cross-engine agreement: 15/15 (100%)** — every paired unit
  agrees, because every unit fails in both engines. Agreement at the
  floor is consistency evidence, not discrimination evidence.
- **§3.3 envelope stability: 0 outcome flips in 120 corner rollouts.**
  Verdict classes are stable for block_stack; for eye_drops the
  per-corner Wilson class reads INDETERMINATE at n=1/corner by
  construction (see §4) — pooled across corners it is FAIL, matching
  nominal. No rollout changed outcome at any corner.
- **Solver failures: 0/435. Invalidated Drake units: 0/15** (every
  Drake record's `frame_content` is present and non-blank).

## 1. Execution record

- **Code (DGX, v2 workdir `controls_battery/`):** `battery.py`
  (scene generation + worklist + resumable worker), `analyze.py`,
  `harness/sim_runner_battery.py` — a copy of the certified
  `sim_runner_padded.py` with exactly these changes (diff-verified):
  scenes dir → `controls_battery/scenes-battery/`; `--corner` /
  `--scene-file` arguments; corner in the record key and record;
  `engine: "mujoco"` in the record; corner in the trajectory filename.
  **No v1.1 file was modified**; cable scenes are v1.1 scene copies
  read under the v1.1 `replay.evaluate()` scoring (the Stage-0
  instrument's criteria + solver detection), welds forced inactive
  as in every policy rollout.
- **Scenes generated (160 files, all compile-checked in MuJoCo
  4.0.1 before launch):** for each of the 20 validated episodes, the
  nominal + v0–v4 variant XMLs copied from the v1.1 suite with only
  the arm `<include>` swapped to the padded v2 arm (the reference
  controllers' `ensure_scene()` convention); plus, for the 5
  envelope precision scenes, the 8 non-nominal corner scenes each,
  applying exactly one deviation from `envelope-corners-v2.json`
  (pose ±σ on the manipulated objects' initial positions; sliding
  friction set per-geom to the corner low/high values — floor, mat,
  walls and the task's object/fixture geoms; manipulated-object mass
  set to the range low/high). **Registered limitation:** the harness
  pins finger friction at runtime (`prepare_model`), so the gripper
  component of the friction corners is not exercised in MuJoCo; all
  scene-geom components are.
- **K convention:** L1's K=5 = the frozen variant set k=0..4; k=0's
  draws are the frozen empty-jitter draws, so **variant "0" is the
  nominal instance** — it is the unit paired against L3's Drake
  nominal (DrakeScene variant 0, same draws) in §3.
- **Full-path smoke before the battery** (one unit per layer, same
  guarded path): L1 base @ block 411 v0 (FAIL, 49 s), L2 base @
  block 499 friction_low (FAIL, recorded with corner field), L3 base
  @ block 499 Drake (FAIL; `frame_content` chunk-0 unique colours
  front 1,122 / top 880 / wrist 1,321, min non-clear ≥ 0.58 — the
  sighted renderer of the certified fix). The smoke caught one
  builder defect — the friction-corner generator dropped geom tag
  closures — fixed and all 160 scenes re-generated and
  compile-verified before the battery launched. Smoke units ran the
  final protocol and count in the 435.
- **Guard discipline:** four streams (one per MuJoCo checkpoint,
  one Drake), each a single launch via `launch_dgx_job.sh` +
  `job_guard.sh`, mode logonly, `JOB_LOCK=0`, CPU-only, `nice -n 10`
  (MuJoCo) / 19 (Drake). Guard exit_code **0 for all four rundirs**
  (`runs/controls-battery-{base,fl_round_30_a5repair,c_exp,drake}`);
  **zero WOULD-KILL lines** in any guard log. Mirror:
  `JOB_GUARD_MIRROR_TARGET` was set to the precedent off-box target
  on every launch; every guard log records **MIRROR FAIL rc=255**
  (ssh connection refused — the known infrastructure gap); **no
  no-mirror override was used**; telemetry is on-box only. DGX
  training was not disturbed (GPU 0–1%, box temp ≤ 59 °C in our
  telemetry; the box was otherwise quiet).
- **Wall time:** streams launched 04:49 CT, last stream done
  09:07 CT (≈4.3 h for the battery, parallel streams; per-rollout
  mean wall: L1 93 s, L2 67 s, L3 70 s; cable_clip units were the
  long tail at up to ~285 s). Completion manifest
  (`controls_battery/completion-manifest.json`): **435/435** —
  L1 100/100 per checkpoint, L2 40/40 per checkpoint, L3 5/5 per
  checkpoint. Records persisted incrementally (`rollouts.jsonl`,
  fsync per record) with per-rollout trajectory `.npz` + chunk-0
  frame PNGs retained in `controls_battery/out/` for the second
  judging layer.
- **Checkpoints** (Part A manifest pins, unchanged): base
  `lerobot/smolvla_base` @ `5e8d12a6…` (weights `7cd549ac…`); FL
  `checkpoint_round_30_a5repair` (weights `5b1cbf29…`); C-exp
  `015000` (weights `8f6807ed…`).

## 2. L1 nominal results (§3.4 + §5 control readouts)

Successes / n (solver-failure-excluded denominator; solver failures
were 0 everywhere), Wilson 95% interval, §5 verdict class. These are
**control readouts, not candidate gates**.

| Checkpoint | Task | Success | Rate | Wilson 95% | §5 class |
|---|---|---|---|---|---|
| base | block_stack (8 scenes × K5) | 0/40 | 0% | [0, 8.8%] | FAIL |
| base | eye_drops_to_basket (4 × K5) | 0/20 | 0% | [0, 16.1%] | FAIL |
| base | tool_insert (1 × K5) | 0/5 | 0% | [0, 43.4%] | FAIL |
| base | cable_clip (7 × K5) | 0/35 | 0% | [0, 9.9%] | FAIL |
| base | **pooled** | **0/100** | 0% | [0, 3.7%] | FAIL |
| FL round-30 | block_stack | 0/40 | 0% | [0, 8.8%] | FAIL |
| FL round-30 | eye_drops_to_basket | 0/20 | 0% | [0, 16.1%] | FAIL |
| FL round-30 | tool_insert | 0/5 | 0% | [0, 43.4%] | FAIL |
| FL round-30 | cable_clip | 0/35 | 0% | [0, 9.9%] | FAIL |
| FL round-30 | **pooled** | **0/100** | 0% | [0, 3.7%] | FAIL |
| C-exp | block_stack | 0/40 | 0% | [0, 8.8%] | FAIL |
| C-exp | eye_drops_to_basket | 0/20 | 0% | [0, 16.1%] | FAIL |
| C-exp | tool_insert | 0/5 | 0% | [0, 43.4%] | FAIL |
| C-exp | cable_clip | 0/35 | 0% | [0, 9.9%] | FAIL |
| C-exp | **pooled** | **0/100** | 0% | [0, 3.7%] | FAIL |

ring_insert: **no gate** — 0 validated scenes (A2.2(iii)), 0 rollouts,
reported per the registered consequence, not silently dropped.

**Secondary criterion (reported, not gated):** block_stack secondary
passes — base 3/40, FL 5/40, C-exp 7/40; all other tasks 0 for every
checkpoint in L1 (L2 secondary: base 1, FL 1, C-exp 2 of 40; L3: 0).
The secondary ordering is monotone in the expected direction on the
one task that shows any signal, at magnitudes far below the primary
bar.

**The zero is behaviour, not a dead harness** (checked before this
report was written): per-stream policy services loaded the correct
manifest checkpoint paths; recorded trajectories show large arm
motion (e.g. C-exp @ block 411 v0: per-joint ranges 0.41–1.67 rad
over 296 steps); objects end essentially unmoved (that record's
cube_a ends 4 mm from its start, `upper_contacts: [world]`) — the
policies approach but never form the pinch-lift / placement the
contact criteria require, in either engine, at any corner. The
scenes are solvable — by the oracle-guided reference executors of
Part A, which consume ground-truth state and IK planning; no policy
in this battery had either.

## 3. §3.2 cross-engine agreement (L3 vs L1 nominal instance)

15 paired units (5 envelope precision scenes × 3 checkpoints),
paired on (success, solver_failure) outcome class:

| Checkpoint | Paired | Agree |
|---|---|---|
| base | 5 | 5 |
| FL round-30 | 5 | 5 |
| C-exp | 5 | 5 |
| **Total** | **15** | **15 (100%)** |

Disagreements: **none to itemize** — every unit is FAIL / no solver
failure in both engines. All 15 Drake records carry non-blank
`frame_content` (no unit invalidated). Recorded plainly: with all
outcomes at the floor, this agreement certifies that the two engines
do not diverge on these scenes; it carries no evidence that they
would agree on a succeeding rollout.

## 4. §3.3 envelope stability (L2)

Nominal verdicts below are computed on L1 restricted to the envelope
scenes (block 4 scenes × K5 = 20 rollouts; eye 1 × K5 = 5). Corner
verdicts use that corner's rollouts (block n=4, eye n=1).

| Checkpoint × task | Nominal class | Corner outcome flips | Corner classes (8 corners) | Stable? |
|---|---|---|---|---|
| base × block | FAIL (0/20) | 0 | FAIL ×8 | yes |
| FL × block | FAIL (0/20) | 0 | FAIL ×8 | yes |
| C-exp × block | FAIL (0/20) | 0 | FAIL ×8 | yes |
| base × eye | FAIL (0/5) | 0 | INDETERMINATE ×8 (n=1) | see note |
| FL × eye | FAIL (0/5) | 0 | INDETERMINATE ×8 (n=1) | see note |
| C-exp × eye | FAIL (0/5) | 0 | INDETERMINATE ×8 (n=1) | see note |

**Note (the eye rows):** at n=1 with 0 successes the Wilson upper
bound is 79%, so the §5 class formula returns INDETERMINATE for
every eye corner — a small-n artifact of the class formula, not an
outcome change: **no eye rollout succeeded at any corner, and none
flipped**. Pooled across the 8 corners (eye n=8, 0 successes, Wilson
upper 32.4%) the corner class is FAIL, matching nominal; block
pooled corners (n=32, 0 successes) likewise FAIL. Friction corners
are [DEFAULT]-at-corner per §3.1 — flagged here as required; no
verdict's stability in this battery depends on a friction corner in
any case, since no outcome moved. Mass corners for block carry the
OI-2 material-contingent flag from the corners register.

## 5. §3.4 known-ordering control — the registered procedure, applied

Reference (Amendment 1 A1.1): judged completion base 0/100 <
FL 26/100 < C-exp 70/100, from open-loop offline replay. The
**cross-modality assumption** is stated as the pre-reg requires:
that ordering comes from a different modality than this gate's
closed-loop sim rollouts, and a mismatch is evidence against the
instrument only after modality effects are excluded by the
registered steps:

1. **Base-fails-all: HOLDS.** Base is 0 on every task with a
   denominator (and its §5 class is FAIL everywhere).
2. **Failure-taxonomy comparison:** on the primary criterion the
   taxonomies are indistinguishable — all three checkpoints fail
   the same contact/support checks (`contact_with_lower`,
   `within_inset_footprint`, `cable_seated_in_clip_closed`, … all
   false; objects left at/near start poses). The one differentiating
   signal is the secondary criterion on block_stack (3 < 5 < 7),
   which is monotone in the reference direction but is not the gated
   criterion and is two orders of magnitude below the reference
   separation (26 vs 70).
3. **Excuse test:** a pair may be excused only if a specific
   modality mechanism is **documented with measurements** (the
   pre-reg's example: closed-loop compounding divergence measured
   as trajectory error growth, where the open-loop signature shows
   no corresponding defect). **This battery collected no such
   measurement** — trajectory error growth against the recorded
   demos was not computed, and no other mechanism was measured.
   Candidate mechanisms exist as *unmeasured hypotheses* (twin pose
   offset vs the policies' training distribution; the measured
   ~24 mm site→pinch-centre offset making the learned grasp
   geometry land off-centre; cable's policy-side manipulation
   having no weld device while its solvability proof used one) —
   none of them is evidence under the registered standard today.
4–5. **Therefore:** no pair is excused, the downgrade path is not
   available on this evidence, and the control **FAILS** — per the
   pre-reg there is no third outcome. Consequences, stated plainly:
   the §3.4 standard is not met for block_stack,
   eye_drops_to_basket, tool_insert, or cable_clip; **no task is
   instrument-certified by this battery**, and no non-control
   checkpoint may be gated (§6) until certification lands by some
   route. tool_insert's read is additionally flagged
   insufficient-denominator (1 scene, n=5) per Part A; ring_insert
   has no gate at all (0 validated scenes).

## 6. §7 Bounds Amendment — DRAFT (not appended to the pre-registration)

Derived from this battery's control dispersion exactly per the §7
procedure (bounds measured from controls, never fitted to
candidates). **This draft is not frozen**; it is appended to the
pre-registration only after independent verification of this
report.

> **Bounds Amendment (DRAFT, 2026-11-23, from the Stage 1 controls
> battery — 435 rollouts, completion manifest 435/435):**
>
> 1. **Cross-engine policy-outcome agreement bound (per task):**
>    disagreement rate ≤ **20%** of paired units. Derivation:
>    observed disagreement 0/15 paired units; the Wilson 95% upper
>    bound of a 0/15 disagreement rate is 20.4%, rounded down to
>    20%. (Agreement observed: 15/15, all at FAIL outcomes; the
>    bound inherits that floor caveat.)
> 2. **Ordering-control noise band (per task):** two checkpoints
>    are distinguishable on a task only if their pooled primary
>    success rates differ by more than the task band, set to the
>    Wilson 95% upper bound of a zero rate at the task's validated
>    rollout count in this battery: **block_stack 8.8 pp** (n=40),
>    **eye_drops_to_basket 16.1 pp** (n=20), **cable_clip 9.9 pp**
>    (n=35), **tool_insert 43.4 pp** (n=5 — recorded as
>    uninformative at its current denominator), pooled 3.7 pp
>    (n=100). Observed control separations in this battery: 0 pp on
>    every task — inside every band, consistent with the §3.4 FAIL
>    in §5 of the battery report.
> 3. **Envelope-corner verdict-stability criterion:** the §3.3
>    qualitative criterion stands (a checkpoint's §5 verdict class
>    on a task must be unchanged across the 8 frozen corners);
>    measured control dispersion behind it: **0 outcome flips in
>    120 corner rollouts**, 0 verdict-class flips at denominators
>    ≥4/corner. Corner classes at n=1/corner are unresolvable under
>    the Wilson formula and must be read pooled across corners.
> 4. **Mechanical/judging disagreement-rate bound: NOT SET by this
>    battery.** The frontier judging layer (§4 second layer) was
>    not run; all 435 rollout artifacts (trajectory `.npz` +
>    chunk-0 frames + records) are retained for it. The bound is
>    set from the judging layer's measured disagreement on these
>    same control rollouts before any candidate is gated, or the
>    amendment records it as an open item — it is not invented here.

## 7. What this battery does not establish

- It does not certify any task's instrument (§3.4 failed), and it
  authorizes no candidate gate run.
- The all-zero primary rates are a statement about these three
  checkpoints in this instrument under this protocol — not a
  real-world success rate for any checkpoint (§8 label applies),
  and not evidence about the checkpoints' relative quality beyond
  the secondary-criterion signal noted in §2.
- The §3.4 downgrade route remains open **only** via its registered
  evidence standard: a measured modality mechanism (e.g. trajectory
  error-growth analysis of these retained rollouts against the
  recorded demos, or a measured grasp-geometry offset analysis).
  That analysis was out of this battery's scope and is the obvious
  next step for the parent/founder to commission — or not.
- Builder ≠ verifier: certification verdicts on this battery belong
  to an independent verifier re-running from the raw records
  (`controls_battery/out/`, `analysis.json`, completion manifest).
