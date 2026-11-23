# Twin-Suite v2 — Stage 1 Certification Synthesis (Controls Battery Part B)

**Date:** 2026-11-23 · **Builder context:** synthesis agent (separate
verifier required per pre-registration §9 — this report is a builder
deliverable, NOT a certification by the verifier).
**Governing:** `twin-suite-v2-preregistration-20261122.md` (FROZEN,
incl. Amendments 1–2), especially §3.1–§3.4, §4, §5, §7, §8.
**Inputs (raw only):** DGX
`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/controls_battery/`
— `out/{L1,L2,L3}/{base,fl_round_30_a5repair,c_exp}/rollouts.jsonl` +
`traj_*.npz`, `completion-manifest.json` (435/435, all guard exit
codes 0), `worklist.jsonl` (the registered 435-unit matrix),
`envelope-corners-v2.json`; validated register and matrix per
`controls-battery-prep-v2.md` §2/§4 (CERTIFIED by
`verification-v2-controls-prep.md`).
**Method:** every number below was recomputed by the synthesis agent
from the raw `rollouts.jsonl` records and `traj_*.npz` files, using
the battery's registered analysis (`controls_battery/analyze.py`:
Wilson score intervals, z = 1.96; solver-failure rollouts excluded
from denominators, counted and reported; §5 verdict rule applied to
pooled successes). Where this synthesis extends `analyze.py` it says
so explicitly (§6, §9). No summary written by anyone else was used
as a numeric source.

> **Every gate output in this report carries the §8 label: funnel
> verdict — sim bar unanchored to real-arm delta (no calibration
> pairs yet).**

## 0. Headline

**No precision task is instrument-certified under Stage 1.** The
battery executed completely and cleanly (435/435 rollouts, **zero
solver failures and zero excluded units in all 435 records**), and
three of the four Stage 1 standards behave well — the parameter
audit stands (§3.1), envelope verdicts never flip (§3.3), and
cross-engine *outcome* agreement is perfect on the paired subset
(§3.2, 15/15). The certification fails on the fourth standard:
**the known-ordering control (§3.4) does not reproduce.** Every
checkpoint — base, FL round-30, and C-exp — scored **0 mechanical
successes in every L1 task cell** (0/300 L1 rollouts in total).
Base-fails-all holds (A1.4), but the strict ordering
base < FL < C-exp does not (0 = 0 = 0 everywhere), and the
pre-registration's registered modality-excuse procedure (§3.4
steps 1–5) produces no documented, measured mechanism that excuses
any checkpoint pair. Under §3.4 step (5) the control therefore
**FAILS**, and a task failing any standard "stays research-grade
and out of the gate" (§3 preamble). ring_insert has no gate at all
(A2.2(iii)); tool_insert's denominator is one scene.

This is a floor result, not an instrument-crash result: the
policies genuinely interact with the scenes (object displacements
up to ~13 cm, one C-exp block rollout lifting a cube 5.5 cm), the
scenes are solvability-validated by executed reference passes
(A2.2), and the secondary (near-miss) criterion *does* order the
checkpoints base < FL < C-exp on block_stack and pooled (§5.3) —
but the registered primary criterion is the mechanical pass, and on
it the gate cannot distinguish a checkpoint that scored 70/100 in
offline replay from the untrained base. That is exactly the
question §3.4 exists to answer, and the answer is no.

## 1. Battery integrity

| Check | Result (recomputed) |
|---|---|
| Units completed | 435/435 (`completion-manifest.json`: L1 3×100, L2 3×40, L3 3×5) |
| Records read | L1 300, L2 120, L3 15 — matches `worklist.jsonl` (435 lines) |
| Solver failures | **0 / 435** (L1 0, L2 0, L3 0) |
| Records flagged `excluded` | **0 / 435** — no denominator is reduced by exclusion anywhere in this report |
| Drake invalid-frame units (`analyze.py` `drake_valid` rule) | **0 / 15** — all L3 records pass (chunk-0 unique colours 848–1,376 per view; min non-clear fraction 0.55–1.00; thresholds are >10 colours and >0.05) |

Validated denominators (A2.2, prep §2, CERTIFIED): block_stack 8
scenes, eye_drops_to_basket 4, tool_insert 1, cable_clip 7,
ring_insert 0. L1 = validated scenes × K=5 variants × 3 checkpoints.

## 2. L1 — nominal gate protocol, MuJoCo (§3.4 + §4)

Mechanical pass = `sim_success`. Wilson 95% intervals per `analyze.py`.
Verdicts per §5 (PASS: lower ≥ 60%; FAIL: upper < 60%).

### 2.1 Per task × checkpoint

| Checkpoint | Task | Success | n | Solver-fail | Rate | Wilson 95% | §5 verdict |
|---|---|---|---|---|---|---|---|
| base | block_stack | 0 | 40 | 0 | 0.000 | [0.000, 0.088] | FAIL |
| base | eye_drops_to_basket | 0 | 20 | 0 | 0.000 | [0.000, 0.161] | FAIL |
| base | tool_insert | 0 | 5 | 0 | 0.000 | [0.000, 0.435] | FAIL |
| base | cable_clip | 0 | 35 | 0 | 0.000 | [0.000, 0.099] | FAIL |
| base | ring_insert | — | 0 | — | — | — | NO GATE (0 validated scenes) |
| base | **POOLED** | **0** | **100** | 0 | 0.000 | [0.000, 0.037] | FAIL |
| FL round-30 | block_stack | 0 | 40 | 0 | 0.000 | [0.000, 0.088] | FAIL |
| FL round-30 | eye_drops_to_basket | 0 | 20 | 0 | 0.000 | [0.000, 0.161] | FAIL |
| FL round-30 | tool_insert | 0 | 5 | 0 | 0.000 | [0.000, 0.435] | FAIL |
| FL round-30 | cable_clip | 0 | 35 | 0 | 0.000 | [0.000, 0.099] | FAIL |
| FL round-30 | ring_insert | — | 0 | — | — | — | NO GATE |
| FL round-30 | **POOLED** | **0** | **100** | 0 | 0.000 | [0.000, 0.037] | FAIL |
| C-exp | block_stack | 0 | 40 | 0 | 0.000 | [0.000, 0.088] | FAIL |
| C-exp | eye_drops_to_basket | 0 | 20 | 0 | 0.000 | [0.000, 0.161] | FAIL |
| C-exp | tool_insert | 0 | 5 | 0 | 0.000 | [0.000, 0.435] | FAIL |
| C-exp | cable_clip | 0 | 35 | 0 | 0.000 | [0.000, 0.099] | FAIL |
| C-exp | ring_insert | — | 0 | — | — | — | NO GATE |
| C-exp | **POOLED** | **0** | **100** | 0 | 0.000 | [0.000, 0.037] | FAIL |

Per-episode detail (all episodes, all checkpoints): every episode
cell is 0/5 mechanical successes. The full per-episode table is in
the machine-readable summary (`stage1-certification-summary-v2.json`).

### 2.2 What the trajectories show (failure taxonomy, §3.4 step 2)

Recomputed from all 300 L1 `traj_*.npz` files (extension beyond
`analyze.py`, which does not read trajectories):

- **The policies are not inert.** Maximum object displacement per
  rollout reaches 8.6 cm (base, one block variant), 9.0 cm (FL),
  9.8 cm (C-exp block variants). Maximum cube z-rise over L1
  block_stack: base 1.3 cm, FL 1.7 cm, C-exp **5.5 cm** — at least
  one C-exp rollout genuinely lifted a cube above stack height,
  yet no rollout satisfied all mechanical criteria at settle
  (inset footprint + contact-with-lower + no other support).
- **eye_drops_to_basket: the carton never moves at all** — max
  z-rise 0.0 cm and zero net displacement in all 60 L1 rollouts
  (all checkpoints, all variants) and in Drake (§4). Every eye
  failure is a no-contact failure in both engines.
- **tool_insert** (ep683 only): the tool is displaced and lifted
  (max z-rise 3.8–4.8 cm across checkpoints) but never seated
  (`centre_in_slot` / tray-contact criteria unmet in all 15).
- **cable_clip** runs under its v1.1 Stage 0 instrument (prep §4);
  0/35 per checkpoint here is a gate-protocol exercise result, not
  a change to cable_clip's v1.1 certification (see §7).

### 2.3 Secondary criterion (reported, not gated — §4)

Secondary passes, L1: block_stack base 3/40, FL 5/40, C-exp 7/40;
all other tasks 0 for all checkpoints; pooled base 3/100, FL 5/100,
C-exp 7/100. The near-miss criterion orders the checkpoints
correctly; the primary criterion does not separate them at all.
This contrast is diagnostic evidence about *where* the floor sits
(the last stage of the task — seating/stacking at settle), not a
substitute verdict: §4 makes the mechanical criteria primary.

## 3. Ordering control (§3.4) — **FAIL**

Registered requirements (per task): (i) base FAILS every task;
(ii) base < FL round-30 < C-exp within the §7 noise band, against
the verified offline-replay reference 0/100 < 26/100 < 70/100 (A1.1).

| Task | base | FL r30 | C-exp | Base fails? | Strict order? |
|---|---|---|---|---|---|
| block_stack | 0/40 | 0/40 | 0/40 | ✓ | ✗ (0 = 0 = 0) |
| eye_drops_to_basket | 0/20 | 0/20 | 0/20 | ✓ | ✗ |
| tool_insert | 0/5 | 0/5 | 0/5 | ✓ | ✗ |
| cable_clip (v1.1 instrument, A1.4) | 0/35 | 0/35 | 0/35 | ✓ | ✗ |
| ring_insert | no scenes | no scenes | no scenes | vacuous | not testable |
| **POOLED** | **0/100** | **0/100** | **0/100** | ✓ | ✗ |

**Registered modality-excuse procedure (§3.4 steps 1–5), worked
through on the record:**

1. *Base-fails-all:* **holds** — base is 0 in every cell including
   cable_clip (A1.4 scope).
2. *Failure-taxonomy comparison:* offline replay's C-exp record is
   70/100 judged completions; in closed-loop sim, C-exp's failures
   are concentrated at the final mechanical stage — objects are
   reached, displaced, occasionally lifted (§2.2), but never left
   satisfying the contact/support criteria at settle; eye_drops
   failures are no-contact in 60/60 rollouts. The taxonomies differ
   in kind (open-loop end-state closeness vs closed-loop contact
   outcome), which is expected — §3.4 anticipates this and requires
   step 3, not a narrative.
3. *Excuse requires a specific modality mechanism documented with
   measurements* (the registered example: closed-loop compounding
   divergence, measured as trajectory error growth, with no
   corresponding open-loop defect). **No such measurement exists in
   this battery's records, and this synthesis does not manufacture
   one.** The trajectory data show policy–object interaction, not
   divergence growth against a reference trajectory; no
   error-growth quantity was registered or computed by the battery.
   The two candidate mechanisms visible in the wider v2 record —
   the Drake approach-path contact lock and the MuJoCo
   site-vs-pinch-centre offset (~24 mm, fingerpad verification) —
   are executor/grasp-geometry facts about *scripted* probes, were
   not measured on these policy rollouts, and in any case the floor
   result is identical in MuJoCo, where C-exp was expected to be
   least affected. They do not meet the step-3 standard as recorded
   evidence for excusing a pair.
4. *Downgrade path:* available **only if** a pair is excused under
   step 3. No pair is excused, so the downgrade (base-fails-all +
   cross-engine agreement) is **not** taken.
5. *Otherwise:* "the control **FAILS** and the task is not
   instrument-certified."

**Ordering-control verdict: FAIL** — for block_stack,
eye_drops_to_basket, tool_insert, and (as a gate exercise)
cable_clip; not testable for ring_insert. This single failure is
sufficient, under §3, to deny Stage 1 certification to every task
the battery exercised, regardless of the other standards' outcomes
below.

## 4. Cross-engine agreement (§3.2) + the Drake watch item — RESOLVED

L3 = the 5 envelope-subset validated precision scenes (block_stack
499/527/546/549, eye_drops 1609) × nominal instance × Drake × 3
checkpoints = 15 rollouts. Comparator = the registered MuJoCo
nominal instance (L1 variant "0", corner nominal) for the same
scene × checkpoint, exactly as `analyze.py` pairs them.

### 4.1 Outcome agreement

| Pairing | Outcome (`sim_success`) agreement | Secondary agreement |
|---|---|---|
| All 15 pairs | **15/15** (all FAIL in both engines) | **15/15** |
| block_stack (12 pairs) | 12/12 | 12/12 |
| eye_drops (3 pairs) | 3/3 | 3/3 |
| Per checkpoint (5 pairs each) | base 5/5, FL 5/5, C-exp 5/5 | 5/5 each |

Invalid Drake units: **0** (§1) — the all-fail pattern is not a
blank-render artifact; every L3 record's `frame_content` shows a
sighted policy (848–1,376 unique colours per view at chunk 0).

### 4.2 The watch item: why does C-exp fail in Drake on validated block_stack scenes?

**Determination: (a) — a genuine policy/task failure that MuJoCo
agrees with, at the level §3.2 registers (policy-outcome
agreement).** It is not (b) a Drake-side grasp/contact artifact
producing the pattern, and not (c) cross-engine disagreement on
identical behavior. Evidence, per scene, from the records and
trajectories (extension beyond `analyze.py`: trajectory
displacement analysis):

| Scene × C-exp | Drake behaviour (traj) | MuJoCo nominal instance | Outcome |
|---|---|---|---|
| block 499 | Both cubes displaced (net 2.6 / 4.4 cm; max excursion 5.1 / 4.7 cm; cube z-max 0.028 / 0.023) | Cubes displaced (net 5.0 / 1.1 cm) | FAIL / FAIL |
| block 527 | cube_a displaced net **12.1 cm** (max 12.8 cm, z-max 0.027) | cube_a displaced net 3.7 cm | FAIL / FAIL |
| block 546 | No object motion | cube_a net 2.3 cm | FAIL / FAIL |
| block 549 | cube_a net 2.4 cm (max 3.0 cm) | cube_a net 0.2 cm | FAIL / FAIL |
| eye 1609 | Carton never moves (net 0.0, z const 0.1075; checks: still on platform) | Carton never moves | FAIL / FAIL |

- **Not (b):** a grasp/contact artifact that *prevented
  interaction* would show unmoved objects in Drake where MuJoCo
  moves them. The opposite pattern holds on the interacting
  scenes: Drake's C-exp moves the cubes as much as or more than
  MuJoCo's (527: 12.1 cm vs 3.7 cm net). The known Drake instrument
  differences are real but do not explain this pattern: the three
  finger-part meshes are excluded in every L3 record
  (`finger_mesh_failed`, the verified Finalize() limitation), yet
  the authored pads + jaw boxes — verified at 0.002 mm cross-engine
  parity — are present, and contact demonstrably occurs. The
  battery-as-specified approach-path contact lock (drake-actuated
  verification §4) is a *scripted-probe* transit phenomenon; these
  are policy rollouts, and their trajectories show contact, not
  lock-out. Eye 1609's no-contact failure is identical in MuJoCo,
  so it cannot be Drake-specific.
- **Not (c):** no pair disagrees on outcome or secondary class
  (15/15, §4.1). Behavior is not *identical* — final positions
  diverge where interaction occurs (cross-engine final-position
  deltas: base ≈0.0001 m on all scenes, since nothing moves; FL up
  to 0.055 m; C-exp up to **0.120 m** on ep527 cube_a) — but §3.2's
  registered quantity is policy-*outcome* agreement, and on that
  quantity the engines agree completely. The trajectory-level
  divergence is recorded here as an instrument fact (contact-rich
  dynamics are not engine-reproducible in detail), consistent with
  the part-1 Drake diagnostics.
- **Therefore (a):** C-exp fails these scenes in Drake because
  C-exp fails them in the twin, full stop — MuJoCo, the primary
  engine, records the same failures on the same instances, and
  across all K=5 variants (0/40 block_stack). The scenes are
  solvability-validated (executed reference passes, A2.2), so the
  failures are policy failures under closed-loop execution, not
  impossible scenes.

### 4.3 §3.2 standard, as far as it can be judged

Policy-outcome agreement on the controls is 15/15 = 100% on the
paired subset. The §7 cross-engine bound itself has not been frozen
(the Bounds Amendment is still owed — §6), so this synthesis
reports the measured agreement and does not declare the §3.2
standard "passed against bound X". Two caveats attach: the paired
n is small (5 scenes), and all 15 pairs sit at the floor (agreement
on universal failure is weaker evidence than agreement across a
spread of outcomes — the same caveat the part-1 verification
recorded for flagged episodes). **§3.2 does not block certification
on its evidence; §3.4 does.**

## 5. Envelope stability (§3.3)

L2 = the same 5 envelope scenes × 8 non-nominal MuJoCo corners
(pose ±x/±y, friction low/high, mass low/high) × 3 checkpoints =
120 rollouts. Nominal comparator per `analyze.py`: L1 restricted to
the envelope scenes (K=5 pooled) for the verdict, and the nominal
instance for per-rollout flips. (Extension: this synthesis also
checks the nominal-*instance* verdicts directly.)

| Checkpoint × task | Nominal (K=5) | Corner results | Outcome flips vs nominal instance | Verdict stable? |
|---|---|---|---|---|
| base × block (20 nom / 32 corner) | 0/20 FAIL | 0/4 at every corner | 0/32 | Yes — FAIL everywhere |
| base × eye (5 / 8) | 0/5 FAIL | 0/1 at every corner | 0/8 | Yes |
| FL × block | 0/20 FAIL | 0/4 at every corner | 0/32 | Yes |
| FL × eye | 0/5 FAIL | 0/1 at every corner | 0/8 | Yes |
| C-exp × block | 0/20 FAIL | 0/4 at every corner | 0/32 | Yes |
| C-exp × eye | 0/5 FAIL | 0/1 at every corner | 0/8 | Yes |

Totals: **0 successes in 120 corner rollouts; 0 outcome flips in
120 corner-vs-nominal-instance pairs; every corner verdict equals
its nominal verdict (FAIL).** The §3.3 standard — gate verdicts
unchanged across envelope corners on the envelope subset — is met
on this evidence, with the standing caveat that stability at a
universal floor is stability of a FAIL verdict, demonstrated over
5 scenes (block 4, eye 1); it certifies nothing about stability
near the PASS bar.

**[DEFAULT]-at-corner flag (§3.1/§3.3, prep §4):** the friction
low/high corners are [DEFAULT]-at-corner for every task (all
sliding-friction ranges derive from [DEFAULT] nominals,
`envelope-corners-v2.json`). No verdict's stability *depends* on a
friction corner in the discriminating sense — the verdict is FAIL
at nominal, at both friction corners, and at every other corner —
so there is no verdict to flag as friction-contingent; the flag is
recorded here for completeness, and any future verdict that flips
only at a friction corner must carry it. The block_stack mass_low
corner additionally sits on the material-contingent 5 g case of
open item OI-2 (§7.1).

## 6. §7 Bounds Amendment — measured control dispersions (inputs, not frozen bounds)

Per §7, bound *values* are set from control dispersion and frozen
in a dated Bounds Amendment before any non-control checkpoint is
gated. This synthesis supplies the measured dispersions; freezing
the amendment is a separate dated act (open item, §8).

| Quantity (§7) | Measured from this battery |
|---|---|
| Cross-engine policy-outcome agreement | 15/15 paired pairs agree (100%); per task block 12/12, eye 3/3. All pairs at floor — dispersion 0 on a degenerate distribution. |
| Ordering-control noise band | All L1 rates are exactly 0 in every cell; observed inter-checkpoint dispersion is 0 at every denominator. The reference spread (0 → 26 → 70 pp) is ~2 orders of magnitude larger than any band this dispersion could justify — the ordering failure is not a noise-band question. |
| Mechanical/judging disagreement rate | Not measurable from this battery: the §4 judging layer (frontier pass/fail on rollout recordings) was not part of the executed Part B records — no judging fields exist in any `rollouts.jsonl`. Open item (§8). |
| Envelope-corner dispersion | 0 flips / 120 corner pairs; corner success counts identical (0) at all 8 corners × 3 checkpoints × 2 tasks. |

Extensions beyond `analyze.py` in this synthesis, for the record:
trajectory displacement/z-rise analysis (§2.2, §4.2), cross-engine
final-position deltas (§4.2), nominal-instance verdict check (§5),
and the judging-layer absence check (this section). Everything
else follows `analyze.py` exactly (Wilson, solver-exclusion,
pairing, verdict rule).

## 7. Per-task instrument rulings (Stage 1, §3)

Standard-by-standard, then the ruling. §3: a task is certified
only when **all four** standards hold for that task.

### 7.1 §3.1 Parameter audit — status (all four precision tasks)

The v2 parameter audit (`parameter-audit-v2.md`: 40 rows, 17
[MEASURED-EST] / 3 [PROP] / 20 [DEFAULT]; friction and restitution
entirely [DEFAULT]) was verified CERTIFIED-WITH-FINDINGS and its
findings F1–F4 were then delta-verified CERTIFIED
(`verification-v2-parameter-audit.md`). The two record measurements
were attempted and remain **open items** — OI-1 (mat dimensions,
left at 550 × 400 × 3 mm [MEASURED-EST]) and OI-2 (cube material /
mass low corner, 5–35 g [MEASURED-EST]) — which per §3.1 "do not
block certification and cannot change any certification outcome."
**§3.1: satisfied for all four precision tasks** (with OI-1/OI-2
carried open).

### 7.2 block_stack — **NOT CERTIFIED**

| Standard | Result |
|---|---|
| §3.1 parameter audit | Satisfied (§7.1) |
| §3.2 cross-engine | Outcome agreement 12/12 on paired scenes; bound unfrozen (§4.3) — does not block on its evidence |
| §3.3 envelope stability | Met — FAIL stable at all corners, 0 flips (§5); friction corners [DEFAULT]-at-corner, no verdict friction-contingent |
| §3.4 ordering controls | **FAIL** — base-fails-all holds; base < FL < C-exp does not reproduce (0/40 = 0/40 = 0/40); no step-3 excuse documented (§3) |

Validated denominator 8 scenes × K=5 — the strongest denominator
in the battery, and the failure is total on it. Not certified;
research-grade, out of the gate (§3 preamble).

### 7.3 eye_drops_to_basket — **NOT CERTIFIED**

| Standard | Result |
|---|---|
| §3.1 | Satisfied (§7.1) |
| §3.2 | Outcome agreement 3/3 (single paired scene, ep1609 — thin evidence, §4.3 caveat) |
| §3.3 | Met on 1 envelope scene — FAIL stable, 0 flips (§5) |
| §3.4 | **FAIL** — 0/20 = 0/20 = 0/20; carton never contacted in 60/60 L1 rollouts (§2.2, §3) |

Validated denominator 4 scenes. Not certified.

### 7.4 tool_insert — **NOT CERTIFIED — insufficient denominator**

One validated scene (ep683, frozen controller; prep §2) of 8
attempted. §3.4's ordering control on n = 5 rollouts per checkpoint
(Wilson upper bound for 0/5 is 0.435 — the cell cannot even FAIL
decisively against the 60% bar in the §5 sense, and cannot
distinguish checkpoints) is not a meaningful exercise of the
standard; the pre-registration registers no exception that would
let a one-scene denominator certify, and A2.2's denominator rule
admits no rounding up. Independently, the observed result is the
same floor (0/5 = 0/5 = 0/5). No envelope or Drake layer exists for
tool_insert (0 validated envelope scenes), so §3.2/§3.3 are
**untested**, not passed. Ruling: not certified — insufficient
denominator (1 validated scene), with §§3.2–3.4 unmet/untestable
on that denominator.

### 7.5 ring_insert — **NO GATE — NOT CERTIFIED**

Zero validated scenes across 8 attempts under both controller
versions (prep §2, CERTIFIED). Per A2.2(iii): "a task with no
validated scenes has **no gate** and is reported not certified."
No rollouts exist and none were run. The open routes remain
human-in-sim (A2.2 route (c)) or a rotation-constraining executor
revision — neither is Stage 1 evidence today.

### 7.6 cable_clip — **v1.1 Stage 0 certification unchanged; v2 gate exercise fails ordering**

cable_clip is not a v2 Stage 1 candidate: its instrument remains
twin-suite v1.1, certified under demo-replay (pre-reg preamble,
"Kept"), and this battery does not modify it. In the Part B gate
exercise (A1.4 base-fails-all scope), base does fail cable_clip
(0/35), but FL and C-exp also score 0/35, so the §3.4 ordering
does not reproduce for cable_clip either. That is recorded as a
fact about closed-loop gate-protocol behaviour on this task; it
does not rescind the v1.1 scene certification, and no v2 verdict
is issued for cable_clip.

## 8. Open items

1. **Bounds Amendment (§7) not frozen.** Measured dispersions are
   in §6. No non-control checkpoint may be gated until it is
   frozen — and, per §3, not until a task is certified, which none
   is.
2. **Judging layer (§4) not executed in Part B.** No
   mechanical/judging disagreement rate exists; the §7 bound for
   it cannot be set from this battery.
3. **56 of 96 gate scenes never solvability-attempted** (prep §2);
   the validated register covers dev + envelope subsets only
   (20/40 attempted). Extending validation is a separate step.
4. **ring_insert executor problem** (A2.2 routes: human-in-sim or
   executor revision) and **tool_insert grasp feasibility** (7 of
   8 attempted scenes unvalidated) — the denominator problem is
   upstream of the gate.
5. **The floor itself.** Why checkpoints that score 26–70/100 in
   offline replay score 0/300 in closed-loop sim is the central
   open scientific question this battery poses. §3.4 forbids
   excusing the ordering control without a *measured* mechanism;
   candidate mechanisms (closed-loop compounding, grasp-approach
   geometry vs the ~24 mm site-to-pinch offset, contact-model
   sensitivity) each need a registered measurement, not a
   narrative. Until one is measured, the honest state is: the
   instrument cannot yet distinguish checkpoints on the primary
   criterion, and no checkpoint earns arm time through this gate.
6. **OI-1 / OI-2** (parameter audit §6) remain open, non-blocking
   per §3.1.
7. **Drake instrument differences carried, not closed:** finger-part
   mesh exclusion (all L3 records), joint-damping gap, and
   trajectory-level (not outcome-level) cross-engine divergence up
   to 0.120 m on interacting scenes (§4.2).

## 9. Recomputation guide (for the verifier)

- L1/L2 tables: group `out/{L1,L2}/<ckpt>/rollouts.jsonl` by
  task (and corner for L2); exclude `solver_failure` records from
  denominators (there are none); Wilson per `analyze.py:wilson`
  (z = 1.96); verdicts per `analyze.py:verdict` = pre-reg §5.
  Running the battery's own `analyze.py` reproduces §§2/4.1/5's
  counts (its `analysis.json` schema) — this synthesis's tables
  were computed independently and agree with that code's rules.
- Ordering: §3's table is §2.1's rates against A1.1's 0/26/70.
- Drake pairing: L3 record ↔ L1 record with same task/episode,
  `variant == "0"`, `corner == "nominal"` (`analyze.py` rule);
  validity per `analyze.py:drake_valid`.
- Trajectory numbers (§2.2, §4.2): for each `traj_*.npz`, per
  `obj_*` array, net displacement ‖a[-1] − a[0]‖, max excursion
  max ‖a − a[0]‖, z-rise max(a[:,2]) − a[0,2]. No rollout was
  re-run for this synthesis; the only re-derivations are these
  reads of the recorded trajectory files.
- Checkpoints are the prep §3 manifest pins (base
  `5e8d12a6…`, FL weights SHA `5b1cbf29…`, C-exp `8f6807ed…`),
  CERTIFIED in `verification-v2-controls-prep.md` §4.

**Bottom line, per task:** block_stack NOT CERTIFIED (§3.4 FAIL) ·
eye_drops_to_basket NOT CERTIFIED (§3.4 FAIL) · tool_insert NOT
CERTIFIED (insufficient denominator; §§3.2–3.4 unmet/untestable) ·
ring_insert NO GATE / NOT CERTIFIED (A2.2(iii)) · cable_clip v1.1
certification unchanged, no v2 verdict. **Stage 1 certifies no
precision task. No gate run on any non-control checkpoint is
authorized.** All verdicts above are funnel verdicts — sim bar
unanchored to real-arm delta (no calibration pairs yet) (§8).
