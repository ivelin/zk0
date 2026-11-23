# Sim-Gate Design Memo — Making Sim Validation a Trustworthy Judged Pass/Fail Gate

**Date:** 2026-11-22 · **Founder directive:** 2026-11-22 21:25 CT (make sim
validation work as the judged task pass/fail gate until physical arm evals
exist; no physical-eval time until the sim/twin pipeline is stabilized).
**Primary records:** twin-suite v1.1 completion report (2026-11-22);
BENCH-003 design (2026-11-19). Numbers are quoted from those two
records except where another source is named in place: the twin-replay
v1 completion report (Phase 6 cost model; parameter-sourcing taxonomy),
the Study 001 final eval table (judged-completion numbers used as
gate controls; independently verified 2026-11-22), and the labeled
[ESTIMATE] figures in the cost section.

## Thesis

The twin program did not fail because the twins are bad; it failed because
we certified the wrong thing. Demo-replay certification asks a scene to
reproduce a recorded outcome under open-loop execution, which requires
initial poses accurate to the task tolerance — and v1.1 proved, on
held-out data, that the recordings contain poses only to σ ≈ 2.5–3.5 cm
against 1–2 cm tolerances (ring: ≲3 mm physical clearance). That
information does not exist in the recordings, so no amount of twin
engineering recovers it. **A pass/fail gate does not need it.** A gate
needs verdicts that are *valid under stated uncertainty*: the checkpoint
ranking and the pass/fail call must be stable across everything we
honestly don't know (pose within its measured σ, friction/mass within
their sourced ranges, engine choice within agreement bounds), and the
instrument must reproduce orderings we already know from independent
evidence. Flip the certification target — certify the *evaluation
instrument* (twin + scoring + uncertainty envelope), run policies
closed-loop inside it, gate on distributional separation — and the gate
is achievable with assets that already exist. Demo-replay certification
survives exactly where v1.1 proved it works: pose-tolerant outcome
classes, today cable_clip.

## Where we stand (from the v1.1 record, not re-derived)

- cable_clip: **CERTIFIED** under open-loop demo replay — 20/20 episodes
  reproduced, mean variant agreement 1.00, solver failures 0/100 after
  the global timestep fix; published (HF `zk0bot/twin-suite`, pinned
  revision `579ad96b…`). Its outcome class is pose-tolerant and its
  recorded outcomes are failed executions, reproduced physically.
- The four precision tasks: **NOT CERTIFIED**, structurally. Held-out
  pose σ per axis (image pipeline): block ≈2.5 cm, ring ≈2.5 cm,
  tool ≈3.5 cm, eye_drops ≈3.5 cm; pooled dwell corroboration ≈2.6
  cm/axis. Placement error compounds to ≈3.5–5 cm against block's
  ≈1.1 cm inset tolerance and ring's ≲3 mm peg clearance. Eye_drops'
  secondary criterion is 0/20 — even when the outcome class reproduces,
  the recorded *placement* does not (nominal finals off by 7–12 cm).
  Tool_insert's median nominal secondary error is 18.4 cm.
- The FK-anchored upgrade — the one path that could have closed the
  gap — was implemented and measured on held-out data: worse than the
  image pipeline on every task (best variants: block 4.9/7.2, ring
  3.1/3.2, tool 3.0/2.5, eye 4.6/4.6 cm). Even the oracle-dwell floor
  is 2.6–4.4 cm; FK↔image frame agreement is ~3.3 cm median. The
  σ ≲ 1 cm target is unreachable from the recordings, with evidence.
- What v1.1 *did* build and verify, and this memo reuses: mechanical
  contact/support scoring via `contacts_with()` for all five tasks
  (positions are diagnostics only); a secondary criterion thresholded
  at 3.0 cm (ring 2.0 cm unengaged; cable exempt); K=5 scene variants
  per episode with byte-identical regeneration; in-harness
  solver-failure detection (suite-wide solver failures 0/480);
  per-parameter sourcing labels (v1 parameter table:
  [MEASURED-EST] 14 / [PROP] 3 / [DEFAULT] 13, with friction and
  restitution entirely [DEFAULT]-labeled — twin-replay v1 completion
  report, audited honest);
  frame rate resolved from dataset metadata (20 fps); frozen suite at
  param_hash `a4eb3f8c3611894d`, denominator 96 (Amendment 3).
- Disclosed limitation carried forward: in v1/v1.1 the validation set
  doubled as the tuning set for global iterations. The 96-episode set
  is therefore *twin-side tuning-exposed*. It is not policy-side
  contaminated (no policy trained or selected on it), but any v2 gate
  design must treat scene-fidelity claims from that set as upper
  bounds and must not tune twins against gate outcomes after freeze.

## The question the gate actually has to answer

The BENCH-003 funnel gives the gate a specific job: **narrow the
candidate set before the expensive stage.** Stage 4 spends arm time
only on (1) models that cleared the sim bar (≥60% judged completion on
held-out families, founder-signed) and (2) discriminating cases. The
gate's output is a *promotion decision*, not a measurement of reality.
That distinction decides everything below:

- A promotion decision needs **ranking validity** (better checkpoints
  pass, worse ones don't) and **error control in the expensive
  direction** (a false pass wastes arm time; a false fail costs a
  re-run). It does not need per-episode reproduction of human demos,
  and it does not need absolute rates that transfer to the real arm —
  the design already reserves those claims for Stage 4 calibration
  pairs (success delta, contact agreement, failure-mode overlap),
  which do not exist yet and cannot exist until the first arm session.
- Therefore: **demo reproduction is sufficient evidence for a gate
  where achievable (cable), but it is not necessary evidence anywhere.**
  The necessary evidence is (i) verdict stability inside the honest
  uncertainty envelope, (ii) agreement across independent physics
  engines, (iii) reproduction of orderings known from independent
  evidence, and (iv) failure modes that are physical, not numerical
  (v1.1's solver-failure standard).

## Option analysis

### A. Closed-loop (servoing) replay twins

**Mechanism.** A tracking controller (PD/servo on joint trajectory,
possibly with visual servoing) executes the recorded trajectory in the
twin, correcting deviations as it goes.

**What it actually measures.** The (twin + controller) pair — as the
v1.1 report itself states, a different replay instrument "would grade
the twin + a controller, not the recorded trajectory." Worse, feedback
destroys the very signal scene certification was testing: a controller
with enough authority corrects a 3 cm initial-pose error on the way
in, so a "reproduction" no longer evidences that the scene's poses or
dynamics were right — it evidences that the controller can absorb
them. A weak controller produces the opposite failure: good scenes
fail. Certification under closed loop is thus a claim about a frozen
controller's authority envelope, and the controller becomes a hidden
tuning knob inside the instrument (gains interact with contact
dynamics; a sim servo is not an STS3215 under LeRobot command limits).

**Verdict.** Inadmissible as the scene-certification standard. Two
legitimate uses remain: (1) a *diagnostic* — if closed-loop replay
reproduces an outcome that open-loop cannot, scene dynamics are
plausible and the residual is pose/trajectory, which is useful
triage information; (2) it is conceptually what Option B does anyway —
a policy *is* a closed-loop controller — so the honest version of
"closed-loop replay" is to replay with the artifact we actually care
about (the checkpoint), not with a reference controller standing in
for it. **Cost:** moderate build, permanent argument cost — every
result carries a controller caveat. Not recommended as the gate.

### B. Checkpoint-native evaluation with instrument certification (RECOMMENDED core)

**Mechanism.** Skip demo-replay certification for the precision
tasks. Run the *policy* closed-loop in the twin, starting from
initial states sampled from the recorded initial-state *distribution*
(pose estimate ± its measured per-task σ; K variants per episode,
machinery already built and determinism-verified). Score by the
mechanical contact/support criteria v1.1 implemented and the
verifiers closed on. Certify the *instrument* instead of the scene,
against four standards:

1. **Parameter audit** (already the v1 standard): every physics
   parameter sourced and labeled; the two cheap physical measurements
   already identified (mat dimensions, cube material) taken, since
   they are cheap — while noting they move calibration marginally,
   not certification.
2. **Cross-engine agreement:** the same scenes and policies run in a
   second engine (the v1.1 suite's engine is MuJoCo, per its XML
   scene format; the BENCH-003 design names Drake the eval gate;
   Genesis is the generation front door and stays off the grasp
   path until it earns its way back, per the design).
   Policy outcome agreement across engines must fall inside a
   registered bound; disagreement localizes a twin suspect, not a
   policy suspect (the design's Stage 3 language).
3. **Envelope stability (sensitivity):** sweep the plausible envelope
   — pose within measured σ, friction/mass within sourced ranges —
   and require that the gate verdict (pass/fail per checkpoint per
   task) is *unchanged* across envelope corners. A verdict that flips
   inside its own uncertainty envelope is reported **INDETERMINATE**,
   never rounded to pass or fail.
4. **Known-ordering controls:** the gate must reproduce orderings
   already established on independent evidence before it may judge
   anything new: the untrained base (0/100 judged completion, sealed
   base-floor) must fail every task; the Study 001 offline-replay
   ordering (base < FL round-30 at 26/100 < C-exp at 70/100 judged
   completion — Study 001 final eval table, independently verified)
   must hold in the gate within registered noise. Stated assumption:
   this control crosses modalities — the reference ordering comes
   from open-loop offline replay, the gate measures closed-loop sim
   rollouts — so a mismatch is *evidence against the instrument*
   only after modality effects are excluded (e.g. a checkpoint whose
   closed-loop behavior genuinely differs from its open-loop replay
   signature); the pre-registration must name how that exclusion is
   argued, or the control downgrades to the base-fails-all check
   plus cross-engine agreement. If the instrument cannot reproduce
   what we already know, the instrument is wrong — that is an
   instrument failure, not a model result.

**What it measures.** Policy competence under the twin's physics,
integrated over the honest uncertainty distribution. That is exactly
the quantity a funnel gate consumes.

**Failure modes.** (1) *Correlated twin error:* both engines could
share a wrong contact model and agree on a wrong verdict — mitigated,
not eliminated, by the controls and by Stage 4 calibration later;
this is why the gate's claims (below) are ranking claims, not rate
claims. (2) *Envelope dishonesty:* friction/restitution in v1 were
entirely [DEFAULT]-labeled; an envelope built on defaults is only as
honest as its labels — the pre-registration must state ranges per
parameter with their labels attached, and any verdict whose stability
depends on a [DEFAULT] parameter at its corner is flagged as such.
(3) *Policy exploiting sim:* checkpoints optimized against the gate
can learn simulator exploits (the field's known failure — see §F);
mitigated by the frozen suite, the second engine, and by never
training against gate episodes. (4) *Visual gap:* the policy consumes
sim renders from the three matched viewpoints; render realism is a
separate fidelity axis from physics and is covered by the two-layer
judging (frontier judges see the rollout recordings and can flag
visually absurd successes) and by the controls.

**Cost.** The expensive piece is the cross-engine port; the rest
reuses v1.1 machinery. See cost section.

### C. Distribution-level criteria (RECOMMENDED as the gate's semantics)

**Mechanism.** An episode's outcome is a distribution over its K
variants, not a bit. A checkpoint's task score is the pooled success
distribution over episodes × variants, reported with Wilson
intervals. The gate compares distributions: PASS requires the
*lower* confidence bound to clear the bar and the verdict to be
separated from the fail region by a registered margin — not a point
estimate crossing a line.

**Why it fits.** It matches the actual uncertainty structure v1.1
measured (per-episode reproduction was already defined as ≥60% of
valid variants agreeing; the variant machinery, seeds, and
determinism are built). It kills false precision: a checkpoint at
61% ± 12 does not "pass" a 60% bar.

**Failure modes.** Distributional separation certifies *separation*,
not absolute truth — both distributions can be biased together
(twin bias shifts all checkpoints). For a funnel gate that is
acceptable (ranking + a bar anchored by controls); for a real-world
rate claim it would be fatal, and the claims list below keeps those
separate. Variant distributions are only as honest as the sampling
distribution — per-task σ is measured, friction/mass ranges are
partly [DEFAULT]; see Option B failure mode 2.

**Cost.** Near-zero incremental: semantics and statistics over
machinery that exists.

### D. Tolerance-robust task variants as gate tasks

**Mechanism.** Re-scope the precision tasks with larger clearances
(bigger ring bore, wider inset) so pose σ no longer binds; gate on
the loosened tasks; keep the original tasks as stretch measures.

**What it does to the claim.** It changes the claim. A pass on a
loosened ring_insert certifies competence at the loosened tolerance,
which is *evidence about* — not evidence *of* — competence at
≲3 mm clearance. The Study 001 tasks are the claim's substance;
quietly substituting easier tasks is exactly the claim-drift failure
this program's discipline exists to prevent. There is also a
measurement subtlety: loosening changes which policy behaviors are
rewarded (approach precision stops mattering), so rankings on
loosened tasks need not preserve rankings on the real ones.

**Verdict.** Admissible only as an *additional, explicitly labeled
screen* (a difficulty curriculum: fail the easy version ⇒ fail,
cheaply; pass ⇒ proceed to the real gate), never as the gate's read
on the study tasks. Optional; not in the recommended critical path.

### E. Richer pose evidence

**State plainly: this path is dead for certification purposes.**
v1.1 measured the floor three ways and they agree: oracle-dwell
selection (choosing the best dwell evidence with hindsight) still
floors at 2.6–4.4 cm; FK↔image frame agreement is ~3.3 cm median —
a *frame-consistency* bound that no estimator fitted to these
recordings can beat, because the two measurement channels disagree
with each other at that scale; multi-frame aggregation is already
what the dwell pipeline is. CAD fitting and the wrist camera are
inputs to the same two channels and inherit the same floor. The
remaining cheap physical measurements (mat dimensions, cube
material) improve calibration marginally and are worth taking for
the parameter audit, but they do not move pose σ and cannot change
any certification outcome. **Recommendation: close this path on the
record; spend nothing further on it beyond the two measurements.**

### F. What the field does (brief, sourced)

In the published approaches surveyed here, manipulation eval scenes
are not certified by open-loop demo reproduction at millimetre
tolerances; the standard is *predictivity evidence plus controls*:

- **SIMPLER** (Li et al., CoRL 2024) evaluates real-trained policies
  in purpose-built simulated environments and validates the
  *instrument* by correlation with real outcomes over ~1,500 paired
  sim/real episodes — explicitly *without* crafting full-fidelity
  digital twins. Key gaps addressed were control and visual
  disparity, mitigated by offline system identification and
  observation alignment; ranking fidelity is quantified (Pearson /
  MMRV-style metrics), not assumed. This is Option B's shape, with
  real pairs we don't have yet.
  (arXiv:2405.05941; https://simpler-env.github.io/)
- **Sim2Real Predictivity** (Kadian et al., RA-L 2020; visual
  navigation, Habitat/PointGoal — cited for the predictivity
  principle, not manipulation evidence) makes the
  general point a metric: the Sim-vs-Real Correlation Coefficient
  (SRCC) of a simulator is a property to *measure and tune for* —
  theirs rose from 0.18 to 0.844 after fixing simulator
  imperfections agents were exploiting (collision sliding). Our
  local instances of the same lesson: solver explosions counted as
  agreement and phantom-clip scoring in v1 — instrument defects
  that masqueraded as results until independent verification caught
  them. (arXiv:1912.06321)
- **PVR sim-predictivity study** (Silwal et al., 2023) found sim/real
  trend correlation of 0.929 *after basic alignment* (camera
  placement, checkpoint-selection scheme) — alignment of the eval
  contract mattered more than raw fidelity. (arXiv:2310.02219)
- **Domain randomization** (Tobin et al., 2017) is the standard
  mechanism for making "robust across the plausible envelope" the
  measured quantity — Option C's sampling is domain randomization
  restricted to *measured/sourced* ranges, per the BENCH-003
  design's rule (randomize around measured parameters, never
  invented ones).
- **Caution from our own design record:** RoboColosseum's first
  study found policy rankings barely transfer across embodiments
  (Kendall W = 0.24) — gate validity is specific to embodiment and
  setup, and the gate's claims must stay scoped to ours.

## Recommended architecture — the sim gate, v2

Four stages. Stages 0–1 certify instruments; Stage 2 judges
checkpoints; Stage 3 is the escalation path the BENCH-003 design
already defines.

**Stage 0 — Demo-replay-certified tasks (exists).** cable_clip under
twin-suite v1.1: scenes certified by open-loop demo replay (20/20,
agreement 1.00). Checkpoints are evaluated closed-loop in these
twins (the design's Phase 6) with dual scoring: mechanical
object-state criteria primary, frontier-model judging of rollout
recordings as the recorded second layer, disagreements surfaced per
episode. This stage is the template and the proof that the
certification pattern works where its evidence standard is
achievable.

**Stage 1 — Instrument certification for the precision tasks
(twin-suite v2).** For block_stack, eye_drops_to_basket, ring_insert,
tool_insert: scenes are *not* certified by demo replay. The
instrument (scenes + harness + envelope) is certified when all four
hold, verified per artifact class by independent contexts:
(1) parameter audit passes with sourcing labels, including the two
cheap physical measurements; (2) cross-engine (MuJoCo vs Drake)
policy-outcome agreement inside the registered bound on the control
checkpoints; (3) envelope-stability machinery demonstrated — verdicts
on controls unchanged across envelope corners; (4) known-ordering
controls reproduce (base fails all tasks; base < FL < C-exp ordering
holds within registered noise). Certification is per-task: a task
whose instrument fails any standard stays research-grade and out of
the gate, as today.

**Stage 2 — The gate run (per checkpoint).** Policy runs closed-loop
in the certified instruments, K=5 variants per episode over the
frozen held-out episode set, initial states sampled from the
per-task pose distribution (σ: block/ring 2.5 cm, tool/eye 3.5 cm —
v1.1 held-out measurements), mechanical criteria primary, frontier
judging of recordings as the recorded second layer. **Pass/fail
semantics, three-valued:**
- **PASS** (per task): Wilson lower bound of pooled success ≥ 60%
  (the founder-signed sim bar, applied to the lower bound, not the
  point estimate) **and** verdict stable across envelope corners
  **and** no disqualifying sim-only failure class (solver artifacts
  are excluded by the harness standard; physically absurd successes
  flagged by the judging layer route to Stage 3, not to PASS).
- **FAIL**: upper bound below the bar, stable across the envelope.
- **INDETERMINATE**: the verdict flips inside the envelope, or
  cross-engine disagreement on this checkpoint exceeds the bound,
  or judging-layer/mechanical disagreement exceeds its registered
  rate. INDETERMINATE routes to Stage 3 escalation or more variants
  — never silently to the arm, and never rounded.
- Checkpoint promotion requires PASS on every task (per-task-first
  reporting per the design §10.3: a pooled average may not hide a
  dead task), or a founder-registered waiver naming the exception.

**Stage 3 — Escalation and recalibration (exists in design).**
INDETERMINATE and judge-disputed episodes go to twin replay under
harsher physics (design Stage 3). When Stage 4 arm sessions
eventually run, their calibration pairs produce the design's
fidelity metrics (success delta, contact agreement, failure-mode
overlap) and the gate's sim bar is re-anchored to a *measured*
sim-vs-real delta instead of an argued one. Until then, every gate
output carries the label: **funnel verdict — sim bar unanchored to
real-arm delta (no calibration pairs yet).**

**Monthly realism re-audit** (founder refinement, 2026-11-22)
applies to v2 unchanged: archived tie points re-derived, nominal
variants re-run, outcome flips investigated, unrealistic scenes
fixed in a new version or retired, never edited in place.

## Pre-registration skeleton (twin-suite v2 / sim-gate v1)

To be frozen before any gate run; amendments dated and pre-result.

1. **Suite identity:** twin-suite v2 = commit of code + parameter
   table + scene JSONs + seed; param_hash coverage per v1.1 standard
   (all pipeline code, parameter doc, calibration data, pose-file
   manifest, fixtures). External inputs pinned by SHA-256 reference.
2. **Episode set:** the frozen held-out set (denominator 96 per
   Amendment 3; exclusions named). Disclosed: twin-side
   tuning-exposed; no twin parameter may be tuned against gate
   outcomes after this freeze. Development/tuning of the instrument
   uses control checkpoints and a named development subset only.
3. **Instruments per task:** Stage 0 (cable, demo-replay certified,
   twin-suite v1.1 pinned revision `579ad96b…`) / Stage 1 (four
   precision tasks, instrument-certified per the four standards).
4. **Variant protocol:** K=5 variants per episode, seed frozen;
   initial-state sampling: pose estimate ± per-task σ (block 2.5,
   ring 2.5, tool 3.5, eye 3.5 cm/axis); friction/mass sampled
   within the parameter table's sourced ranges; every parameter's
   label ([MEASURED-EST]/[PROP]/[DEFAULT]) printed in the run
   manifest.
5. **Scoring:** mechanical contact/support criteria as implemented
   in v1.1 (positions diagnostic only); solver-failure detection
   on, exclusions counted and reported; secondary criterion
   reported, not gated (thresholds per v1.1).
6. **Second layer:** frontier-model pass/fail judging of rollout
   recordings under the fixed rubric, checkpoint identity hidden;
   mechanical/judging disagreements surfaced per episode; a
   disagreement rate above the registered bound ⇒ INDETERMINATE.
7. **Controls (run first; gate is void if controls fail):**
   base checkpoint must FAIL all five tasks; ordering base <
   FL-r30 < C-exp (Study 001 judged-completion reference) must hold
   within the registered noise band, subject to the cross-modality
   assumption stated in Option B; cross-engine agreement on
   controls inside the registered bound. Every "registered bound"
   in this skeleton (cross-engine agreement, noise band,
   disagreement rate) gets its value set during Stage 1 from the
   control checkpoints' measured dispersion across engines and
   envelope corners, and is frozen in this pre-registration before
   any non-control checkpoint is gated — bounds are measured from
   controls, never fitted to candidates.
8. **Bars:** per-task PASS = Wilson lower bound ≥ 60% and
   envelope-stable; FAIL = Wilson upper bound < 60%, envelope-
   stable; otherwise INDETERMINATE. Promotion = PASS on all tasks
   or a named founder waiver. Envelope corners enumerated in the
   manifest (pose at ±σ extremes, friction/mass at range corners,
   engine swap).
9. **Blinding:** checkpoint identity hidden from frontier judges
   and from any human judging sample; mechanical scoring is
   identity-blind by construction; the founder's blinded spot-check
   set is drawn at random from judged rollouts, checkpoint hidden
   (Phase 7 pattern).
10. **Independent verification:** per-class verifiers (objects/
    assets, environments/poses, code, end-to-end) in separate
    contexts, raw artifacts only; end-to-end verifier re-runs a
    stratified sample (v1.1 precedent: 160 runs, exact-match
    standard) plus all controls; certification verdicts are the
    verifiers', not the builders'.
11. **Reporting:** per-task first, pooled second; every number with
    its Wilson interval, variant count, solver-exclusion count,
    envelope-stability result, and the unanchored-bar label until
    Stage 4 calibration pairs exist.
12. **Change control:** any change to scenes, criteria, envelope,
    or bars after freeze is a new suite version (minor/major per
    the v1.1 versioning rules) with full re-validation of the
    affected tasks and fresh verification.

## Cost to stabilize [ESTIMATE — basis stated]

Anchors: the v1.1 fix pass (six items, five verifier classes) is
the scale anchor for agent-work size; the Phase 6 cost model in the
twin-replay v1 completion report prices closed-loop policy eval at
≈2–5 GPU-hours per checkpoint per certified task (at 20 episodes ×
3 variants; the gate below uses 5 variants, so treat the model as a
floor) and ≈26–65 GPU-hours for a full 13-checkpoint matrix on one
task.

- **Instrument build (one-time):** cross-engine (Drake) port of the
  four precision-task scenes + agreement harness — the dominant
  item and the widest uncertainty; envelope/stability harness and
  gate runner reuse v1.1 variant machinery; controls battery runs
  on existing checkpoints (base, FL-r30, C-exp — no training).
  Estimate **30–50 agent-hours** and **15–30 DGX-hours** (MuJoCo and
  Drake rollouts are CPU-bound; v1.1 validation ran CPU-only on the
  DGX without disturbing training). The Drake port is the risk: if
  scene semantics diverge badly, agreement work could double that
  line — the pre-registration should time-box it and report.
- **Per-checkpoint gate run:** 96 episodes × 5 variants = 480
  closed-loop rollouts + envelope-corner re-runs on a stratified
  subset + judging layer. From the Phase 6 model (adjusted up for
  5 variants vs its 3), **≈10–25 DGX-hours per checkpoint** for the
  full five-task gate;
  a cable+one-task canary configuration is a fraction of that.
- **Verification:** per-class verification of v2 plus end-to-end
  re-runs — comparable to v1.1's verification load, **≈10–15
  agent-hours**, no DGX training contention (CPU sim).
- **Total to a stabilized, certified gate:** roughly **45–65
  agent-hours + 25–55 DGX-hours**, dominated by the cross-engine
  port. No physical-arm time. No new training runs.

## Claims the gate would and would NOT support

**Would support** (once Stage 1 certification lands, per task):
- "Checkpoint X **passes/fails the sim gate** on task T" — a funnel
  verdict under the registered envelope, three-valued, with
  INDETERMINATE a first-class outcome.
- **Relative ranking** of checkpoints on the five study tasks under
  the twin's physics and the honest uncertainty distribution,
  stable across the envelope and across two engines.
- "Checkpoints below the gate bar do not earn arm time on the
  bar-clearing track" — the funnel claim, backed by controls and
  ordering reproduction. (The design's second arm-time class —
  discriminating cases, where models or judges disagree — is a
  separate track and is unaffected by a checkpoint's bar status.)
- Failure-taxonomy claims in sim (grasp miss vs place miss vs
  contact violation), routed per the design's taxonomy.
- For cable_clip additionally: demo-replay-certified scene claims,
  as already published for twin-suite v1.1.

**Would NOT support** (until Stage 4 calibration pairs exist, and
some not even then):
- Any **absolute real-world success rate** ("the model succeeds
  60% on the real arm"). Sim rates are not real rates; the
  success-delta metric is unmeasured until calibration pairs
  exist, and every gate output carries that label.
- **Per-episode reproduction** claims for the four precision tasks
  ("the twin reproduces episode N") — proven impossible from the
  recordings; v2 does not resurrect them.
- **Novel-task generalization** claims — the gate runs the five
  study tasks; novel tasks remain the Stage 1 screen's territory
  with their own evidence.
- **Contact-force fidelity** claims beyond the parameter audit —
  no sim-vs-real force agreement data exists yet (design Stage 3
  metric, unmeasured).
- Claims about **other embodiments, setups, or task families** —
  gate validity is scoped to this arm, this setup, these tasks
  (the W = 0.24 lesson).
- Any claim that a PASS **guarantees** the real-arm bar (≥50% over
  ≥10 trials/task). PASS earns the arm session; it does not
  pre-judge it.

## The founder decision

One decision gates all of this: **approve the certification-target
flip** — pre-register twin-suite v2 as an *instrument-certified,
checkpoint-native, distribution-gated* standard (Options B + C),
retiring demo-replay certification for the four precision tasks
(it stays for cable and any future pose-tolerant outcome class),
and closing the richer-pose-evidence path (Option E) on the record.
Everything else — the pre-registration, the build, the controls —
follows mechanically from that choice. The alternative is to keep
spending on a certification standard v1.1 proved unreachable from
the evidence the recordings contain.

---

*Verification: independently verified in a separate context against
the v1.1 completion report and the BENCH-003 design — verdict
CERTIFY-WITH-FIXES (`sim-gate-memo-verification-20261122.md`). All
v1.1 and BENCH-003 numbers and all three literature claims checked
clean. Fixes applied per the verifier: sourcing scope corrected in
the header (Study 001 control numbers, the Phase 6 cost model, and
the sourcing-label taxonomy now cite their actual records — the
Study 001 final eval table and the twin-replay v1 completion
report, both located and quoted in place); the arm-time claim
reworded to the design's two-class rule; the cross-modality
assumption behind the ordering control stated; §F's scope softened
to the surveyed literature; Kadian scoped to navigation; rollout
count corrected to 480 and the cost model to 13 checkpoints at
3 variants (gate's 5-variant uplift noted); bound-setting procedure
added to the pre-registration skeleton.*
