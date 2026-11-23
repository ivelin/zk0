# Study 001 Twin-Replay — Completion Report (Phases 1–5), 2026-11-22

Coordinator synthesis. Builders: A (frames/inventory/parameters),
B (poses), C (twin build/validation). Verification: four independent
contexts (V-Objects+Assets, V-Environments+Poses, V-Code, V-E2E), raw
artifacts only. Governing docs: twin-replay-preregistration-20261122.md,
twin-replay-amendment1-20261122.md, twin-replay-amendment2-20261122.md.

## Bottom line

**twin-suite v1 is NOT CERTIFIED for any task.** The twins were built for
all 100 held-out episodes (97 in the validation denominator), the
demo-replay machinery is deterministic and its numbers independently
reproduce exactly — and the recorded outcomes reproduce at rates far below
the pre-registered 80% bar for four tasks, with the fifth (cable_clip)
meeting its outcome bar only on a corrected variant set it has not yet
been re-issued with. Per Amendment 2, the certified set is therefore
EMPTY: nothing is published to Hugging Face as a certified suite this
round. The full record (code, scenes, docs, validation, verification) is
committed to the study branch as a non-certified research record.

The core scientific finding is structural, not a single bug: pose
estimation from the recordings carries σ ≈ 2.5 cm per pose; a placement
inherits ≈ √2 × that (≈3.5–5 cm), while the tasks' success tolerances are
1–2 cm (ring engagement ≲3 mm physical clearance; block stack inset
≈1.1 cm). The demos were executed closed-loop (visually corrected); the
pre-registered replay is open-loop. Open-loop replay at this pose accuracy
cannot reproduce precision placements, and no global parameter change
within the sourced ranges closes that gap (agreement plateaued across 3
full-set runs).

## Per-task validation results (97 episodes; K=5 variants, seed 20261122)

Episode REPRODUCED = ≥60% of valid variants agree with the recorded
outcome (Amendment 1). Recorded outcomes (split.json, coordinator-verified):
success ×20 for block_stack / eye_drops / ring_insert / tool_insert;
failure ×20 for cable_clip.

| Task | In | Reproduced | Mean variant agreement | Bar (≥80%) |
|---|---|---|---|---|
| block_stack | 20 | 0/20 | 0.00 | NOT MET |
| eye_drops_to_basket | 20 | 4/20 (ep1757, 1778, 1810, 1811) | 0.28 | NOT MET |
| ring_insert | 20 | 1/20 (ep253, partly spawn coincidence — V-E2E) | 0.06 | NOT MET |
| tool_insert | 17 | 0/17 | 0.00 | NOT MET |
| cable_clip | 20 | 20/20 outcome class on valid variants | 1.00 (class) | MET on corrected set only — see below |

Cable_clip detail: 10/100 variants (6 of 20 nominals) numerically explode
(chain ejected metres from the workspace). The builders counted them as
valid/agreeing — contrary to Amendment 1 (solver failure = exclusion
ground). V-E2E re-ran and confirmed the explosions, and confirmed the
non-exploded variants genuinely reproduce the recorded failures (cable
worked toward the block, never seated; nearest segment median 9.7 cm).
Excluding exploded variants, every cable episode still agrees on 100% of
its remaining valid variants (worst case ep2068, 2 valid). So cable's
episode-level agreement survives correction, but the certified variant
set must be re-issued at ≤90 variants, and the systematic explosion mode
is itself an unexplained failure mode under the pre-registration's
certification clause. Ruling: NOT CERTIFIED as issued; the only task with
a short, defined path to certification (see v1.1 list).

Secondary criterion (final planar position ≤3.0 cm, ring 2.0 cm): computed
but — V-Code finding F-02 — never thresholded in the harness. The 4
reproduced eye_drops episodes violate it on nominal (7.3–12.4 cm): they
reproduce the outcome class (box ends in the basket) without reproducing
the recorded placement. Reported here so the record is complete.

## Episodes in / out

- OUT (pre-registered, low initial-pose confidence): tool_insert ep630,
  ep871, ep876 — tool occluded by the arm in the only views. Named, never
  guessed. Denominator 97.
- IN with secondary skipped: eye_drops ep1618, ep1760 (final pose
  unobservable; initials medium).
- Variant-level exclusions: 12/485 (logged build defects). V-Env nuance:
  the 8 tool exclusions fall exactly on the fabricated-pose episodes
  (symptoms of the pose defect); the 4 eye exclusions are a systematic
  platform-edge spawn effect (identical 9.0 cm pre-settle drop).
- POSE FABRICATIONS discovered in verification: 7 tool_insert initial
  poses (ep697, 779, 784, 885, 888, 892, 903) are wrong by 12–35 cm —
  the segmentation locked onto fixed chrome (tray bars / arm hardware),
  producing near-identical detection pixels across episodes. All labeled
  "medium" (σ 3.5 cm). These episodes' 0% agreement measures the pose
  defect, not twin physics. Control episodes with grounding picks verify
  at 1.6–2.3 cm.

## What the verification established (per-class verdicts)

- **V-Objects+Assets — CERTIFIED-WITH-FINDINGS.** Dimensions in the built
  scenes match the parameter table exactly; my verifier's independent
  frame re-estimates land inside the stated uncertainties; 8 parameter
  sources audited in depth — honest. Settled: ring board recess count = 6
  (Builder A's docs say 5 — wrong; B/C correct). Findings: scene XMLs
  ignore colour variants (every second cube renders green, every ring
  yellow — 11/20 block and ≥3–4/20 ring episodes would render the wrong
  colour, which matters for any future visual judging); the tool tray's
  target slot recess is not modelled (tray is a solid box); ring-mass
  derivation arithmetic wrong (value itself plausible); the asset
  inventory's "SIM-1/SIM-2 era validation" label is uncorroborated (the
  underlying testing is real — Nov 15 demo + audit logs).
- **V-Environments+Poses — NOT-CERTIFIED.** Tool-pose allegation confirmed
  and understated (see above). Block pose pipeline internally inconsistent:
  27/62 block entries cannot be re-derived from the preserved detection
  pixels (>2 cm, max 13.1 cm) — B's auditability claim fails for 44% of
  block entries. Calibration: B's ±2.5 cm is a budget, not a measurement;
  B's frame is NOT the stated FK world frame (the correcting affine has a
  reflection; the two maps disagree by median 29.4 cm over the mat). The
  affine is a single global map (code-verified by two verifiers), but the
  claimed tie-point improvement (12.8 → 1.4 cm) is UNVERIFIABLE AS
  ARCHIVED — the 93 dwell pairs were never saved (fk_tiepoints.json holds
  only home poses). Variants otherwise pass: K=5 everywhere, jitter
  statistics match N(0,σ), mass/friction draws inside sourced ranges,
  seed regeneration exact. Camera/layout claims confirmed.
- **V-Code — NOT-CERTIFIED.** Blocking: the pre-registered contact/support
  criteria were NEVER implemented — block/eye/cable outcomes were scored
  by position proxies (a contacts_with() helper exists, never called; the
  clip's jaws are static geometry that never closes). Secondary criterion
  never thresholded (F-02). Replays ran at 20 fps against the 30 fps
  pre-registration, unresolved (F-05). The grasp device departs from the
  registered device in five ways (F-06; the departures were disclosed by
  the builder as interpretations, and A's literal spec indeed never fires
  for top grasps — but they are departures, and one doc still describes
  the non-inverted gripper formula). Changelog timestamps are
  irreconcilable with run/code mtimes (retrospective write-up; numbers
  and sequence cohere). param_hash excludes the files where changes live
  (replay.py/build_scenes.py). Peg+board — the outcome reference — was
  jittered per variant from run 3 (not in Amendment 1's design). Pose
  jitter is unbounded Gaussian, not clipped "within" σ. Certified
  sub-items: 1.0 s settle exact; ≥60%/≥80% arithmetic exact; affine is a
  single global transform; DETERMINISM — variant regeneration
  byte-identical (the suite is exactly reproducible from code+seed).
- **V-E2E — per-task certification rulings: all five NOT CERTIFIED**
  (cable: not as issued). Independent re-runs (14 episodes × 3 variants)
  matched validation_results.json EXACTLY (42/42, to 4 decimals) — the
  numbers are real, the harness deterministic. Calibration corrections
  ruled legitimate calibration, not result-tuning (fitted on
  outcome-independent tie points; thresholds unchanged), with the caveat
  that the validation set doubled as the tuning set for global iterations
  (logged per the rules; no held-out confirmation exists).

Coordinator's certification ruling: I adopt the verifiers' rulings in
full. **No task's twins are certified; Phase 6 may not use twin-suite v1.**

Also established along the way (dataset facts, now on record): the
recorded gripper channel is INVERTED (low = holding) — proven from
trajectory geometry; and several block_stack demos carrying success labels
do not physically stack under the mechanical criteria (the manipulated
cube is placed at a target point without engaging the base cube; ep411 is
the clearest) — a reference-data property the twin programme cannot fix
and any future criteria must reckon with.

## Parameter-source honesty summary

Parameter table: [MEASURED-EST] 14 / [PROP] 3 / [DEFAULT] 13. Independent
audit found the tags honest (one derivation arithmetic slip, value
unaffected). Friction and restitution are entirely literature/MuJoCo
defaults — nothing was measured from the recordings; they were the
designated first change-log candidates and were varied only inside their
declared ranges. Every value that entered a scene can be traced to a
tagged source; the failures of this round are NOT parameter dishonesty —
they are pose accuracy, criteria implementation, and bookkeeping.

## Archive & publication (Amendment 2)

- GitHub: twin code + scene JSONs + docs + validation records committed to
  `ivelin/zk0` @ `study001/fl-substrate-20261121` under `twin-replay/`,
  with a SUITE-STATUS file stating v1 is NOT CERTIFIED and why. (Frames
  and renders are excluded — reproducible from the dataset via
  code/extract_frames.py; generated XMLs excluded — regeneration verified
  byte-identical.)
- Hugging Face: NOT PUBLISHED. Amendment 2 §2 admits only scenes that
  passed validation AND certification; the certified set is empty.
  Creating a `zk0bot/twin-suite-v1` repo now would mislabel the suite.
  The repo is created when a task certifies (expected: cable_clip first,
  via the v1.1 list), with pinned revision + hashes recorded in the
  GitHub reference file at that time. (Separately: the current HF token
  cannot create HF Collections at all — dataset repo is the publishable
  form when the time comes.)

## Suite versioning rules (Amendment 1 §A4, as adopted)

- The suite is code + parameter table + scene JSONs + seed (20261122);
  scenes regenerate deterministically (verified byte-identical), so a
  version = a commit of those inputs.
- PATCH (v1.x): bookkeeping corrections that change no scene and no
  score (e.g. re-issuing cable's certified-variant list after excluding
  solver-failure variants — exclusion bookkeeping only, with the
  exclusion evidence attached).
- MINOR (v1.1, v1.2…): parameter-table or scene-template changes, pose
  re-estimation, criteria-implementation fixes. Requires full
  re-validation + fresh independent verification before any certification
  claim; prior versions stay runnable and citable.
- MAJOR (v2): episode-set changes, criterion changes, or a different
  replay device (e.g. any closed-loop/servoing replay — a different
  instrument, needs its own pre-registration).
- A version's certification status is part of its name in every citation:
  "twin-suite v1 (not certified)".

## Monthly realism re-audit — protocol (Amendment 1 §A5 deliverable)

Design (from Builder C's validation report, adopted with the additions
marked †):

- Cadence: monthly; first run one month after suite v1 freezes (i.e. on
  the v1 record as it stands — the audit tracks realism drift and
  estimation-method improvements, certified or not).
- Sample: all archived tie points re-derived from the frozen records +
  20 episodes (4/task) drawn at random from the included set.
- Checks: (1) tie-point residual distribution vs the suite-v1 baseline
  (claimed median 1.4 cm — †the audit's first job is to ARCHIVE the dwell
  pairs so this is recomputable; until then the baseline is the
  builder-reported value, flagged unverified); drift >1 cm median
  triggers investigation. (2) Re-run sampled episodes' nominal variants;
  any outcome-class flip vs the frozen validation_results.json is
  investigated and logged. (3) Re-examine 5 sampled scenes against their
  source frames with the then-current best pose-estimation method; scenes
  shown unrealistic (parameter outside sourced ranges, or pose error
  >2σ) are fixed in a new suite version or retired — never silently
  edited in place. (4) Physics-parameter spot checks against any new
  sourced measurements (e.g. a physical mat measurement, cube material
  identification — the two cheapest accuracy upgrades identified).
- Per-scene realism record (the format the cron consumes), one JSON per
  scene under `realism/` in the suite repo:
  {suite_version, episode, task, last_audit_date, tie_residual_cm,
  nominal_outcome_agreement, pose_error_estimate_cm, parameter_sources_ok,
  status: current|suspect|retired, auditor_model, notes}
- Output: a dated audit note in the suite docs; suite version increments
  only through the same builder + independent-verifier path as v1.
  Fix-or-retire is decided per scene from the record; a scene whose
  status is retired leaves the certified set in the next minor version.
- The parent creates the cron when suite v1 lands; this report is the
  protocol it runs.

## What certification would require — the v1.1 fix list (costed)

Ordered by dependency. Estimates are engineer-agent hours + DGX CPU
(replays are cheap: a full 485-variant set ran in ~5–8 min wall).

1. **Criteria implementation** (V-Code F-01/F-02/F-05): implement the
  registered contact/support tests (block contact+support, basket
  support, clip closure state), threshold the secondary criterion in
  output, resolve the fps question from the dataset's own metadata
  (LeRobot fps field) and re-run at the registered rate. ~3–4 h + 1 h
  re-validation.
2. **Solver-failure detection + certified-set re-issue** (cable): detect
  chain explosions in-harness (segment displacement bound), re-issue
  validation_results with exploded variants excluded per Amendment 1,
  disclose the explosion rate as a standing caveat; investigate the
  explosion cause (ball-joint chain stiffness/timestep) — fix or accept
  with the rate published. ~2–3 h. This alone can bring cable_clip to a
  certifiable state (episode agreement already 20/20 on valid variants).
3. **Pose pipeline repair**: re-estimate the 7 fabricated tool poses
  (grounding method, as the verified controls used); audit all detect4
  episodes for the identical-pixel signature; publish per-entry
  authoritative-source reconciliation for the 27 inconsistent block
  entries; archive the 93 dwell pairs. ~3–4 h. Then re-run tool/block.
4. **Pose accuracy upgrade** (the structural gap): FK-anchored pose
  estimation — use grasp/place dwell tie points (which measure object
  positions to ~1.4 cm in the builder's residuals) as pose anchors for
  all tasks, not just calibration; target σ ≲1 cm. Plus the two cheap
  physical measurements if the founder ever has the rig to hand: mat
  dimensions and cube material. ~6–10 h research-grade work; success not
  guaranteed — this is the honest bottleneck for block/ring/tool.
5. **Scene fidelity fixes**: colour variants into the XML (11 block +
  ~4 ring episodes), tray slot recess geometry, recess-count correction
  in Builder A's docs, ring-mass derivation fix. ~1–2 h.
6. **Bookkeeping**: param_hash extended to all build/replay code;
  changelog written contemporaneously (timestamps); peg jitter removed
  or pre-registered as its own amendment with justification. ~1 h.

Total v1.1: ~15–22 agent-hours + <2 h DGX CPU. Expected outcome if all
land: cable certified; eye_drops improved but placement-limited;
block/ring/tool hinge entirely on item 4.

## Phase 6 plan — closed-loop checkpoint evals (NOT started; gated)

Gate: Phase 6 runs only on CERTIFIED tasks of a certified suite version.
On today's record that set is empty, so Phase 6 is a plan, not a launch.

Design (when the gate opens):
- Subjects: the Study 001 checkpoints (base, specialists ×5 full-FT,
  spec_exp ×5 when available, C-exp, FL round-30, sequential) — 13–14
  checkpoints, on certified tasks only.
- Rollout: the policy consumes sim renders from the three matched camera
  viewpoints (front/top fixed cameras fitted to the recordings' geometry;
  wrist camera arm-mounted in sim), proprioception from sim state;
  actions applied to the sim arm's actuators (the twin scenes gain an
  actuated arm variant — the kinematic body is replaced by the vendored
  model with position actuators; control rate = dataset fps). Initial
  state = the episode's certified variant set (nominal + certified
  sampled variants), so each checkpoint × episode yields a small
  distribution, not a point.
- Scoring — DUAL, both recorded per episode (Amendment 1 §A3):
  (a) mechanical object-state criteria (the pre-registered definitions,
  as implemented post-fix-1); (b) pass/fail judging of the rollout
  RECORDINGS (rendered multi-view videos) by frontier models applying
  the fixed rubric — two independent model judges where practical,
  agreement recorded; disagreements between (a) and (b) surfaced per
  episode in the results table, never silently resolved.
- Cost model (estimates, assumptions stated): demo replay measured
  ≈1–3 s CPU per episode-variant (485-variant set ≈5–8 min wall on 8
  procs). Closed-loop adds per-step policy inference + 3-view rendering:
  at ~300 steps/episode and SmolVLA-class inference, budget ≈2–5 min per
  episode-variant on one GPU stream. Per checkpoint per certified task
  (20 episodes × 3 variants): ≈2–5 GPU-hours. Full matrix on one
  certified task ≈ 13 checkpoints ≈ 26–65 GPU-hours — schedule only
  after DGX training contention ends, and consider nominal+2 variants
  and the canary discipline from the FL queue (milestone judging) to
  cut waste. CPU-side sim/render can split to the Pop!_OS desktop.
- Output: per checkpoint × task — mechanical completion distribution,
  judged completion distribution, inter-judge agreement, (a)/(b)
  disagreement list; committed + independently verified before any
  number is reported as settled (standing rule).

## Phase 7 plan — blinded founder set (NOT started; gated on Phase 6)

- Draw ~12 rollouts at random from Phase 6's judged set, stratified
  across checkpoints and outcomes, checkpoint identity hidden (the
  blinded-test machinery from 2026-11-22 — sealed key, neutral case
  labels — reused; colours now render correctly after fix 5).
- The founder judges pass/fail from the videos; agreement with the
  mechanical criteria and with the model judges is scored and filed with
  the eval record. Disagreements are findings against the judging layer,
  not embarrassments to be averaged away.

## Files

- DGX workdir (full record): /home/openclaw/muse-work/zk0-study001-twin-20261122/
- Notes mirror: twin-suite-v1-docs/ (scene inventory, physics parameters,
  asset inventory, pose estimation, twin validation report, 4
  verification reports) beside this report in
  ~/workspace/research_notes/zk0-bench003-design-20261119/
- GitHub: ivelin/zk0 @ study001/fl-substrate-20261121, folder twin-replay/
  (commit recorded in twin-replay-progress-20261122.md).
