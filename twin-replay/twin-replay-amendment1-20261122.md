# Study 001 Twin-Replay — Amendment 1 (2026-11-22, ~17:50 CT)

Founder refinement issued 2026-11-22 17:47 CT, folded in by the coordinator
BEFORE any twin validation run (poses still being estimated; no demo replay
has executed). Where this amendment and the pre-registration differ, this
amendment governs; everything else in the pre-registration stands.

## A1. Verified-asset reuse first

Before building or estimating anything new, inventory the project's tested
sim assets (zk0-sim-v01-20261115 and sibling sim workdirs, SIM-1/SIM-2 and
grasp/BAM-era scenes/objects, the ivelin/zk0 repo's sim folders). Tested
assets are the default building blocks; estimation is the last resort and
stays labeled. The scene spec gains an asset-inventory section: per building
block — asset, provenance, what testing it survived.

## A2. Scene variants (replaces the single-twin design)

Per episode, a variant family of K=5 scenes: 1 nominal (Builder B's pose,
Builder A's nominal parameters) + 4 sampled variants — object initial pose
jittered within Builder B's per-pose quantitative error estimate, friction
and mass sampled within Builder A's sourced ranges (where a range exists;
otherwise ±20% around nominal, logged as a sampling choice). Variant
parameters are drawn from a fixed seed (20261122) and recorded per variant
in the scene file, so the suite is exactly reproducible.

Validation semantics (coordinator's reading of the founder's rule, stated
here so it can be overruled before results exist):
- Every validly built variant is run and logged. A variant may be EXCLUDED
  only for a build defect (spawn interpenetration, initial instability,
  solver failure), with the defect logged per variant. Outcome failure is
  never an exclusion reason in the certification statistics.
- Episode reproduction: an episode counts as REPRODUCED if ≥60% of its
  valid variants agree with the recorded outcome (primary criterion;
  secondary position criterion assessed on the nominal variant).
- Task certification bar unchanged: ≥80% of medium+ confidence episodes
  reproduced, no unexplained systematic failure mode.
- Phase-6 suite membership (for the future eval plan): the CERTIFIED
  VARIANT SET = variants that reproduced their demo outcome. Episodes with
  <60% variant agreement are excluded from Phase 6 entirely (fragile
  twins), logged. This is the founder's "excluded and logged, not tuned
  into compliance": excluded variants/episodes are never parameter-tuned
  to force a pass; global parameter changes remain governed by the
  pre-registration (full-set re-run + changelog).

## A3. Judging layer (Phase 6 plan requirement)

The Phase 6 plan in the completion report must specify dual assessment,
both recorded per episode: (a) mechanical object-state criteria in sim;
(b) pass/fail judging of the rollout recordings by frontier models applying
the fixed rubric (more than one model where practical, agreement recorded).
Disagreements surfaced per episode, never silently resolved. Phase 7's
blinded founder set is drawn from these judged rollouts.

## A4. Suite versioning

The finished suite is named **twin-suite v1**. Versioning rules (full text
in the completion report): parameter-table or scene-template changes bump
the minor version and require re-validation; episode set or criteria
changes bump the major version; old versions stay runnable (scenes are
generated from versioned code + parameter table + seed).

## A5. Monthly realism re-audit (design deliverable)

The completion report includes the audit protocol + per-scene realism
record format for a ~monthly frontier-model-assisted re-evaluation of the
suite against the human recordings, with fix-or-retire outcomes per scene.
The parent creates the cron when v1 lands; this workstream delivers the
design only.
