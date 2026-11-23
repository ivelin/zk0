# Twin-Suite v2 — Closed-Loop vs Offline Gap: Mechanism Diagnosis

**Date:** 2026-11-23 · **Type:** read-only analysis of existing records.
No new rollouts, no training, no repo writes were performed.
**Inputs:** `stage1-certification-synthesis-v2.md` (+ its independent
verification and errata), L1 rollout records + trajectories on the DGX
(`zk0-study001-twin-v2-20261122/controls_battery/out/L1/`, 300 rollouts),
Study 001 offline eval records (DGX
`zk0-study001-20261119/run/eval-traj-20261122/records/`, per-episode
`gt`/`pred`/`err` arrays), `rollout-harness-v2.md`,
`fingerpad-fix-v2.md`, twin-suite v1.1 completion report.
**Method:** all numbers below were computed from the raw records by
the analysis script preserved on the DGX at `/tmp/gap_analysis.py`
(plus a compact re-run for §2). Joint errors are L2 norms across the
6 joints, in degrees (offline `gt`/`pred` are degrees; twin `qpos`
converted from radians).

> Every gate output discussed here carries the §8 label: funnel
> verdict — sim bar unanchored to real-arm delta.

## 1. The contradiction, restated with the new measurements

Offline replay (teacher-forced on the demo's own frames) orders the
checkpoints cleanly, and that ordering is visible in raw joint
tracking: per-step joint error vs the demo trajectory (median over
episodes of the per-episode mean; aggregation choice does not
affect any conclusion — verifier's note, §9) is
**base 94.7° · FL 33.8° · C-exp 22.5°**. Closed-loop in the twins,
the same quantity is **base 87.6° · FL 94.6° · C-exp 89.7°** — the
ordering vanishes entirely. Whatever the twins do to these policies,
they do it to the trained and untrained policies alike, within the
first tenth of the episode. That is the fact any mechanism must
explain, alongside 0/300 mechanical successes and a correctly
ordered near-miss (secondary) criterion (3 < 5 < 7).

## 2. Candidate (a) — closed-loop compounding error: MEASURED, and it is total

Per-step joint divergence of the closed-loop rollout (nominal
variant) from the demo trajectory of the same episode, median over
the 20 shared episode-cells per checkpoint (all four tasks):

| Checkpoint | t=0 CL / OL | 10% CL / OL | 50% CL / OL | 90% CL / OL | End CL / OL | Mean CL / OL |
|---|---|---|---|---|---|---|
| base | 15.5 / 158.1 | 99.2 / 124.9 | 79.3 / 83.0 | 102.6 / 97.5 | 121.7 / 141.3 | 87.6 / 94.7 |
| FL r30 | 16.3 / 20.1 | 86.7 / 45.6 | 108.3 / 27.4 | 89.2 / 28.4 | 122.3 / 55.2 | 94.6 / 33.8 |
| C-exp | 11.0 / 7.7 | 53.2 / 24.0 | 100.6 / 16.6 | 81.9 / 16.2 | 119.2 / 12.8 | 89.7 / 22.5 |

Reading: for the trained checkpoints, open-loop error *falls* over
the episode (C-exp 24.0° → 12.8°) while closed-loop error *explodes
within the first 10%* (C-exp 11° → 53° → ~100°) and never returns.
This is exactly the §3.4 step-3 registered example — trajectory
error growth with no corresponding open-loop defect — now measured.
Two cautions keep (a) from being the whole answer: (i) joint-space
distance from the demo is not itself failure (other valid
trajectories exist), so divergence is the *phenomenon to explain*,
not yet a root cause; (ii) divergence this fast (by 10% of episode,
before most object contact) points upstream — at what the policy
*sees* — which is candidate (e).

## 3. Candidate (e) — observation mismatch: DOCUMENTED, structural, unmeasured in the logs

The rollout harness's own build record (`rollout-harness-v2.md`
§3.4) documents that the twin's observation stream differs from the
recorded stream the policies were trained and offline-evaluated on,
in at least three ways:

1. **Wrist camera: invented.** The v1.1 scenes have no wrist camera.
   The harness synthesises one (eye 6 cm behind the gripper along
   the wrist→gripper axis + 5 cm up, target 10 cm past the site) and
   states plainly it is "not a certified viewpoint match." The
   policies were trained on a real, fixed-mount wrist camera.
2. **Front camera: re-aimed.** The scene `frontcam` renders
   near-black, so the harness substitutes `frontfix` — same
   position/fovy, different look-at target (workspace centre). The
   framing the policy sees is not the dataset framing.
3. **State channel: approximated.** The gripper component of
   `observation.state` passes through a fixed global calibration
   mapping (52.3° dataset max) in both directions; the twin's
   initial arm pose already sits 11–16° (L2) from the demo's frame-0
   joints (§2, t=0 column), before any policy action.

**What the existing data cannot decide:** L1 rollout records saved
no rendered frames (frames exist only for the 4 smoke rollouts), so
no on-disk measurement can separate "the policy misreads the twin's
views" from "the policy reads them fine but the dynamics/grasp
betray it." (e) is therefore ranked on documentary evidence plus
the speed of the §2 divergence, not on a rollout-log measurement.

**Cheapest decisive experiment (NOT run):** a first-chunk action
agreement probe — for a handful of validated scenes, run the policy
service once on the twin's rendered initial frames and once on the
recorded demo's initial frames from the identical initial state,
and compare the 50-action chunks. A large step-0 disagreement
proves observation sensitivity before any dynamics enter. Follow-on
if positive: a "ghost rollout" (policy fed recorded frames while
its actions execute in the twin) on 2 block_stack episodes.

## 4. Candidate (c) — grasp/contact completion: MEASURED as the final-stage failure, not the first

Funnel over all 300 L1 rollouts (object motion from `obj_*`
trajectories; "lift" = max object z-rise ≥ 5 mm; gripper-close
attempt = gripper joint driven below 0.6 rad at any step):

| Checkpoint | No contact | Contact, no lift | Lift (weak+strong) | Gripper-close attempts | Secondary |
|---|---|---|---|---|---|
| base | 73/100 | 15/100 | 12/100 | 100/100 | 3/100 |
| FL r30 | 42/100 | 46/100 | 12/100 | 100/100 | 5/100 |
| C-exp | 50/100 | 36/100 | 14/100 | 100/100 | 7/100 |

Per task (contact = 100 − no-contact): eye_drops is a total
no-contact failure for **every** checkpoint (0/60 rollouts touch
the carton); block_stack contact rises with training (base 3/40 →
FL 28/40 → C-exp 23/40) and lifts likewise (2 → 4 → 8 of 40);
tool_insert lifts in 4/5 rollouts for all checkpoints including
base (indiscriminate interaction). Findings:

- Policies command gripper closes in **100/100** rollouts each —
  the gripper *pattern* the offline rubric credits is present; what
  fails is placement: reaching the object (eye), and converting
  contact into a settled, criteria-satisfying placement (block:
  lifts happen, incl. C-exp's 5.5 cm cube lift, but no rollout ends
  with inset footprint + contact-with-lower + no other support).
- This matches the fingerpad record: the padded gripper pinch-lifts
  the 30 mm cube only under a fingers-down, face-centred scripted
  approach; the policy's learned approach geometry (calibrated on
  the real gripper, with the ~24 mm site-to-pinch-centre offset
  documented in the v2 record) is not the scripted one.
- Verdict: (c) is a **real, measured contributor at the final
  stage** — it is sufficient to zero out block_stack even where
  contact occurs — but it cannot explain eye_drops' 0/60 contact or
  the §2 divergence that precedes contact. It is the second
  mechanism, not the first.

## 5. Candidate (b) — initial-state distribution mismatch: WEAKENED where testable, untestable where it matters

Available test: nominal variant (best-estimate pose) vs the four
pose-jittered variants, block_stack. If initial-pose error within
the sampled distribution were discriminating, nominal should win.
It does not: secondary passes nominal 0/8, 1/8, 1/8 (base/FL/C-exp)
vs jittered 3/32, 4/32, 6/32; mean max object displacement nominal
0.000/0.027/0.034 m vs jittered 0.006/0.029/0.027 m. Within ±σ pose
jitter, outcomes are indistinguishable (at a floor, admittedly).

**What the existing data cannot decide:** all five variants share
the same *estimated* nominal pose, so a systematic per-episode pose
bias (the FK↔image frame disagreement, ~3.2–3.4 cm held-out
tie-point residual) shifts every variant together and is invisible
to the comparison above. Per-episode pose-error values were never
archived as a numeric field — only global residuals survive — so
the requested per-episode correlation **cannot be computed from
data on disk**. Indirect evidence against (b) as the primary
mechanism: the scenes are solvability-validated by executed
reference passes starting from these same initial states, and the
§2 divergence is in *arm-joint* space from step ~0, larger than the
arm-side initial offset (11–16°) within the first 10%.

**Cheapest decisive experiment (NOT run):** none needed until (e)
is decided — if the ghost rollout (§3) recovers contact, initial
poses are exonerated with it; if it does not, a 5-rollout probe
starting C-exp from the demo's exact frame-0 arm pose with
oracle-placed objects separates (b) from dynamics.

## 6. Candidate (d) — action-chunking / inference cadence: RULED DOWN by construction and by measurement

- By construction: the rollout harness mirrors the offline eval
  contract (`rollout-harness-v2.md` §2) — one query per
  50-action chunk, actions held 0.05 s at 20 fps, same pre/post
  processors, same per-(episode, chunk) noise seeding. (The one
  recorded deviation is a forced CPU pin of the base checkpoint's
  processor device — placement only, no effect on the cadence
  contract.) There is no cadence difference to explain a
  difference in outcome.
- By measurement (offline boundary means computed over all 100
  offline records per checkpoint; closed-loop over all 100 L1
  trajectories): chunk-boundary action discontinuity exists in
  both modalities and is far *larger* in the offline predictions
  that scored 26–70 (mean max-joint step at chunk boundaries:
  FL 36.7°, C-exp 29.7° in open-loop `pred`, vs slew-limited
  ≈0.13–0.17 rad in closed-loop `ctrl`; mid-chunk means 4.2°/2.9°
  vs ≈0.05–0.06 rad). Offline scoring tolerated bigger boundary
  jumps than the twins ever produce. Cadence artefacts cannot be
  the differentiator.

## 7. Ranking and recommendation

| Rank | Candidate | Status on existing data | Key numbers |
|---|---|---|---|
| 1 | (a) Closed-loop compounding divergence | **Measured** — total, by 10% of episode, ordering-destroying | CL mean err 87.6/94.6/89.7° vs OL 94.7/33.8/22.5° (base/FL/C-exp) |
| 2 | (e) Observation mismatch (invented wrist view, re-aimed front view, gripper-state mapping) | Documented structurally; **not decidable** from rollout logs (no frames saved) | 2 of 3 camera views differ from training; t=0 arm offset 11–16° |
| 3 | (c) Grasp/contact completion at settle | **Measured** contributor at the final stage | Close attempts 100/100 per ckpt; block lifts 2/4/8 of 40; 0 settled passes; eye contact 0/60 (upstream of grasp) |
| 4 | (b) Initial-state mismatch | Weakened within-distribution; systematic-bias form **not computable** from disk | Nominal ≈ jittered on secondary + displacement; no per-episode pose-error field archived |
| 5 | (d) Chunking/cadence mismatch | **Ruled down** — identical contract; boundary jumps larger in the modality that scored well | OL boundary 29.7–36.7° vs CL ≈0.13–0.17 rad |

**Synthesis:** the measured phenomenon is (a); the leading
root-cause hypothesis is (e) — the policies are being fed views
(above all a wrist view that does not exist in the training data's
geometry) that put them off-distribution from the first chunk,
which produces exactly the observed signature: immediate,
checkpoint-independent divergence, failure to even reach the
object on eye_drops, and residual final-stage grasp losses (c)
where contact does occur. (e) and (c) are not rivals: fixing the
views would still leave the settle-stage grasp geometry to prove.

**Recommended decisive experiment (for registration, NOT launched
here):** the two-step observation probe of §3 — (i) first-chunk
action agreement, sim-rendered vs recorded initial frames,
identical initial state, C-exp + FL on the 8 validated block_stack
scenes (16 policy-service calls, minutes of compute); (ii) if
agreement is poor, a ghost rollout on 2 block_stack episodes.
A positive result gives §3.4 the measured modality mechanism it
requires, localised to the observation contract; a negative result
exonerates the views and promotes (c)/dynamics to the lead, with
the §5 oracle-pose probe as its decider.

## 8. What this diagnosis does NOT do

It does not revisit §3.4, excuse any checkpoint pair, or certify
anything. It registers one measured phenomenon (a), one measured
final-stage contributor (c), one documented-but-unmeasured
structural suspect (e), and the two experiments that decide
between them — which is the input §3.4 step 3 demands before the
ordering control can be re-examined.

## 9. Independent verification (2026-11-23) — CERTIFY-WITH-FINDINGS

A verifier in a separate context recomputed every headline number
from the raw records with its own script: the §2 divergence table
(all 36 cells), the §4 funnel (pooled and per-task, incl.
gripper-close 100/100 and eye_drops 20/20 no-contact per
checkpoint), the §5 secondary counts, and the §6 boundary means
**all reproduce exactly**, and both documentary claims about
`rollout-harness-v2.md` (synthesised wrist camera / substituted
front camera; mirrored eval contract) check out verbatim. Four
wording findings, none changing any number or conclusion, were
folded back into the text above: (F1) the "mean" figures are
medians over episodes of per-episode means (§1 now says so;
mean-of-means values — CL 89.1/100.0/88.9, OL 95.9/37.5/23.8 —
support the same conclusion); (F2) §5's FL nominal mean
displacement 0.027 m sits on a rounding boundary (exact 0.0265 m);
(F3) §6's offline boundary means are over all 100 records per
checkpoint (now stated; restricted to the shared 20 cells they
are FL 35.78° / C-exp 34.24°, same conclusion); (F4) the harness's
one forced CPU device-pin deviation is now noted in §6.
