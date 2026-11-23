# Verification — Twin-Suite v2 Pre-registration (Phase 1)

**Verifier:** independent context, raw artifacts only (builder's conclusions not used).
**Date of verification:** 2026-11-22 (CST timestamps as in sources).
**Artifacts checked:**
- Pre-registration `twin-suite-v2-preregistration-20261122.md` — md5 verified locally:
  `46fb99ff71e47ab5f6ac60934578b69e` ✓ matches the brief.
- Governing memo `sim-gate-design-memo-20261122.md` — md5 verified:
  `5a8230fce3a14e42cb81a66d419a0b46` ✓ matches the brief. Its "Pre-registration
  skeleton" items 1–12 are the standard.
- SSOT Decision 62, `Business/Cocoonhive/Research/decisions/zk0-sim-decisions-20261115.md`
  line 578 (final entry).
- Sources: `twin-replay-v11-completion-report-20261122.md`,
  `study001-eval-completion_table.md`, `study001-fl-reeval-3cam-results-20261122.md`,
  `study001-a5-repair-20261122.md`, v1 pose-estimation/verification records,
  DGX split.json and DGX git repo (see §7).

## VERDICT: CERTIFIED-WITH-FINDINGS

The pre-registration implements the memo skeleton, its numbers trace to their
sources, the cross-modality exclusion/downgrade procedure is registered in full,
claims discipline is faithful, Appendix A recomputes exactly from the DGX split,
Decision 62 is faithful to the founder approval, and the freeze commit is intact.
Four minor findings (F1–F4) below; none changes a bar, a verdict rule, or a claim.

---

## 1. Skeleton items 1–12 — implementation check

| # | Skeleton item | Pre-reg location | Result |
|---|---|---|---|
| 1 | Suite identity; param_hash per v1.1 standard; external inputs SHA-256-pinned | §1 | PASS with F3 — identity, v1.1 base `param_hash a4eb3f8c3611894d`, commit `1055f4b` lineage, and SHA-256 pinning of arm model / split.json / dataset meta / eval-records manifest / control checkpoints all present; the explicit hash-coverage enumeration is narrower than the skeleton's (F3). |
| 2 | Episode set, denominator 96, exclusions named; tuning-exposure disclosed; dev subset + controls only for tuning | §2, App. A | PASS — disclosure verbatim in substance; "no twin parameter, scene, or criterion may be tuned against gate outcomes on this set" after freeze. |
| 3 | Instruments per task: Stage 0 cable (v1.1 pinned) / Stage 1 four precision tasks, four standards | §1, §3 | PASS — four standards as §3.1–§3.4. |
| 4 | Variant protocol: K=5, seed frozen, pose ± per-task σ, friction/mass in sourced ranges, labels printed in manifest | §4, §3.3 | PASS — K=5, seed lineage 20261122, labels printed. |
| 5 | Scoring: mechanical contact/support primary, positions diagnostic, solver-failure detection on and counted, secondary reported not gated | §4 | PASS — includes v1.1 thresholds (3.0 cm; ring 2.0 cm unengaged; cable exempt) and solver-exclusion counting. |
| 6 | Second layer: frontier judging, identity hidden, disagreements per episode, rate above bound ⇒ INDETERMINATE | §4 | PASS — plus judge-flagged absurd successes route to Stage 3, not PASS. |
| 7 | Controls run first; base FAILS all; ordering within noise band subject to cross-modality assumption; cross-engine bound; bounds set from controls in Stage 1, frozen before candidates | §3.2, §3.4, §6, §7 | PASS with F4 — all elements present; "all five tasks" not explicitly enumerated for the base control (F4). |
| 8 | Bars: PASS = Wilson LB ≥60% + envelope-stable; FAIL = Wilson UB <60% stable; else INDETERMINATE; promotion = PASS all tasks or named founder waiver; corners enumerated | §5, §3.3 | PASS — corners enumerated (pose ±σ/axis, friction/mass corners, engine swap). |
| 9 | Blinding: judges + human sample blinded; founder spot-check random, hidden | §9 | PASS. |
| 10 | Independent verification per class, separate contexts, raw artifacts; e2e re-runs stratified sample + controls; verdicts are verifiers' | §9 | PASS with F4 — per-class verifiers (a)–(d) and the 160/160 exact-match precedent present; class list differs from the skeleton's naming (F4). |
| 11 | Reporting: per-task first, pooled second; Wilson, variant count, solver exclusions, envelope result, unanchored label | §10 | PASS — §10 is a superset (adds cross-engine result and parameter labels). |
| 12 | Change control: post-freeze change = new suite version, re-validation + fresh verification | §11 | PASS — patch/minor/major rules inherited from v1.1; monthly realism re-audit carried over. |

No skeleton item is missing. No bar, verdict rule, or claim is weakened.
Additional pre-reg content beyond the skeleton (freeze rule, §3.2 20-agent-hour
Drake time-box, §12 compute discipline / job guard) strengthens, not weakens.

## 2. Numbers against sources

| Number | Pre-reg | Source check | Result |
|---|---|---|---|
| Denominator 96; per-task 20/20/20/16/20 (block/eye/ring/tool/cable) | §2, App. A | v1.1 report: "denominator 96 (Amendment 3)"; recomputed from DGX split (§5 below): 100 − 4 tool exclusions = 96 | ✓ |
| Exclusions tool ep630, ep871, ep876 + ep779 | §2 | ep630/871/876: v1 pose-estimation record "low-confidence poses (no pose emitted)", excluded from v1 validation denominator 17/20; ep779: v1.1 Amendment 3, denominator 97→96 (v1.1 report lines 79–82, 42) | ✓ |
| Per-task σ: block 2.5, ring 2.5, tool 3.5, eye_drops 3.5 cm/axis | §3.3, §4 | v1.1 report line 88–89: "image pipeline — block ≈2.5 cm, ring ≈2.5 cm, tool ≈3.5 cm, eye_drops ≈3.5 cm" | ✓ |
| K=5 | §4, §5 | Memo skeleton item 4 / Option C machinery | ✓ |
| PASS bar: Wilson lower bound ≥60% | §5 | Memo Stage 2 semantics; founder-signed sim bar per memo | ✓ |
| cable pinned HF revision `579ad96b960d1c6ad756b3e08518aa0b04eb60e7` | §1 | v1.1 report line 159: identical full revision for `zk0bot/twin-suite` | ✓ |
| v1.1 `param_hash a4eb3f8c3611894d` | §1 | v1.1 report line 26; memo line 68 | ✓ |
| FL weights SHA-256 `5b1cbf290b8b4beeeb9a77473f0d03c0e3ffce8c038b7aab56bc6795fc7a84b4` | §6 | `study001-a5-repair-20261122.md` line 38 and FL 3-cam re-eval records: identical full hash | ✓ |
| Base pinned revision `5e8d12a6…` | §6 | P1-verified pinned snapshot in multiple Study 001 records (e.g. armscope verification, full `5e8d12a6e2975b0e5fce7c8caf47c371d257b6`); pre-reg cites the prefix, as the brief specified | ✓ |
| Controls: base 0, FL 26, C-exp 70 (/100) | §3.4, §6 | Base 0/100 and C-exp 70/100 in the final eval table; FL 26/100 in the verified 3-camera re-eval (`study001-fl-reeval-3cam-results-20261122.md` line 41, pooled 26/100). See F1 on the source label. | ✓ (F1) |

## 3. Cross-modality assumption + exclusion/downgrade procedure

PASS. §3.4 states the assumption explicitly ("the reference ordering comes from
open-loop offline replay; the gate measures closed-loop sim rollouts") and
registers a concrete 5-step procedure, satisfying memo Option B item 4's
requirement to name how the exclusion is argued: (1) confirm base-fails-all;
(2) compare per-checkpoint failure taxonomies between offline-replay record and
gate rollouts; (3) a pair may be excused only with a specific modality mechanism
**documented with measurements** (example given: closed-loop compounding
divergence measured as trajectory error growth, with no corresponding open-loop
defect); (4) any excused pair **downgrades** the ordering control for this suite
version to base-fails-all + cross-engine agreement, recorded with its evidence
in the certification report; (5) no evidenced mechanism ⇒ control FAILS, task
not certified. "There is no third outcome and no silent pass." This is at least
as strong as the memo requires.

## 4. Claims discipline

PASS. §8's would-support list matches the memo's list (funnel verdicts
three-valued; relative ranking on the five study tasks stable across envelope
and two engines; the bar-clearing-track funnel claim; sim failure-taxonomy
claims; cable_clip v1.1 demo-replay claims). The would-NOT-support list matches
all six memo items: no absolute real-world rates; no per-episode reproduction
for the precision tasks; no novel-task generalization; no contact-force fidelity
beyond the audit; no other embodiments/setups/task families; no claim PASS
guarantees the real-arm bar (≥50% over ≥10 trials/task — "PASS earns the arm
session"). The mandatory label appears verbatim in §8 (and §10 reporting):

> "funnel verdict — sim bar unanchored to real-arm delta (no calibration pairs yet)"

## 5. Appendix A — recomputation from the DGX split

Source read on the DGX (2026-11-22, via `~/workspace/tools/dgx.sh`):
`/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/split/split.json`
(seed 20261122, per_task 20).

- Split lists 20 episodes per task, 100 total. Tool_insert split list includes
  exactly ep630, ep779, ep871, ep876; removing those four yields the Appendix A
  tool list (622, 629, 683, 684, 697, 784, 824, 830, 852, 859, 883, 885, 888,
  892, 902, 903 — 16). All other tasks' Appendix A lists are identical to the
  split lists, episode for episode. **Gate set: EXACT MATCH, denominator 96.**
- Development subset (seed 20261124 per pre-reg) and envelope subset (seed
  20261125 per pre-reg §2/App. A): fallback checks all PASS — exactly 4
  episodes per task in each subset; every member is inside the gate set;
  dev ∩ envelope = ∅ for every task.
- Exact RNG reproduction: NOT reproduced. The generation script was not among
  the artifacts provided, and the stated seeds alone did not reproduce the
  draws under the standard schemes tried (single-stream `random.Random(seed)`
  sampling in split order, fresh RNG per task, seed+task-index / seed+task-name
  variants, sample-8-then-split, shuffle-then-split, NumPy `default_rng`).
  This is F2: the seed labels as written are not self-sufficient to regenerate
  the subsets; the subsets themselves are valid by the fallback checks.

## 6. SSOT Decision 62

PASS. Decision 62 (line 578, final entry) records: founder approval
2026-11-22 21:34 CT and the 21:25 CT directive; demo-replay RETIRED for the four
precision tasks, KEPT for cable_clip / pose-tolerant classes, Option E CLOSED;
the four instrument standards; closed-loop gate, K=5, denominator 96; three-valued
verdicts with the Wilson ≥60% lower-bound PASS bar; bounds-from-controls frozen
in a Bounds Amendment before any candidate; the exact unanchored-bar label;
builder ≠ verifier. It claims only that the standard is **adopted and
pre-registered** — it does not claim instrument certification has happened, no
gate result, and no number beyond the source records (σ ≈2.5–3.5 cm, FK floor
~3.3 cm, control references 0 / 26 / 70 all trace as in §2). Nothing overstated.

One nuance, not a finding against the pre-registration: Decision 62 describes
the two cheap measurements as "attempted from records, open items if
unobtainable, never guessed", matching pre-reg §3.1; the memo's Option B wording
("taken, since they are cheap") is stronger on its face. The SSOT and the
pre-registration agree with each other, and §3.1 states the measurements "do not
block certification" per the memo — recorded here for completeness.

## 7. Freeze integrity (DGX)

Repo `~/muse-work/zk0-study001-fl-20261121` on the DGX, checked 2026-11-22:

- `git show --stat bb11d20`: commit `bb11d202f8861b94073e329eb324dafbb3110217`,
  exactly 2 files changed, 921 insertions:
  `twin-replay/sim-gate-design-memo-20261122.md` and
  `twin-replay/twin-suite-v2-preregistration-20261122.md`.
  `git diff-tree --name-only -r bb11d20` confirms exactly those two files. ✓
- `git ls-remote origin study001/fl-substrate-20261121` =
  `bb11d202f8861b94073e329eb324dafbb3110217` — equals the required hash. ✓
- Committed file hashes: `git show bb11d20:twin-replay/twin-suite-v2-preregistration-20261122.md | md5sum`
  = `46fb99ff71e47ab5f6ac60934578b69e`; memo = `5a8230fce3a14e42cb81a66d419a0b46`.
  Both identical to the local md5s in the brief. ✓

## Findings

- **F1 (minor, source label).** §3.4 attributes the control references
  (base 0/100, FL 26/100, C-exp 70/100) to "the Study 001 judged-completion
  table". That table in fact shows FL round-30 at **25/100** (its original
  2-camera-config evaluation); **26/100 is the verified 3-camera re-eval**
  figure. The number 26 is correct for the identity-pinned control checkpoint
  (the A5-repaired 3-camera checkpoint), and §6 attributes it correctly
  ("identical to the checkpoint that scored 26/100 in the verified 3-camera
  re-eval"). Fix by a dated amendment correcting the §3.4 source label; no
  number changes.
- **F2 (minor, reproducibility).** Appendix A subset draws could not be
  reproduced from the stated seeds (20261124 / 20261125) with standard RNG
  schemes, and no generation script is among the frozen artifacts (see §5).
  Fallback checks (size 4/task, disjoint, membership in gate set) all pass, so
  the subsets are usable as frozen lists — they are printed in full in
  Appendix A, which is what the freeze actually binds. Recommend a dated
  amendment or Stage 1 manifest note naming the generation script/procedure
  so the seeds are independently regenerable. (Note: the coordinator's
  description of the envelope subset as continuing the dev subset's RNG stream
  could not be confirmed or refuted from the artifacts; the pre-registration
  itself labels it seed 20261125.)
- **F3 (minor, enumeration).** Skeleton item 1 lists param_hash coverage as
  "all pipeline code, parameter doc, calibration data, pose-file manifest,
  fixtures". Pre-reg §1 says the hash is computed "by the v1.1-standard hasher
  extended over all v2 pipeline code, the v2 parameter table, the Drake scene
  files, and the variant manifests" — calibration data, pose-file manifest, and
  fixtures are not named. The "v1.1-standard hasher" phrase plausibly
  incorporates them by reference, but the Stage 1 suite manifest should
  enumerate the actual hashed inputs so coverage is checkable, not inferred.
- **F4 (minor, explicitness).** (a) Skeleton item 7 requires the base checkpoint
  to FAIL **all five** tasks; pre-reg §3.4 says base "FAILS every task" inside
  the Stage 1 (four precision tasks) section, while cable_clip gate evals are
  expressly out of this build's scope (§ "Scope of this build", Phase 6 under
  separate v1.1 authorization). The five-task reading is available but not
  enumerated; a dated amendment should state explicitly whether the v2 base
  control includes cable_clip. (b) Skeleton item 10 names verifier classes
  (objects/assets, environments/poses, code, end-to-end); pre-reg §9 names
  (parameter audit; Drake port + agreement harness code; envelope/gate harness
  code; end-to-end). The v2 classes are a reasonable re-cut for an
  instrument-certification build and the end-to-end standard (all control runs
  on the envelope subset + stratified nominal sample, exact-match) is
  preserved, but objects/assets and environments/poses verification is not
  separately named in v2 — Stage 1 verification planning should map the §9
  classes onto the skeleton classes explicitly.

## Bottom line

Pre-registration is frozen, faithful to the governing memo and to Decision 62,
numerically sourced, and the freeze commit is intact on the remote branch.
**CERTIFIED-WITH-FINDINGS** (F1–F4, all minor; F1 and F2 are best closed by
dated amendments before Stage 1 control runs, per the pre-registration's own
change-control rule).
