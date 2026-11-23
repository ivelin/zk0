# Twin-Suite v2 Stage 1 — Controls Battery Part B: Independent Verification (2026-11-23)

**Verifier:** independent context (builder ≠ verifier, pre-reg §9).
**Subject:** `controls-battery-v2.md` (builder's report) — 435 control
rollouts (L1 300 / L2 120 / L3 15) on the three Part A manifest
checkpoints, DGX workdir
`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/controls_battery/`.
**Method:** raw artifacts only — all 435 `rollouts.jsonl` records
re-aggregated from scratch (independent Wilson/§5 implementation),
checkpoint files re-hashed, harness code diffed against the certified
padded runner, retained `.npz` trajectories compared across
checkpoints, and 6 rollouts re-run by the verifier in DGX scratch
(`/tmp/verify_rerun/`) from the retained code and scenes.
Every gate output carries the §8 label: **"funnel verdict — sim bar
unanchored to real-arm delta (no calibration pairs yet)."**

## Verdict

**CERTIFIED-WITH-FINDINGS** — on the battery's execution and analysis.

**Ruling on the battery's own conclusion: the §3.4 known-ordering
control FAIL stands as settled.** It is earned by the raw records:
base fails every task, and FL round-30 and C-exp also score 0 on
every task under the primary mechanical criteria, so the registered
ordering does not reproduce and no pair was excused. This
verification certifies the battery and its FAIL conclusion; it does
not certify any task's instrument, and it authorizes no candidate
gate run (§6 remains blocked until certification lands by some
route).

## 1. Completeness — PASS

- Records: **435/435**, exactly the certified Part A matrix —
  L1 100/checkpoint (300), L2 40/checkpoint (120), L3 5/checkpoint
  (15); `worklist.jsonl` = 435 units; `completion-manifest.json`
  435/435 with the same per-layer split. All 435 keys
  (layer × checkpoint × task × episode × variant × corner) unique;
  every record `status: complete`; **solver failures 0/435,
  excluded 0/435**; 435 trajectory `.npz` retained.
- Guard: all four stream rundirs
  (`runs/controls-battery-{base,fl_round_30_a5repair,c_exp,drake}`)
  have guard `exit_code` **0** and **zero WOULD-KILL lines**.
- No v1.1 modification: every file in the v1.1 workdir
  (`zk0-study001-twin-20261122/code/`, `scenes/`) carries an mtime of
  2026-11-22 evening, before the battery window (2026-11-23
  04:49–09:07 CT); `code/replay.py` md5
  `c48b92053a9d67d3d0cc5f6c8c8fc9ed`.

## 2. The decisive check — the zeros are behaviour, not a dead harness

**(a) Verifier re-runs (6, all 3 checkpoints, 3 tasks incl. one C-exp
block and one FL cable).** Primary outcome reproduced **6/6** (all
FAIL, solver_failure false everywhere):

| Re-run | Record | Re-run | Match |
|---|---|---|---|
| C-exp block 411 v0 | F, secondary T | F, secondary T, checks identical | ✓ |
| FL cable 2068 v0 | F, secondary n/a | F, checks within 0.005 m | ✓ |
| base block 499 v0 | F, secondary F | F, objects_final to ~1e-9 m | ✓ |
| FL block 411 v0 | F, secondary **T** | F, secondary **F** | primary ✓, secondary ✗ (F2) |
| C-exp cable 1905 v2 | F, secondary n/a | F, checks within 0.007 m | ✓ |
| base eye 1609 v0 | F, secondary F | F, checks + objects_final identical | ✓ |

**(b) Correct checkpoint per stream.** The records carry
`checkpoint_id` only (F1); identity is proven by the chain:
`battery.py` CKPTS paths = the Part A manifest paths, and the
verifier's own SHA-256 of the three `model.safetensors` files at
exactly those paths equals the manifest pins —
base `7cd549ac…a01eb`, FL `5b1cbf29…84b4` (= pre-reg §6 pin),
C-exp `8f6807ed…8379`. Per-rollout logs (435 in
`controls_battery/logs/`) record each unit's invocation.

**(c) Policies acted differently per checkpoint.** From the retained
`.npz` files on identical scenes, commanded-action (`ctrl`)
trajectories differ by max-abs **0.81–2.02 rad** between every
checkpoint pair (block 411 v0, cable 2068 v0, eye 1609 v0);
per-joint `qpos` ranges are large (C-exp block 411 v0: 0.41–1.67 rad
over 296 steps, matching the builder's spot claim; its cube_a ends
~4 mm from start, `upper_contacts: [world]`). Identical actions
across checkpoints — the dead-harness signature — are absent.
Objects end essentially unmoved in all three streams: the policies
move, approach, and never form the contact the criteria require.

**(d) Scoring code.** `sim_runner_battery.py` scores via
`v1replay.evaluate(...)` imported read-only from the v1.1 workdir —
the unmodified v1.1 function. The full diff against the certified
`sim_runner_padded.py` is exactly: scenes dir → `scenes-battery/`;
`--corner` / `--scene-file` arguments; corner in the record key and
record; `engine: "mujoco"` in the record; corner in the trajectory
filename — plus one cosmetic suite-label string (F3). Nothing else.

## 3. Recomputation from raw records — PASS, exact

Independent aggregation reproduces the builder's table cell-for-cell:

- **Primary: 0 successes in every L1 task × checkpoint cell** —
  block 0/40, eye 0/20, tool 0/5, cable 0/35, pooled 0/100, for base,
  FL and C-exp alike. Wilson 95% uppers: 8.8 / 16.1 / 43.4 / 9.9 /
  3.7% respectively; **§5 class FAIL in every cell** (upper < 60%).
- **Secondary (reported, not gated): block base 3/40 < FL 5/40 <
  C-exp 7/40; zero on every other L1 task**; L2 secondary base 1,
  FL 1, C-exp 2 of 40; L3 0. Exactly as reported (but see F2).
- `analyze.py` itself is a faithful implementation (Wilson z=1.96,
  §5 class formula, Drake-validity gate, flip detection);
  `analysis.json` agrees with the independent recomputation.

## 4. §3.2 and §3.3 — PASS as characterized

- **§3.2:** pairing each L3 Drake record against its L1 variant-0
  nominal twin on (success, solver_failure): **15/15 agree**,
  recomputed. All 15 Drake records carry present, non-blank
  `frame_content` (every camera: unique_colors ≫ 10,
  non_clear_frac > 0.05); invalidated units 0/15. The builder's
  caveat stands and is correct: agreement at the floor is
  consistency evidence, not discrimination evidence.
- **§3.3:** **0 outcome flips in 120 L2 corner rollouts**,
  recomputed (every corner rollout FAILs, as does every nominal).
  Block per-corner classes are FAIL (n=4, 0 successes, Wilson upper
  49.0% < 60%), and pooled corners (n=32, upper 10.7%) FAIL,
  matching nominal FAIL. Eye per-corner: at n=1 with 0 successes the Wilson
  upper is **79.35%** ≥ 60%, so the §5 formula returns
  INDETERMINATE for all 24 eye corner cells — the builder's
  "INDETERMINATE-at-n=1 by construction" characterization is
  arithmetically right; pooled across corners (n=8, upper 32.4%)
  the eye corner class is FAIL, matching nominal. No verdict's
  stability depends on a friction corner.

## 5. §3.4 reasoning — faithful to the pre-registration

- Base-fails-all **holds** (0 on every task with a denominator; FAIL
  class everywhere). Ordering **does not reproduce** on the primary
  criterion, pooled or per task (0 = 0 = 0). Failure taxonomies are
  indistinguishable on the primary checks across checkpoints
  (recomputed L1 false-check sets: block — `contact_with_lower`,
  `no_other_support_contact`, `within_inset_footprint` false in
  120/120; cable — `cable_seated_in_clip_closed` false in 105/105;
  eye — basket checks false in 60/60).
- The refusal to invoke the downgrade path is exactly what the
  pre-reg requires. Governing lines (§3.4, pre-registration
  2026-11-22):

  > "(3) a checkpoint pair may be excused only if a specific
  > modality mechanism is **documented with measurements** (e.g.
  > closed-loop compounding divergence, measured as trajectory
  > error growth across the rollout, where the open-loop signature
  > shows no corresponding defect); (4) if any pair is excused, the
  > ordering control **downgrades** for this suite version to
  > base-fails-all + cross-engine agreement, and the downgrade and
  > its evidence are recorded in the certification report; (5) if
  > no mechanism is evidenced, the control **FAILS** and the task
  > is not instrument-certified. There is no third outcome and no
  > silent pass."

  This battery measured no modality mechanism (no trajectory
  error-growth analysis was computed); the builder's candidate
  mechanisms are labeled unmeasured hypotheses, which is the honest
  state. With no measured mechanism, step (5) is the only
  registered outcome. **The FAIL is the pre-reg working as written,
  not a judgment call.**

## 6. Draft Bounds Amendment arithmetic — PASS

Recomputed Wilson 95% uppers of a zero rate:

- Cross-engine bound: disagreement 0/15 → upper **20.39%**; the
  draft's "≤ 20%, rounded down" is arithmetically correct (and
  conservative in the tightening direction).
- Ordering noise bands: block (n=40) **8.76 → 8.8 pp**; eye (n=20)
  **16.11 → 16.1 pp**; cable (n=35) **9.89 → 9.9 pp**; tool (n=5)
  **43.45 → 43.4 pp**; pooled (n=100) **3.70 → 3.7 pp**. All match.
  Observed control separations are 0 pp on every task — inside
  every band, consistent with the §3.4 FAIL.
- Envelope item: 0 flips / 120 and the n=1 pooled-reading rule are
  as recomputed in §4 above.
- Mechanical/judging bound: honestly **left unset** — the §4
  judging layer was not run; the draft does not invent a value.
  Correct per §7 ("measured from controls, never fitted").

## Findings (none overturns any gated conclusion)

- **F1 — Checkpoint identity is indirect in the records.** Records
  carry `checkpoint_id`, not the weights hash; the identity proof
  rests on `battery.py` paths + manifest + file hashes (all
  verified here). Future batteries should stamp the weights SHA-256
  into each record.
- **F2 — Rollouts are not bit-deterministic.** 1 of 6 verifier
  re-runs flipped the *secondary* criterion (FL block 411 v0:
  recorded True, re-run False), and final object positions shifted
  ~1 cm in 2 of 6 re-runs (e.g. C-exp block 411 cube_a). Primary
  outcomes reproduced 6/6 and every gated number in this battery is
  a primary-criterion count with wide margins (all zeros), so no
  conclusion moves; but the secondary 3 < 5 < 7 ordering should be
  read as ± run noise, and any future near-threshold result needs a
  determinism/replication pass first.
- **F3 — Runner diff has one unlisted cosmetic change** (suite label
  string) beyond the builder's enumerated list. No functional
  effect.
- **F4 — Guard mirror failed on all four streams** (MIRROR FAIL
  rc=255, telemetry on-box only), disclosed in the builder's report;
  the known infrastructure gap. No no-mirror override was used.
- **F5 — L3 records carry suite label `twin-suite-v2-dev`** (the
  Drake runner's string), not the battery suite label. Cosmetic;
  layer/checkpoint/task fields are correct.

## Bottom line

Execution complete and disciplined (435/435, guard exit 0 ×4, no
v1.1 file touched), analysis arithmetically exact under independent
recomputation, zeros proven behavioural (distinct, large-motion
policy trajectories; 6/6 primary re-runs reproduce), and the §3.4
application faithful to the registered text.

**Verdict: CERTIFIED-WITH-FINDINGS. The §3.4 known-ordering control
FAIL stands as settled; no precision task is instrument-certified
by this battery; the Draft Bounds Amendment arithmetic is verified
and it may be appended on that basis (its mechanical/judging item
remaining an open item, as drafted).**
