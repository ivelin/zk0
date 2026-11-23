# Study 001 Twin-Replay — v1.1 Completion Report, 2026-11-22

Coordinator synthesis. Builders: B1 (harness code), B2 (poses), B3 (scene
fidelity), fix-pass builder, micro-fix builder, three validation runners.
Verification: five independent contexts (V-Objects+Assets,
V-Environments+Poses, V-Code, V-Code-Delta, V-E2E), raw artifacts only.
Governing docs: twin-replay-preregistration-20261122.md, Amendment 1,
Amendment 2, Amendment 3 (v1.1 ep779 bookkeeping).

## Bottom line

**twin-suite v1.1: cable_clip is CERTIFIED. The other four tasks are NOT
CERTIFIED.** Every item on the v1.1 fix list landed and was independently
verified. The harness defects that invalidated v1's scoring (unimplemented
criteria, unthresholded secondary, phantom-clip scoring, solver explosions
counted as valid, fabricated poses, hash gaps) are closed and
regression-evidenced. The four uncertified tasks fail for the reason v1
diagnosed and v1.1 confirmed with better instruments: pose-estimation
accuracy from the recordings (σ ≈ 2.5–3.5 cm) is structurally larger than
the tasks' success tolerances (1–2 cm), and the one upgrade path that
could have closed it — FK-anchored pose estimation — was implemented,
measured honestly on held-out data, and is *worse* than the image pipeline
on every task. No global parameter change within the sourced ranges closes
that gap, and none was forced.

Frozen state: param_hash `a4eb3f8c3611894d`; denominator 96 (Amendment 3);
variant regeneration byte-identical (aggregate SHA-256
`e254e311ede3e975639019cb4edc0d5e1d9e9c9a82b2b92fff0c134cff48da2e` over
600 XMLs, twice); RUN2→RUN3 deep-JSON equality excluding the hash stamp
(the micro-fixes changed no scene and no score — proven, not assumed).

## Per-task reproduction — before / after

Episode REPRODUCED = ≥60% of valid variants agree with the recorded
outcome (Amendment 1). Task bar ≥80%.

| Task | v1 | v1.1 final | Mean agreement v1 → v1.1 | What changed |
|---|---|---|---|---|
| block_stack | 0/20 | 0/20 | 0.00 → 0.00 | Criteria now mechanical (contact+support); poses reconciled (provenance, no values moved). Still pose-limited. Secondary (nominal, newly thresholded): 4/20. |
| eye_drops_to_basket | 4/20 | 3/20 | 0.28 → 0.2825 | Churn is honest-criteria churn: ep1757 now passes the primary by a pure contact test; 1778/1811 out, 1798 in. Secondary 0/20 — even reproduced outcomes do not reproduce the recorded *placement* (nominal finals off by 7–12 cm). |
| ring_insert | 1/20 | 0/20 | 0.06 → 0.01 | The single v1 "reproduction" (ep253) was a spawn coincidence created by jittering the peg+board fixture; with the fixture at nominal it disappears. That is the fix working, not a regression. Secondary 7/20. |
| tool_insert | 0/17 | 0/16 | 0.00 → 0.00 | Denominator 17→16 by the ep779 ruling (below). All 7 fabricated poses repaired (one later re-anchored by verification evidence); build defects fell suite-wide 12/485 → 2/480; agreement still 0 — pose σ, not fabrication, is now the binding limit. |
| cable_clip | 20/20 on a wrongly-counted set | **20/20 on the full valid set** | 1.00 → 1.00 | Explosions 10/100 → **0/100** (global timestep fix); V11-01 phantom-clip scoring found by V-Code and fixed; the final 20/20 is the first cable score produced by correct scoring of a fully valid variant set. Certified set = all 100 variants. |

Suite-wide solver failures: 0/480 (v1: 10/100 cable variants exploded and
were counted as agreeing).

## The fix list, as executed

1. **Criteria** — the pre-registered contact/support tests are implemented
   (`contacts_with()` is now the scorer for block/eye/ring/tool/cable);
   position values are diagnostics only. Secondary criterion thresholded
   (3.0 cm; ring 2.0 cm unengaged; cable exempt) and reported per episode.
   FPS resolved from the dataset's own metadata: the LeRobot `meta/info.json`
   says **20 fps** — v1's rate was right, the pre-registration's 30 was
   wrong; the code now reads the rate from metadata permanently.
2. **Cable** — solver-failure detection in-harness (non-finite / radial
   >1.0 m / z outside [−0.05, +0.50] m ⇒ excluded per Amendment 1), and the
   cause fixed globally: the stiff ball-joint chain is integrator-unstable
   at 500 Hz; timestep 0.002 → 0.0005 for all tasks/variants eliminated
   explosions across the full set (0/100, independently re-run). Certified
   set re-issued as the full 100 variants under v1.1 (a minor version with
   full re-validation + fresh verification, per the versioning rules —
   stronger than the patch the v1 report anticipated, because the fix
   changed physics, not just bookkeeping). **Cable carried through
   certification and publication (below).**
3. **Poses** — the 7 fabricated tool poses re-estimated by the verified
   grounding method; the identical-pixel audit re-scanned all 270 preserved
   entries (operative chrome-lock = exactly the 7, no more); the 27 block
   entries reconciled per-entry (published JSON authoritative in all 27 —
   the defect was missing provenance, now supplied; no values changed);
   the dwell tie-point pairs are archived this time
   (`scenes/dwell_tiepoints_v11.json`, 275 dwells, 135 paired) and every
   quoted residual recomputes from them — including the honest correction
   that v1's "12.8 → 1.4 cm" improvement was in-sample: held-out is
   3.21–3.43 cm. Verification then caught one repair the builders got
   wrong: **ep784** was re-anchored to the visible tool fragment (two
   independent verifier picks + a fresh grounding pick agree within
   ~25 px; the FK anchor it replaced sat 6.7–7.7 cm off), and **ep779**
   (FK-only, visually unconfirmable) was relabeled indeterminate and —
   by the pre-registration's own low-confidence exclusion rule, recorded
   as Amendment 3 — moved OUT of the denominator (97 → 96; its interim
   agreement was 0.0, no headline moves).
4. **Pose accuracy upgrade — honest negative.** FK-anchored estimation was
   fully implemented (4 dwell selectors × 2 anchor models, train/test
   split by episode parity, all σ held-out) and is worse than the image
   pipeline on every task and kind. Achieved σ (per axis, held-out):
   image pipeline — block ≈2.5 cm, ring ≈2.5 cm, tool ≈3.5 cm,
   eye_drops ≈3.5 cm (pooled dwell corroboration ≈2.6 cm/axis);
   FK-anchored best variant — block 4.9/7.2, ring 3.1/3.2, tool 3.0/2.5,
   eye 4.6/4.6 cm (grasp/place). Even the oracle-dwell floor is
   2.6–4.4 cm: FK↔image frame agreement (~3.3 cm median), jaw-offset
   variation with wrist orientation, and dwell identifiability bound what
   the recordings contain. **The σ ≲ 1 cm target is unreachable from the
   recordings, with evidence.** FK estimates were not substituted; FK
   anchoring served as cross-check/adjudicator only (134 additive
   `fk_anchor` fields).
5. **Scene fidelity** — per-episode colors wired into the XMLs (29
   body-instances corrected across 20 episodes; independently recomputed,
   0 errors; one source-data correction, ring ep48 yellow→red, verified
   against its frame); tool tray slot recess modeled (settle test:
   0.0459 m in-slot vs 0.0499 m flat; depth 5 mm is a labeled [ESTIMATE]);
   recess count corrected to 6 in both source docs; ring-mass derivation
   arithmetic fixed (≈14.5 cm³ → ≈10 g; value and range unchanged).
6. **Bookkeeping** — `param_hash` now covers all pipeline code (incl.
   patch/finalize/builder2 files), the parameter doc, the calibration
   data, and a sorted manifest of all 100 pose files (+ fixtures.json
   after delta finding D-01); changelog entries are contemporaneous and
   clock-stamped (the two post-dated v1.1 entries were corrected on the
   record, originals left as-written); peg+board jitter removed — and its
   cable twin (clip_block jitter, V11-01) removed with it, the clip now
   scored from the sim body; external inputs not under the hash (arm
   model, trajectories, split, dataset metadata) are pinned by SHA-256
   reference in HF-REFERENCE.md (finding D-02).

## Verification — per-class verdicts

- **V-Objects+Assets — CERTIFIED-WITH-FINDINGS.** Every Builder 3 claim
  re-derived (colors 0 errors; recess settle reproduced exactly; counts
  and arithmetic confirmed). Minor (uncorroborated asset-era label)
  corrected by FP-K. Cable content provisionality resolved by V-E2E.
- **V-Environments+Poses — CERTIFIED-WITH-FINDINGS.** All headline claims
  re-derived exactly (audit scan, 62/62 reconciliation distances, dwell
  residuals 12.83 / 3.21 / 3.43 cm, all eight fix-4 σ values to the
  printed digit). Findings produced the ep784 re-anchor, the ep779
  relabel/ruling, and three bookkeeping corrections — all landed.
- **V-Code — NOT-CERTIFIED, then CERTIFIED-WITH-FINDINGS on delta
  verification.** The first pass caught the blocking V11-01 phantom-clip
  defect (same bias direction as v1's worst finding) plus hash and
  timestamp findings. The delta pass closed every finding with live
  evidence (functional probe reproduced to the digit; hash recomputed;
  missing-file behavior verified on a copy), leaving three residuals —
  D-01 and D-03 fixed in the micro-pass, D-02 pinned by reference.
  No per-episode or per-variant special-casing exists anywhere in the
  changed code (checked twice, independently).
- **V-E2E — rulings adopted.** Independent re-run of all 100 cable
  variants + stratified samples of the other tasks (160 runs) against the
  frozen scenes: **160/160 exact matches** on every compared field.
  Denominator audit and bar arithmetic recomputed from raw records:
  zero discrepancies. Freeze integrity confirmed live.

## Certification

| Task | Status | Basis |
|---|---|---|
| cable_clip | **CERTIFIED** (twin-suite v1.1) | 20/20 reproduced, mean agreement 1.00, bar met; failure modes explained (recorded outcomes are failed executions, reproduced physically by non-seating); certified set = all 100 variants; all verification classes closed. |
| block_stack | NOT CERTIFIED | 0/20 vs ≥80% bar — pose-σ structural limit (≈3.5–5 cm placement error vs ≈1.1 cm inset tolerance). |
| eye_drops_to_basket | NOT CERTIFIED | 3/20 (15%) — same structural limit; secondary criterion 0/20 shows placement does not reproduce even when outcome class does. |
| ring_insert | NOT CERTIFIED | 0/20 — ≈3.5 cm placement budget vs ≲3 mm physical peg clearance. |
| tool_insert | NOT CERTIFIED | 0/16 — pose-σ limit compounded by residual pose error (median nominal secondary error 18.4 cm). |

Coordinator's ruling: I adopt the verifiers' rulings in full.
**Phase 6 may use the cable_clip twins of twin-suite v1.1 only.**

## Publication (certified set only, per Amendment 2 + the founder rule)

- **Hugging Face:** dataset repo
  [zk0bot/twin-suite](https://huggingface.co/datasets/zk0bot/twin-suite),
  **pinned revision `579ad96b960d1c6ad756b3e08518aa0b04eb60e7`**, public,
  CC-BY-4.0. 168 files, 19,081,155 bytes, readback-verified (HEAD ==
  pinned revision, zero size mismatches). Contents: the 20 certified
  cable episode scenes + all 120 XMLs (20 base + 100 variants), the arm
  mesh set, the governing docs, the final validation results and
  changelog, and a card that scopes certification to cable_clip only.
  Manifest aggregate SHA-256:
  `1cb900db08c8a4b3d0ff6f4e87b7ff5c728a002b1c3fb9f8c77cc8979152b773`
  (full manifest committed as `twin-replay/hf-manifest-v11.json`).
- **GitHub:** ivelin/zk0 @ `study001/fl-substrate-20261121`, folder
  `twin-replay/` — commits `902d6c4` (code+scenes+validation),
  `70d19b7` (docs), `a079f14` (SUITE-STATUS v1.1 + HF-REFERENCE),
  pushed and ls-remote-verified. SUITE-STATUS carries the per-task
  certification; HF-REFERENCE.md carries the pinned revision, the
  manifest, and the external-input pins (arm model
  `d75253eb…`, split.json `a8ca1528…`, dataset info.json `2bba6714…`,
  eval-records manifest over 800 files `bc4d351d…`).

## What still blocks the uncertified tasks — honestly

The bottleneck is information, not effort: the recordings do not contain
object poses to better than σ ≈ 2.5–3.5 cm by any method tried (image
pipeline, FK anchoring, oracle dwell selection), and open-loop replay of
closed-loop human demos cannot correct placement errors the human
corrected visually. The remaining paths are instrument changes, not
parameter changes: (a) richer pose evidence — the two cheap physical
measurements already identified (mat dimensions, cube material) help
calibration only marginally; multi-view geometric reconstruction beyond
the current pipeline is unproven headroom, not a plan; (b) a different
replay instrument — closed-loop/servoing replay — which is a MAJOR suite
version (v2) needing its own pre-registration, and which changes what is
being measured (it would grade the twin + a controller, not the recorded
trajectory); (c) distribution-level criteria (does the *distribution* of
sim outcomes match the recorded distribution) — a criteria change, also
v2 territory. Until one of those is chosen and pre-registered, block,
eye_drops, ring, and tool twins stay research-grade: useful for method
development, inadmissible as certification evidence. Disclosed per the
standing rule: the validation set doubled as the tuning set for the
global iterations of v1/v1.1 (all changes global and logged; no held-out
confirmation set exists) — one more reason the four uncertified tasks'
numbers should be read as upper bounds on their true reproduction rates,
and cable's certification rests on the strength of its margin (20/20 at
1.00 mean agreement, physically reproduced failures) rather than on
tuning sensitivity.

## Files

- DGX workdir (full record): /home/openclaw/muse-work/zk0-study001-twin-20261122/
- Notes mirror: twin-suite-v11-docs/ beside this report in
  ~/workspace/research_notes/zk0-bench003-design-20261119/
- GitHub: ivelin/zk0 @ study001/fl-substrate-20261121, twin-replay/
- Hugging Face: zk0bot/twin-suite @ 579ad96b (cable_clip certified set)
