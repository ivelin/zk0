# Twin-Suite v1.1 — Full Re-Validation Report (Re-Validation Runner, 2026-11-22)

Run: 2026-11-22 19:09:55–19:16:23 CST (DGX clock), CPU-only, `nice -n 10`,
≤4 workers, no GPU use. DGX training was live during the run (GPU 96% at a
mid-run check — the training workload, not this job) and was not disturbed.
Code/poses/parameters/criteria were run exactly as the three v1.1 builders
landed them; NOTHING was changed by this runner. No crashes, no blocking
code bugs, no changelog entry required (next free number #17 unused).

Outputs: `validation/validation_results_v11.json` (new). The v1 record
`validation/validation_results.json` is preserved byte-identical: the landed
`run_validation.py` hardcodes its output to that path, so the runner backed
it up before the run, copied the fresh output to the v11 filename, and
restored the v1 file (SHA-256 before = after =
`396378dca6e545dc4243540e188b6363fdf88084ba70c6dcbfc66e4634d7c5b5`).
Runner script: `validation/run_revalidation_v11.sh`; logs:
`logs/revalidation-v11-{status,regen1,regen2,validation}.log`.

## 1. Scene regeneration + determinism

`code/build_scenes.py` run twice from current code + scene JSONs (K=5,
seed 20261122):

- Both runs: **500 variant scenes built, 0 skipped** (600 XML files total
  including the 100 nominal copies).
- Aggregate SHA-256 over all 600 XMLs (`sha256sum *.xml | sha256sum`):
  - Run 1: `48fccccef3dfe168feaddd24b1b08821481c742d5dd76d2c2438b532a7730843`
  - Run 2: `48fccccef3dfe168feaddd24b1b08821481c742d5dd76d2c2438b532a7730843`
  - **DETERMINISM: MATCH (byte-identical).**
- Note: this hash differs from Builder 3's interim hash
  (`85d3f51f…`) because Builder 1's harness landing and Builder 2's pose
  repairs changed hash inputs after Builder 3's regen; determinism within
  the final landed state is exact, which is the property that matters.

## 2. Full re-validation — headline table (same format as the v1 completion report)

Episode REPRODUCED = ≥60% of VALID variants agree with the recorded outcome
(Amendment 1). Denominator 97 (tool_insert ep630/871/876 OUT, pre-registered).

| Task | In | Reproduced | Mean variant agreement | Bar (≥80%) |
|---|---|---|---|---|
| block_stack | 20 | 0/20 | 0.00 | NOT MET |
| eye_drops_to_basket | 20 | 3/20 (ep1757, 1798, 1810) | 0.28 | NOT MET |
| ring_insert | 20 | 0/20 | 0.01 | NOT MET |
| tool_insert | 17 | 0/17 | 0.00 | NOT MET |
| cable_clip | 20 | 20/20 — **TO BE SUPERSEDED per V11-01** (see §5) | 1.00 — superseded | — |

### Before / after (v1 → v1.1)

| Task | v1 reproduced | v1.1 reproduced | v1 mean agree | v1.1 mean agree | v1 valid variants | v1.1 valid variants | v1.1 secondary (nominal) | v1.1 solver failures |
|---|---|---|---|---|---|---|---|---|
| block_stack | 0/20 | 0/20 | 0.00 | 0.00 | 100/100 | 100/100 | 4/20 | 0/100 |
| eye_drops_to_basket | 4/20 | 3/20 | 0.28 | 0.28 | 96/100 | 99/100 | 0/20 | 0/100 |
| ring_insert | 1/20 | 0/20 | 0.06 | 0.01 | 100/100 | 100/100 | 7/20 | 0/100 |
| tool_insert | 0/17 | 0/17 | 0.00 | 0.00 | 77/85 | 84/85 | 0/17 | 0/85 |
| cable_clip | 20/20 | 20/20 † | 1.00 | 1.00 † | 100/100 | 100/100 | exempt (null) | **0/100** |

† Cable agreement figures are reported for completeness and are **to be
superseded per V11-01** (§5). Cable solver-failure counts are valid data.

Reading of the deltas (runner's factual notes, not a certification):
- **eye_drops 4→3 with churn**: v1's reproduced set {1757, 1778, 1810, 1811}
  became {1757, 1798, 1810} under the new contact/support primary criterion —
  1778 and 1811 dropped, 1798 entered. Mean agreement is essentially
  unchanged (0.2775 → 0.2825). Notably, v1.1 nominal secondary passes = 0/20:
  no eye episode both reproduces the outcome class AND lands within 3 cm on
  the nominal variant.
- **ring 1→0**: the single v1 reproduction (ep253) was already flagged in
  the v1 completion report as "partly spawn coincidence"; with peg/board
  fixture jitter removed it no longer reproduces. Secondary passes 7/20 on
  nominal (≤2 cm planar without engagement) — position is often close, the
  3 mm-clearance engagement boolean is what fails.
- **tool 0→0 despite the 7 pose repairs**: all 7 repaired episodes (697,
  779, 784, 885, 888, 892, 903) still show 0.0 variant agreement under the
  new slot-frame/contact criteria. The repairs did remove the build-defect
  symptom: tool valid variants rose 77/85 → 84/85 (v1's 8 tool build
  defects, all on fabricated-pose episodes, are down to 1).
- **block 0→0**: unchanged headline; 4/20 nominal secondary passes show
  near-placement without the contact/exact-support primary being met.

## 3. Solver failures (v1.1 detection + exclusion active)

- **cable_clip: 0 of 100 variants flagged solver_failure at dt 0.0005**
  (v1 at dt 0.002: 10/100 numerically exploded, incl. 6 of 20 nominals).
  The global 2000 Hz timestep eliminated the explosion mode on the full
  cable set — this extends Builder 1's 2-episode evidence base to all 100
  variants. No MuJoCo instability WARNINGs in the validation log (0).
- All other tasks: 0 solver failures. Total across suite: **0/485**.

## 4. Exclusion log (complete)

- Episode-level (pre-registered, unchanged): tool_insert ep630, ep871,
  ep876 — initial-pose confidence=low. Denominator stays 97.
- Variant-level build defects: **2 of 485** (v1: 12 of 485):
  - eye_drops_to_basket ep1808 v4 — "carton: displaced 0.091 m / z=0.017
    after pre-settle" (the known platform-edge spawn effect).
  - tool_insert ep697 v3 — "tool: displaced 0.035 m / z=0.005 after
    pre-settle".
- No run errors, no unexpected skips in scene building (0 skipped, both
  regens), no solver-failure exclusions (none occurred).

## 5. Cable agreement — TO BE SUPERSEDED PER V11-01

Per the coordinator's mid-run note: V-Code verification found a cable-only
scoring defect (V11-01 — clip_block jittered in scenes but scored against
its nominal pose). The cable agreement numbers in §2 (20/20, mean 1.00)
were produced by that scoring path and **will be superseded by a corrected
cable re-run after the code fix**. The cable **solver-failure / explosion
counts (0/100 at dt 0.0005) are unaffected and stand as valid data**.
Block/eye/ring/tool results are unaffected by V11-01. This runner patched
nothing.

## 6. Sanity checks

- **param_hash**: v1.1 results stamped `4d8d350b6907427c`; recomputed from
  the current code state after the run: `4d8d350b6907427c` — MATCH. Differs
  from v1's `5ef4c692a314e87b` as required. The same hash is stamped in the
  scene XML headers (spot-checked ep697_v0).
- **fixture_jitter**: 0 occurrences across all 500 variant draw logs.
- **Timestep/FPS**: `twin_params` TIMESTEP = 0.0005, FPS = 20.0 (read from
  dataset metadata); XML spot-checks (tool ep697_v0, cable ep1939_v0) both
  carry `timestep="0.0005"`.
- **Repaired tool poses validated as repaired**: all 7 episodes (697, 779,
  784, 885, 888, 892, 903) were in the validated set with their new poses.
  Spot-check ep697: scene JSON carries the repaired pose
  `[0.1682, -0.0981, 0.005, null]` (fabricated value preserved under
  `pose_v1_fabricated`), and the regenerated XML (header:
  twin-suite-v1.1, param_hash above, frame_correction=fk-affine-v2) places
  the tool body at (0.0691305, 0.125156, 0.005) — the repaired B-frame pose
  carried through the suite's standard frame correction.
- **Suite/seed**: results carry suite_version twin-suite-v1.1, seed
  20261122, K=5.

## 7. Anomalies / notes for the coordinator

1. **ep697 JSON bookkeeping inconsistency** (Builder 2 record, not a run
   defect): `scenes/tool_insert__ep697.json`'s `v11_repair.chosen_source`
   reads "fk_anchor_only", while its `method` string and `rule_v2` block
   say grounding consensus, and Builder 2's report table lists ep697 as
   consensus (rule step 1; the two grounding candidates are ~8 px apart,
   both ~88–92 px from the FK anchor). The pose VALUES match Builder 2's
   published repair exactly, so validation used the intended pose; only
   the source label inside the JSON disagrees with itself. Recommend a
   bookkeeping correction by the owning builder — this runner changed
   nothing.
2. Validation wall time was 388 s for 485 variants at 2000 Hz with 4
   workers — far under Builder 1's 20–35 min estimate (that estimate
   assumed the v1 per-variant cost scaled 4×; in practice most episodes
   are short). Total job wall (2 regens + validation + restore): ~6.5 min.
3. v1.1 headline outcome: no task newly meets the 80% bar; the v1.1 changes
   fixed the harness defects (criteria, secondary thresholding, solver-
   failure handling, explosion mode, pose fabrications, build defects
   12→2) without moving reproduction rates — consistent with the v1
   completion report's structural finding (open-loop replay vs ~2.5–3.5 cm
   pose σ against 1–2 cm tolerances). Certification is the verifier's /
   coordinator's call, not this runner's.

— Re-Validation Runner, 2026-11-22 ~19:18 CST
