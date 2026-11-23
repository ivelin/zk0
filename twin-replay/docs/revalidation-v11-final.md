# Twin-Suite v1.1 — FINAL Re-Validation Report (Final Re-Validation Runner, RUN2, 2026-11-22)

Run: 2026-11-22 19:25:16–19:32:08 CST (DGX clock), CPU-only, `nice -n 10`,
≤4 workers, no GPU use (GPU 96% at launch check was the live DGX training
workload, not this job; it was not disturbed). Code/poses/parameters/criteria
were run exactly as the fix pass landed them; **NOTHING was changed by this
runner** — no code, pose, parameter, or criteria edits. No crashes, no
blocking defects, so no RUN2-* changelog entry was required.

Outputs: `validation/validation_results_v11.json` (final, SHA-256
`80fc033d804b6229eb80eeb32881c8e2dea6862a591742145319c17a8976df35`).
Runner script: `validation/run_revalidation_v11_final.sh`; logs:
`logs/revalidation-v11-final-{status,regen1,regen2,validation}.log`
(status file: `logs/revalidation-v11-final-status.txt`).

## ⚠ HEADLINE DEVIATION FROM THE BRIEF — denominator is 96, not 97

The brief expected denominator 97 (only tool ep630/871/876 out). The final
run included **96 episodes and excluded 4**: the three pre-registered tool
episodes **plus tool_insert ep779**. This is a direct mechanical consequence
of the fix pass as landed, not a runner change:

- FP-H relabeled ep779's tool confidence medium → **low** (FK-only,
  frame-indeterminate; σ 3.5 → 4.3 cm).
- `build_scenes.load_episode_poses` sets episode confidence = minimum over
  initial objects, and `run_validation.py` excludes confidence ∉
  {medium, high}. With its only initial object now `low`, ep779 is excluded
  by the suite's own pre-existing rule.

Consequences: tool_insert denominator 17 → **16**; total variant jobs
485 → **480**; tool valid variants 84/85 (interim) → **79/80** (final; the
one defect is the same ep697 v3, see §4). Whether ep779's exclusion is the
intended reading of FP-H is a coordinator/verifier call — this runner
changed nothing to force either outcome and reports the run as it landed.

## 1. Interim record preserved

- `validation/validation_results_v11.json` (interim, previous runner) →
  copied to `validation/validation_results_v11_interim.json` **before** any
  regeneration.
- SHA-256 before copy = after copy =
  `aaa2940d69639c000ef3ad5a2e20a572e243472f1ccf24b9acd0c3ebcac56d68` —
  byte-identical.

## 2. Scene regeneration + determinism

`code/build_scenes.py` run twice from current (post-fix-pass) code + data
(K=5, seed 20261122):

- Both runs: **500 variant scenes built, 0 skipped** (600 XML files total
  including the 100 nominal copies; note the builder builds all 100 pose
  files including the 4 validation-excluded episodes).
- Aggregate SHA-256 over all 600 XMLs (`sha256sum *.xml | sha256sum`):
  - Run 1: `0e7eceb6a2e7f82fda0ad1e614b17285428629f529255d607062ff0611d1e06f`
  - Run 2: `0e7eceb6a2e7f82fda0ad1e614b17285428629f529255d607062ff0611d1e06f`
  - **DETERMINISM: MATCH (byte-identical).**
- This hash differs from the interim run's `48fcccce…` because the fix pass
  changed hash inputs (code + ep779/ep784 poses + hash coverage itself);
  determinism within the final landed state is exact.

## 3. Full final re-validation — headline table (Amendment 1 rule)

Episode REPRODUCED = ≥60% of VALID variants agree with the recorded outcome.
Secondary = nominal variant (v0) pass. Denominator **96** (see deviation
note above).

| Task | In | Reproduced (primary) | Mean variant agreement | Secondary pass (nominal) | Solver failures |
|---|---|---|---|---|---|
| block_stack | 20 | 0/20 | 0.00 | 4/20 | 0/100 (0%) |
| eye_drops_to_basket | 20 | 3/20 (ep1757, 1798, 1810) | 0.2825 | 0/20 | 0/100 (0%) |
| ring_insert | 20 | 0/20 | 0.01 | 7/20 | 0/100 (0%) |
| tool_insert | **16** | 0/16 | 0.00 | 0/16 | 0/80 (0%) |
| cable_clip | 20 | **20/20** | **1.00** | exempt (null) | 0/100 (0%) |

Total solver failures: **0/480 (0%)**. No task except cable meets the 80%
bar; cable meets it (bar_80pct_met = true) — now under correct scoring (§5).

### Three-way comparison (primary numbers): v1 → v1.1-interim → v1.1-final

| Task | v1 reproduced | v1.1-interim reproduced | v1.1-final reproduced | v1 mean agree | interim mean agree | final mean agree |
|---|---|---|---|---|---|---|
| block_stack | 0/20 | 0/20 | 0/20 | 0.00 | 0.00 | 0.00 |
| eye_drops_to_basket | 4/20 | 3/20 | 3/20 | 0.2775 | 0.2825 | 0.2825 |
| ring_insert | 1/20 | 0/20 | 0/20 | 0.06 | 0.01 | 0.01 |
| tool_insert | 0/17 | 0/17 | 0/16 † | 0.00 | 0.00 | 0.00 |
| cable_clip | 20/20 | 20/20 ‡ | **20/20** | 1.00 | 1.00 ‡ | **1.00** |

† Tool denominator change is solely the ep779 exclusion (§deviation note);
ep779's interim agreement was 0.0, so no reproduced/agreement number moves.
‡ Interim cable figures were scored under the V11-01 defect (clip_block
jittered in the scene but scored against its nominal pose) and were marked
TO BE SUPERSEDED by the interim runner. **The final cable row is the first
honest cable score under correct scoring** — fixture un-jittered (FP-A) and
scored from the sim body (replay.py `pos("clip_block")`). The honest score
is numerically identical to the defective one (20/20, mean agreement 1.00):
the V11-01 defect did not, in the end, inflate cable's headline — but that
is now established by a correct run rather than assumed.

All other tasks' final numbers are identical to interim, episode for
episode (same reproduced sets, same agreement means, same secondary
counts: block 4/20, eye 0/20, ring 7/20, tool 0).

## 4. Exclusion log (complete)

- Episode-level (from the final results file):
  - tool_insert ep630 — initial-pose confidence=low (pre-registered)
  - tool_insert ep871 — initial-pose confidence=low (pre-registered)
  - tool_insert ep876 — initial-pose confidence=low (pre-registered)
  - tool_insert ep779 — initial-pose confidence=low (per-object
    `{'tool': 'low'}`) — **new in the final run**, the mechanical result of
    the FP-H relabel (see deviation note).
- Variant-level build defects: **2 of 480** (interim: 2 of 485; v1: 12 of 485):
  - eye_drops_to_basket ep1808 v4 — "carton: displaced 0.091 m / z=0.017
    after pre-settle" (same as interim).
  - tool_insert ep697 v3 — "tool: displaced 0.035 m / z=0.005 after
    pre-settle" (same as interim).
- Valid variants: **478/480**. No run errors, no scene-build skips
  (0 skipped, both regens), no solver-failure exclusions (none occurred),
  no MuJoCo instability WARNINGs in the validation log (0).

## 5. Spot-checks

- **Cable ep1939, clip at nominal in the XMLs — PASS.** `clip_block` body
  position is the nominal `(0.27226967686214654, −0.08705115372165617, 0)`
  in **every** variant v0–v4. Variant draw logs confirm jitter keys for
  k≥1 are exactly `["cable"]` (clip_block no longer jittered, FP-A).
- **Tool ep784, built pose = fix-pass grounding pose — PASS.** Pose file
  carries the FP-G grounding pose `[0.2094, −0.0365, 0.005, null]`
  (px (363, 306) → calib_v2 homography). The regenerated nominal XML
  places the tool body at `(0.144571, 0.0760123, 0.005)`, which equals the
  suite's standard FK-affine frame correction of the grounding pose
  exactly: `correct_xy([0.2094, −0.0365]) = [0.14457111, 0.07601226]`
  (same correction path the interim runner verified on ep697).

## 6. Integrity checks

- **param_hash: `38e5a9e7dd1d0330`** stamped in the final results —
  **MATCHES** the fix-pass value exactly (FP-C extended coverage). The same
  hash is stamped in the regenerated scene variant logs (spot-checked
  ep784). No regeneration-induced drift: the results' own scene rebuilds
  did not change any hash-covered file.
- **v1 record preserved:** `validation/validation_results.json` SHA-256
  before = after restore =
  `396378dca6e545dc4243540e188b6363fdf88084ba70c6dcbfc66e4634d7c5b5`
  (full value; the `396378dca6…` prefix in the brief matches, and the full
  value matches the previous runner's recorded value). The file was only
  touched by this runner's own backup/restore cycle around
  `run_validation.py`'s hardcoded output path (backed up → fresh output
  copied to `validation_results_v11.json` → v1 restored byte-identical).
  Its companions `validation_results_run1.json` / `run2.json` were not
  touched.
- **Interim record preserved:** §1 (SHA-256 `aaa2940d…`, byte-identical).
- Suite/seed in final results: twin-suite-v1.1, seed 20261122, K=5.

## 7. Wall time

- Scene regeneration: ~3 s per run (×2).
- Validation: **406 s** for 480 variants at 2000 Hz with 4 workers.
- Total job wall (preserve + 2 regens + validation + restore):
  **412 s (~6.9 min)**, 19:25:16–19:32:08 CST.

## 8. Runner's notes

1. The only delta between interim and final outside cable scoring is the
   ep779 exclusion (§deviation note) — flagged here prominently rather
   than silently absorbed into a "97" the run did not produce.
2. Cable's honest 20/20 under correct scoring closes the V11-01
   supersession the interim report left open; the interim cable agreement
   figures should now be retired in favour of these.
3. Certification is the verifier's / coordinator's call, not this
   runner's.

— Final Re-Validation Runner (RUN2), 2026-11-22 19:32 CST
