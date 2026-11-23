# Twin-Suite v1.1 — DEFINITIVE Re-Validation Report (RUN3, 2026-11-22)

Run: 2026-11-22 19:37:10–19:44:01 CST (DGX clock; total wall 411 s, validation
wall 404 s), CPU-only, `nice -n 10`, 4 workers, no GPU use (GPU was at 96%
from the live DGX training workload at launch check; it was not disturbed).
The suite was run EXACTLY as frozen at param_hash `a4eb3f8c3611894d`
(docs/microfix-v11.md, changelog MF-1..MF-3). **This runner changed NOTHING**
— no code, pose, parameter, or criteria edits. Anomalies logged as RUN3-1..
RUN3-3 in validation/param-changelog.md.

Runner script: `validation/run_revalidation_v11_run3.sh`; logs:
`logs/revalidation-v11-run3-{status,regen1,regen2,validation}.log`
(status file: `logs/revalidation-v11-run3-status.txt`).
Output: `validation/validation_results_v11.json` (RUN3, SHA-256
`4b526d529d5bdb5e6093e0ffbbffcabe86d697e4fce0a5b543ca5ee122df5cf9`).

## 1. RUN2 record preserved

- `validation/validation_results_v11.json` (RUN2, stamped param_hash
  `38e5a9e7dd1d0330`, SHA-256
  `80fc033d804b6229eb80eeb32881c8e2dea6862a591742145319c17a8976df35`)
  copied to `validation/validation_results_v11_run2.json` BEFORE any
  regeneration — byte-identical (SHA-256 verified before and after copy).
- v1 original `validation/validation_results.json` verified at SHA-256
  `396378dca6e545dc4243540e188b6363fdf88084ba70c6dcbfc66e4634d7c5b5`
  before the run and restored + re-verified byte-identical after the run
  (run_validation.py hardcodes that filename; the fresh RUN3 output was
  copied to `validation_results_v11.json` first, as prior runners did).

## 2. Regeneration + determinism (frozen code+data, K=5, seed 20261122)

`code/build_scenes.py` run twice; both runs: 500 variant scenes built,
0 skipped (600 XML files total including the 100 nominal copies).

- Aggregate SHA-256 over all 600 XMLs (`sha256sum *.xml | sha256sum`):
  - Regen 1: `e254e311ede3e975639019cb4edc0d5e1d9e9c9a82b2b92fff0c134cff48da2e`
  - Regen 2: `e254e311ede3e975639019cb4edc0d5e1d9e9c9a82b2b92fff0c134cff48da2e`
  - **DETERMINISM: MATCH (byte-identical).**
- Note (changelog RUN3-2): this aggregate differs from RUN2's recorded
  `0e7eceb6a2e7f82fda0ad1e614b17285428629f529255d607062ff0611d1e06f` ONLY
  because every generated XML embeds `param_hash=<hash>` in its header
  comment, and the frozen hash changed `38e5a9e7…` → `a4eb3f8c…` under
  MF-1..MF-3. Proof: replacing the stamp `a4eb3f8c3611894d` →
  `38e5a9e7dd1d0330` in all 600 current XMLs and re-running the identical
  sha256sum pipeline reproduces RUN2's aggregate `0e7eceb6…` EXACTLY.
  The scenes are byte-identical modulo the stamp.

## 3. Full validation — headline table (RUN3)

Episode REPRODUCED = ≥60% of VALID variants agree with the recorded outcome.
Secondary = nominal variant (v0) position criterion. Denominator **96**.

| Task | In | Reproduced (primary) | Mean variant agreement | Secondary pass (nominal) | Solver failures |
|---|---|---|---|---|---|
| block_stack | 20 | 0/20 | 0.00 | 4/20 | 0/100 (0%) |
| eye_drops_to_basket | 20 | 3/20 | 0.2825 | 0/20 | 0/100 (0%) |
| ring_insert | 20 | 0/20 | 0.01 | 7/20 | 0/100 (0%) |
| tool_insert | 16 | 0/16 | 0.00 | 0/16 | 0/80 (0%) |
| cable_clip | 20 | 20/20 | 1.00 | exempt (null; recorded 0) | 0/100 (0%) |

- Fresh output stamped param_hash == `a4eb3f8c3611894d` ✔, suite_version
  `twin-suite-v1.1`, seed 20261122.
- Total solver failures: **0/480 variant jobs (0%)**. Valid variants
  478/480; the 2 non-valid variants are pre-settle build defects, identical
  to RUN2: eye_drops_to_basket ep1808 v4 (carton displaced 0.091 m) and
  tool_insert ep697 v3 (tool displaced 0.035 m).
- Only cable_clip meets the 80% bar (bar_80pct_met), as in RUN2 and v1.

## 4. Exclusion log (denominator 96, Amendment 3 ruling)

Exactly 4 exclusions, all tool_insert, all via the registered
low-confidence rule (episode confidence = minimum over initial objects ∉
{medium, high}):

| Episode | Reason (as recorded in results) |
|---|---|
| ep630 | initial-pose confidence=low (per-object: {tool: low}) |
| ep779 | initial-pose confidence=low (per-object: {tool: low}) — OUT per Amendment 3 / MF-3 |
| ep871 | initial-pose confidence=low (per-object: {tool: low}) |
| ep876 | initial-pose confidence=low (per-object: {tool: low}) |

Per-task denominators: block 20, eye 20, ring 20, tool 16, cable 20 = 96 ✔.

## 5. RUN2 vs RUN3 identity check — PASS, zero differences

Compared `validation_results_v11_run2.json` (RUN2, stamp `38e5a9e7…`)
against the fresh RUN3 file (stamp `a4eb3f8c…`):

- Task summaries: **identical** (every field, all 5 tasks).
- Per-episode outcome fields (recorded_success, valid/agreeing/certified
  variants, build_defects, variant_agreement_frac, reproduced, fragile,
  nominal secondary pass, nominal sim success, nominal planar error,
  solver failures): **0 differences over all 96 episodes**.
- Exclusion lists: identical.
- **Full deep-JSON equality excluding the `param_hash` stamp: TRUE** —
  every number in the file, including all per-variant detail floats, is
  identical. The micro-fixes changed no scene and no score, exactly as
  the freeze predicted. No red flags.

## 6. Primary row across versions: v1 → RUN2 → RUN3

| Task | v1 reproduced (mean agr.) | RUN2 reproduced (mean agr.) | RUN3 reproduced (mean agr.) |
|---|---|---|---|
| block_stack | 0/20 (0.00) | 0/20 (0.00) | 0/20 (0.00) |
| eye_drops_to_basket | 4/20 (0.2775) | 3/20 (0.2825) | 3/20 (0.2825) |
| ring_insert | 1/20 (0.06) | 0/20 (0.01) | 0/20 (0.01) |
| tool_insert | 0/17 (0.00) | 0/16 (0.00) | 0/16 (0.00) |
| cable_clip | 20/20 (1.00) | 20/20 (1.00) | 20/20 (1.00) |

(v1: denominator 97, stamp `5ef4c692a314e87b`, pre-v1.1 criteria — the
v1→v1.1 deltas are the fix-pass changes, not run noise; RUN2→RUN3 is the
pure frozen-suite reproducibility check and is exact.)

## 7. Verdict

Twin-suite v1.1 is FROZEN and REPRODUCIBLE: deterministic regeneration
(byte-identical XMLs across regens), a full validation at the frozen hash
`a4eb3f8c3611894d` over the ruled denominator of 96, and a RUN2↔RUN3
identity check with zero numeric differences anywhere. Headline stands:
block 0/20, eye 3/20, ring 0/20, tool 0/16, cable 20/20 — no task except
cable_clip meets the 80% certification bar.
