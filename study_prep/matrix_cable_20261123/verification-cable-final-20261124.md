# Independent Verification — zk0 Study 001 Closed-Loop CABLE Matrix, Final Frozen State

- Verifier: independent subagent, separate context (Muse). Builder/shepherd numbers not trusted; all counts recomputed from raw record JSONs with a verifier-written script (`/tmp/verify_cable.py` on the DGX, run 2026-11-24).
- Scope: `cells.cable_clip` of `/home/openclaw/muse-work/zk0-study001-matrix-20261123/aggregate.json` (generated 2026-11-24T06:32:27-06:00; file mtime 2026-11-24 06:32:27 CT, verified).
- Sources recomputed:
  - phase6: `/home/openclaw/muse-work/zk0-study001-phase6-20261123/out/records/*.json`
  - matrix: `/home/openclaw/muse-work/zk0-study001-matrix-20261123/cable/out/records/*.json`
- Grok was not used for anything (founder ruling 2026-11-23, trust suspended). No records were modified; no rollouts were run.
- Counting rule applied independently (matching the frozen aggregator's stated rule, re-derived from the records): a record is *excluded* iff top-level `solver_failure` is true; `total` = non-excluded records; `pass` = non-excluded records with `sim_success` true.

## Overall verdict: CERTIFY-WITH-FINDINGS

Zero numeric discrepancies against the aggregate. One minor provenance finding on Item 4 (records do not embed the param_hash; identity is established through the harness and scene stamps, not per-record). The matrix is a frozen *partial* set: 13 of the 16 checkpoints have cable records; `sequential`, `spec_exp_ring_insert`, and `spec_exp_tool_insert` have zero records in both sources and are correctly absent from the aggregate's cable cells.

Headline recomputed totals: phase6 416 raw / 416 counted / 2 pass / 0 excluded; matrix 701 raw / 700 counted / 0 pass / 1 excluded; combined phase6+matrix 1,117 raw / 1,115 counted / **2 pass** / 1 excluded.

---

## Item 1 — Per-checkpoint recount vs aggregate `cells.cable_clip` — CERTIFY

Verifier recount (pass / total / excluded), cell-by-cell identical to the aggregate in every cell:

| Checkpoint | phase6 (mine) | phase6 (agg) | matrix (mine) | matrix (agg) | combined_phase6_matrix (mine) | combined (agg) |
|---|---|---|---|---|---|---|
| base | 0/96/0 | 0/96/0 | — | — | 0/96/0 | 0/96/0 |
| cexp | 1/97/0 | 1/97/0 | — | — | 1/97/0 | 1/97/0 |
| fl_r30_3cam | 0/100/0 | 0/100/0 | — | — | 0/100/0 | 0/100/0 |
| fl_r20_3cam | — | — | 0/100/0 | 0/100/0 | 0/100/0 | 0/100/0 |
| fl_r10_3cam | — | — | 0/100/0 | 0/100/0 | 0/100/0 | 0/100/0 |
| spec_block_stack | — | — | 0/100/0 | 0/100/0 | 0/100/0 | 0/100/0 |
| spec_eye_drops | — | — | 0/99/1 | 0/99/1 | 0/99/1 | 0/99/1 |
| spec_ring_insert | — | — | 0/100/0 | 0/100/0 | 0/100/0 | 0/100/0 |
| spec_tool_insert | — | — | 0/100/0 | 0/100/0 | 0/100/0 | 0/100/0 |
| spec_cable_clip | 1/97/0 | 1/97/0 | — | — | 1/97/0 | 1/97/0 |
| spec_exp_block_stack | — | — | 0/50/0 | 0/50/0 | 0/50/0 | 0/50/0 |
| spec_exp_eye_drops | — | — | 0/51/0 | 0/51/0 | 0/51/0 | 0/51/0 |
| spec_exp_cable_clip | 0/26/0 | 0/26/0 | — | — | 0/26/0 | 0/26/0 |
| sequential | 0 records | absent | 0 records | absent | — | absent |
| spec_exp_ring_insert | 0 records | absent | 0 records | absent | — | absent |
| spec_exp_tool_insert | 0 records | absent | 0 records | absent | — | absent |

`secondary` is 0 in every cable cell in both my recount (`secondary_pass` true count) and the aggregate — consistent (cable is exempt from the secondary criterion in the frozen scorer; records carry `secondary_pass: null`).

Part B L1 separation (out-of-scope cells, separation only): **confirmed.** The aggregate keeps partB-L1 cable in its own source bucket (base 0/35, cexp 0/35, fl_r30_3cam 0/35) and `combined_phase6_matrix` never includes it — e.g. base combined is 96, not 96+35=131; cexp 97, not 132; fl_r30_3cam 100, not 135. The aggregate's `source_policy` states the same rule.

## Item 2 — The excluded spec_eye_drops (matrix) record — CERTIFY

Exactly one excluded record exists in the entire two-source set (1,117 records scanned):

- File: `spec_eye_drops__cable_clip__ep1903_v1.json` (matrix source; checkpoint `spec_eye_drops`, episode 1903, variant 1)
- Exclusion reason: **`solver_failure`** — top-level `solver_failure: true` and `checks.solver_failure: true`. It is **not** a build defect: `build_defect: null`.
- Supporting fields: `sim_success: false`, `checks.cable_seated_in_clip_closed: false`, `checks.max_seg_radial_m: 1.11` (a cable segment ended far outside the frozen workspace envelope — the Amendment 1 A2 solver-failure condition in the scorer), `checks.nearest_seg_to_block_xy_m: 1.1278`.
- Aggregate accounting matches: raw files for spec_eye_drops (matrix) = 100, counted total = 99, excluded = 1.

## Item 3 — The two pass records — CERTIFY

My recount finds exactly two `sim_success: true` records in both sources combined — the same two units implied by the aggregate's two pass cells (cexp phase6 = 1, spec_cable_clip phase6 = 1; all other cells 0):

1. `cexp__cable_clip__ep2020_v2.json` — checkpoint **cexp**, episode **2020**, variant **2** (phase6)
   - checks: `cable_seated_in_clip_closed: true`, `solver_failure: false` (checks and top-level), `nearest_seg_to_block_xy_m: 0.006`, `max_seg_radial_m: 0.38`
2. `spec_cable_clip__cable_clip__ep2020_v2.json` — checkpoint **spec_cable_clip**, episode **2020**, variant **2** (phase6)
   - checks: `cable_seated_in_clip_closed: true`, `solver_failure: false` (checks and top-level), `nearest_seg_to_block_xy_m: 0.0051`, `max_seg_radial_m: 0.488`

Frozen v1.1 criterion (from the frozen scorer `replay.evaluate` in the twin suite, read at source): cable SUCCESS iff a segment is seated in the clip with the clip closed over it, mechanically registered as *a segment inside the channel volume between the jaws AND in contact with the `clip_block` body at settle* (`in_channel and "clip_block" in contacts_with(...)` → `seated`), and `success = seated and not solver_failure`. Both pass records satisfy exactly this as recorded: the `cable_seated_in_clip_closed` boolean (which encodes both the in-channel and clip_block-contact clauses) is true, and there is no solver failure at either level. Both records are internally consistent (`sim_success` = seated ∧ ¬solver_failure). Note: `nearest_seg_to_block_xy_m` is a reported diagnostic in the frozen scorer, not a criterion term; the contact clause is carried by the seated boolean itself.

## Item 4 — Scene-set identity (twin-suite v1.1, param_hash a4eb3f8c3611894d) — CERTIFY-WITH-FINDINGS

Confirmed:
- All **1,117 / 1,117** records in both sources carry `suite_version: "twin-suite-v1.1"` and `task: "cable_clip"` (full-set scan, no exceptions). Both sources cover the identical episode set (the same 20 held-out cable episodes: 1896, 1903, 1905, 1939, 1959, 1960, 1968, 1969, 2020, 2032, 2042, 2046, 2059, 2068, 2085, 2092, 2101, 2125, 2130, 2146).
- Harness scene source: `phase6_worker.py` inserts `/home/openclaw/muse-work/zk0-study001-twin-20261122/code` on `sys.path`, imports `twin_params` / `build_scenes` / `replay` from the frozen suite, and loads scenes as `P.TWIN / "scenes" / cable_clip__ep{epi}_v{variant}.xml` where `P.TWIN = /home/openclaw/muse-work/zk0-study001-twin-20261122`. The matrix driver (`matrix_cable_driver.py`) invokes that same Phase 6 worker unchanged — one harness, one scene set, both sources.
- Hash: I recomputed the suite's own `param_table_hash()` live against the current twin directory — it returns **`a4eb3f8c3611894d`**, identical to the frozen value. The same hash is stamped in `validation/validation_results_v11.json` (`suite_version: twin-suite-v1.1`, `param_hash: a4eb3f8c3611894d`) and in **120 / 120** cable scene XML headers under `scenes/cable_clip__ep*.xml`.

Finding (minor, provenance only): the record JSONs themselves do **not** carry a `param_hash` field (full key scan of all 1,117 records: no such key). Per-record scene identity therefore rests on `suite_version` plus the harness/scene stamps above, not on a per-record hash. This does not affect any count, but a future harness revision should embed the hash per record.

## Item 5 — Phase 6 frozen set — CERTIFY

Exactly **416** record files. Per-checkpoint recount matches the brief exactly:

| Checkpoint | Records | Pass |
|---|---|---|
| base | 96 | 0 |
| cexp | 97 | 1 |
| fl_r30_3cam | 100 | 0 |
| spec_cable_clip | 97 | 1 |
| spec_exp_cable_clip | 26 | 0 |
| sequential | 0 | 0 |
| **Total** | **416** | **2** |

0 excluded in Phase 6.

## Item 6 — Stand-down integrity — CERTIFY

- (a) Process sweep (`ps aux` and `ps -ef`, patterns `retry_launch`, `matrix_cable_driver`, `guard-cable2`, `phase6_worker`, `policy_server`): **0 matching processes.**
- (b) Preservation: **0 files** in either records directory have mtime after 2026-11-24 06:33 CT. Latest mtimes: phase6 — 2026-11-23 14:19:00 CT; matrix — 2026-11-24 06:31:10 CT (consistent with the 06:31 CT stand-down ruling, before the 06:32 CT aggregate generation).
- (c) JSON validity: **0 invalid** — all 1,117 record JSONs (416 + 701) parse cleanly.

## Item 7 — Gemini agy CLI availability on the DGX — REPORTED (present)

- `which agy` (with `~/.local/bin` on PATH): `/home/openclaw/.local/bin/agy` (257,774,256 bytes, dated Nov 23 22:18).
- Version (bounded check, `agy --version`): **1.5.16**.
- No installs performed. agy was not otherwise used in this verification; the recount above is my own script's.

---

## Discrepancy list

None numeric. Every aggregate cable cell (per-source and combined) reproduces exactly from the raw records. The only finding is Item 4's minor provenance note (no per-record `param_hash` field), which changes no result.

## Verdict summary

- Item 1: CERTIFY
- Item 2: CERTIFY
- Item 3: CERTIFY
- Item 4: CERTIFY-WITH-FINDINGS (minor provenance note only)
- Item 5: CERTIFY
- Item 6: CERTIFY
- Item 7: Reported — agy present, v1.5.16
- **Overall: CERTIFY-WITH-FINDINGS** — the frozen cable matrix aggregate (2 passes / 1,115 counted rollouts across phase6+matrix; 1 solver-failure exclusion) is accurate as generated 2026-11-24 06:32 CT.
