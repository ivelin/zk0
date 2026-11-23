# Study 001 Twin-Replay — Code Delta-Verification, twin-suite v1.1 (V-Code Delta)

Verifier: V-Code Delta, fresh independent context (did not write the fix pass).
Date: 2026-11-22 (~19:24+ CST, DGX clock). Scope: CODE class only — verification
of the fix pass that followed `docs/verification-v11-code.md` (NOT-CERTIFIED,
blocking V11-01, plus V11-02/03/04 and minors). Claims in
`docs/fixpass-v11.md` were treated as claims to check, never as evidence.
All evidence below was re-derived from the raw files / live probes on the DGX
at audit time. CPU-only, `nice -n 10`, DGX training undisturbed. No full
re-validation was run by this verifier (a separate runner owns that); this
certifies code, not results.

## VERDICT: CERTIFIED-WITH-FINDINGS (code class, twin-suite v1.1 final)

The blocking defect V11-01 is closed by BOTH of the remediation routes the
original V-Code report offered (clip no longer jittered AND evaluator reads
the clip from the sim body), and I reproduced the fix-pass's own functional
proof numbers exactly with an independent probe. V11-02, V11-03 (with two
minor latent residuals, D-01/D-02) and V11-04 are closed. All listed minors
are closed. Three non-blocking findings remain (D-01, D-02, D-03), none of
which changes any scene or score produced by the current tree. The pose-data
repairs (ep784/ep779/ep697) verify as data, with one provenance trap in
ep784's nested label recorded as D-03.

---

## Check 1 — V11-01 closure (BLOCKING in v1.1 V-Code): CLOSED

Code:
- `code/build_scenes.py:145` — `make_draws` cable_clip jittered bodies are
  now `["cable"]` only; `clip_block` removed, with the FP-A comment citing
  the ring peg+board principle (changelog #12).
- `code/build_scenes.py:565-567` — `build()` still calls `jpose("clip_block",
  cbp)`, but `jpose` (build_scenes.py:431-435) is a no-op when the body has
  no jitter entry, and no draw ever contains `clip_block` now — defence in
  depth, not a residual path.
- `code/replay.py:462-475` — `evaluate()` cable branch reads the clip pose
  from the SIM BODY via `pos("clip_block")` (mirrors ring's peg read),
  falling back to pose-file/fixtures only if the body is absent.

Live draws (independent, `nice -n 10`, this audit):
- cable_clip ep1939, all 5 variants: jitter keys for k≥1 are exactly
  `["cable"]`; clip_block XML position is the nominal
  (0.27226967686214654, −0.08705115372165617, 0) in EVERY variant v0–v4.
  Repeated on ep1896: clip at its nominal in all 5 variants, keys `["cable"]`.
- All-task scan (block ep411, eye ep1609, ring ep7, tool ep622, cable
  ep1939): jitter keys are manipulated objects only (cube_a/cube_b, carton,
  ring, tool, cable). `grep fixture_jitter code/` = 0 hits. No remaining
  fixture/reference jitter anywhere, any task.

Functional probe (independent): loaded the landed
`scenes/cable_clip__ep1939_v0.xml` in the sim venv, called `evaluate()` at
initial state: `nearest_seg_to_block_xy_m` = 0.0993. Copied the XML with ONLY
the clip_block body shifted +0.10 m in x (pose file untouched): the same
quantity moved to 0.1736. This reproduces the fix-pass's claimed values
(0.0993 → 0.1736) exactly and proves the evaluator tracks the sim body, not
the pose file. Probe temp file removed afterwards.

## Check 2 — V11-02 closure: CLOSED, no value drift

- `code/twin_params.py:155-165` — named constants with documented basis
  (built jaw geometry: clip_jaw_a/b centres y = ±(CLIP_CHANNEL_W/2 + 0.004),
  half-extents (0.012, 0.004, 0.006) m; x = jaw half-length, y reaches the
  jaw inner faces' outer edge, z = 2× jaw half-height):
  `CABLE_CHAN_X_HALF_M = 0.012`, `CABLE_CHAN_Y_EXTRA_M = 0.002`,
  `CABLE_CHAN_Z_HALF_M = 0.012`.
- `code/replay.py:490-491` — the channel test consumes exactly these
  constants (`P.CABLE_CHAN_X_HALF_M`, `CLIP_CHANNEL_W/2 +
  P.CABLE_CHAN_Y_EXTRA_M`, `P.CABLE_CHAN_Z_HALF_M`); no channel literals
  remain in replay.py.
- Values are IDENTICAL to the pre-fix literals quoted in
  verification-v11-code.md (V11-02: |x| ≤ 0.012, |y| ≤ CLIP_CHANNEL_W/2 +
  0.002, |z − chan_z| ≤ 0.012). No drift. FP-2 (param-changelog.md:43)
  registers them, curing the "unregistered tolerance" defect.

## Check 3 — V11-03 closure: CLOSED with two minor latent residuals (D-01, D-02)

- Recomputed `param_table_hash()` myself from the current tree (live
  import): **`38e5a9e7dd1d0330`** — equals the FP-C claim exactly.
- Coverage in code (`code/build_scenes.py:32-64`) matches the FP-C claim:
  HASH_FILES = twin_params, build_scenes, replay, run_validation,
  calibration, finalize, **patch_finalize, patch2**,
  docs/physics-parameters.md; HASH_GLOBS = `code/builder2_stage*.py`
  (resolves sorted to stage1, 1b, 1c, 1d, 2, 2b — all six present);
  HASH_DATA_FILES = scenes/calibration_fk_affine.json,
  scenes/calibration_v2.json, scenes/detections_px.json; then a sorted
  manifest (name + contents) of the pose JSONs. I counted the manifest:
  exactly **100** files (`scenes/*__ep*.json` excluding `*_variants.json`).
- Missing-file no-crash claim tested on a COPY (live files never renamed):
  copy-tree hash = `38e5a9e7dd1d0330`; with `code/patch2.py` renamed away in
  the copy, hash computed without crash = `ec2675e21523a946` — the exact
  intermediate value the fix-pass quoted; restored → back to
  `38e5a9e7dd1d0330`. (Silent-skip semantics: a missing file changes the
  hash rather than crashing — as claimed.)
- Landed scene headers corroborate: `scenes/cable_clip__ep1939_v0.xml`
  header carries `param_hash=38e5a9e7dd1d0330`.

Remaining unscored inputs that can, in principle, change a scene/score:

- **D-01 (MINOR, latent) — `scenes/fixtures.json` is not hashed.**
  `load_fixtures` (build_scenes.py:366-390) merges this global file under
  the per-episode fixtures. I verified it is currently INERT: every
  eye episode's pose JSON carries platform+basket, every tool episode
  tray+target_slot, every ring episode board+peg (the operative names),
  and the global block/cable entries are empty — so today the hashed pose
  JSONs fully shadow it. But an edit to fixtures.json that added a fixture
  name an episode lacks would change scenes with no hash change. One-line
  fix: add it to HASH_DATA_FILES.
- **D-02 (MINOR, noted) — external ground-truth / vendored inputs are not
  hashed:** the arm model `P.ARM_XML` + its meshes (twin_params.py:12),
  the recorded trajectories `P.RECORDS/*.npz` (replay.py:72), `P.SPLIT`
  (run_validation.py:39), and the dataset metadata file FPS is read from
  (twin_params.py:24-28; the v1.1 V-Code report already flagged this one
  as low-risk). These are the dataset/vendored assets the suite replays,
  not suite-generated parameters, and FP-3's corrected coverage statement
  lists exactly what IS covered and no longer overclaims — so this is a
  scope note, not a false claim. A future suite freeze should pin these
  by reference/hash in the run manifest.

No other unscored suite-internal input was found: all other
`scenes/*.json` files (calibration_v1, detections4/5_px, fk_tiepoints,
mat_*, slot_px, etc.) are upstream pose-pipeline artifacts not consumed by
the build/replay path (which reads only the three hashed data files, the
pose JSONs, and fixtures.json per D-01).

## Check 4 — V11-04 closure: CLOSED

- FP-13 exists (`validation/param-changelog.md:54`, stamped 2026-11-22
  19:22 CST): it corrects B2-5 (stamped 19:12, line 40) and COORD-1
  (stamped 19:14, line 41) as post-dated by a few minutes, appended by
  19:10 DGX time per file mtime, originals left as-written. That
  description is accurate against verification-v11-code.md's V11-04
  (changelog mtime 19:08:34 at that audit; builder2 report mtime 19:07:47)
  and against the B2-5/COORD-1 texts I re-read — the post-dating and its
  size (minutes) are correctly characterised.
- FP-1…FP-13 all stamped 19:22 CST: PLAUSIBLE. Every file the fix pass
  touched (code/twin_params.py, build_scenes.py, replay.py; the three
  scene JSONs; docs/asset-inventory.md; validation/param-changelog.md)
  carries mtime 2026-11-22 19:22:28.5x CST, and docs/fixpass-v11.md
  19:24:04 — the stamps are not future-dated relative to their files
  (the V11-04 defect), and the changelog's own mtime equals its last
  entries' stamp. Observation (not a finding): the mtimes cluster within
  ~25 ms across all touched files, i.e. the pass landed as one batch
  write; "a fresh `date` read immediately before its append" is
  consistent with a scripted batch executed inside that minute, and
  nothing in the timestamps contradicts it.

## Check 5 — Minors: ALL CLOSED

- SECONDARY_THRESH_M consumed: `code/replay.py:507,510` —
  `thr = P.SECONDARY_THRESH_M[task]` for block/eye/tool and for ring;
  cable remains exempt (None). Grep confirms no 0.030/0.020 literals remain
  in evaluate()'s secondary block.
- Stale rate text: `code/replay.py:5` now reads "(2000 Hz)"; grep finds no
  "(500 Hz)" in replay.py. (Historical 500 Hz mentions in twin_params.py /
  changelog are accurate as history — FP-5 notes this correctly.)
- F-08 correction: FP-6 (changelog:47) accurately restates the block
  final-pose rule — candidate nearest the base cube's initial pixel when
  that detection exists; mean/<80 px only as fallback — matching both
  verification-code.md F-08 and the code I re-read in
  `load_episode_poses` (build_scenes.py, block finals block).
- F-07 disclosure: FP-7 (changelog:48) discloses the hand-pick tables with
  counts I spot-verified against the source and against
  verification-code.md F-07: finalize.py HAND_BLOCK_FIRST (ep411/530/
  539×2/546), EYE_FIRST_GROUND (11), EYE_FINAL_GROUND (18), EYE_FINAL_LOW
  (2: ep1618, ep1760), TOOL_GROUND (10), TOOL_LOW (3), confidence overrides
  (ep411 cube_green, ep513 cube_purple), fixture pixels; patch_finalize.py
  CABLE_BLOCK_GROUND (all 20 cable episodes) + per-episode ring peg/board
  derivation; patch2.py ep1780/1783/1814. Accurate.
- Carried minor (not in the fix brief, disclosed in fixpass "Not done"):
  the dead JITTER tier table in twin_params.py remains. It is consumed by
  nothing (sampler uses per-pose sigmas) and cannot affect scores; recorded
  here so the carry-over is explicit.

## Check 6 — Pose-data fixes (verified as data): VALUES CORRECT; one provenance trap (D-03)

- **ep784** (`scenes/tool_insert__ep784.json`): operative pose is the
  fix-pass grounding — `objects.tool.pose_xyz_yaw` =
  [0.2094, −0.0365, 0.005, null], method = visual grounding (FP-G pick
  (363, 306) px), σ_xy 3.5 cm, confidence medium. The superseded FK pose
  [0.1808, 0.0192, 0.005, null] is preserved both as
  `pose_v11_fk_anchor_superseded` and inside
  `v11_repair.fixpass_fp_g` (chosen_source "visual_grounding_fixpass",
  chosen_px [363, 306], verifier cross-check picks and distances 23.0 /
  26.9 px recorded), and `fk_anchor.note` now truthfully states the anchor
  DID supply the v1.1 repair pose and is now cross-check only.
  **D-03 (MODERATE bookkeeping, no score effect) — the provenance layering
  is a trap for a reader of the nested field:** the top-level
  `v11_repair.chosen_source` still reads `"fk_anchor_only"` (and
  `v11_repair.rule_v2.chosen_source` likewise). That is defensible as
  history (it records the rule-v2 outcome, and the fix-pass disclosed
  leaving it), and the TOP-LEVEL method/pose fields are unambiguous — but
  `v11_repair.chosen_source` is exactly the field FP-I treated as the
  operative label when it corrected ep697, so a future reader or script
  consuming that field for ep784 will conclude the operative pose is
  FK-only when it is in fact the visual-grounding pose. Recommended
  one-line cure: set `v11_repair.chosen_source` to the fixpass source (or
  add an explicit `operative_source` pointer) and keep the rule-v2 value
  only inside `rule_v2`. This does not block code certification: no code
  path reads `chosen_source`, and the pose value itself is correct.
- **ep779** (`scenes/tool_insert__ep779.json`): pose VALUE unchanged
  [0.162, 0.0201, 0.005, null] — identical to Builder 2's repair value in
  docs/builder2-v11-poses.md:33. Confidence now **low**, σ_xy **4.3** cm,
  method text truthfully states FK-only, visual status **indeterminate**,
  not frame-verified; `fk_anchor.note` corrected (anchor DOES supply the
  pose); `v11_repair.fixpass_fp_h` records value-unchanged + basis. Here
  `v11_repair.chosen_source = "fk_anchor_only"` is accurate and operative.
  Clean.
- **ep697** (`scenes/tool_insert__ep697.json`): `v11_repair.chosen_source`
  now `"grounding_consensus"`, consistent with `rule_v2` in the same file
  (consensus mean (242.0, 346.5) px) and the top-level method string. Pose
  values identical to the pre-fix v1.1 repair: [0.1682, −0.0981, 0.005,
  null], matching docs/builder2-v11-poses.md:32 and
  validation/builder2_stage2/2b.json (the v1 backup under
  validation/builder2_backup_v1 predates the v1.1 repair and is not the
  right comparator). Clean.

## Check 7 — No new damage: PASS

- No per-episode/per-variant special-casing introduced: the changed code
  branches by task/object class only (make_draws bodies table keyed by
  task; evaluate() cable branch keyed by task). Grep of the changed code
  files for episode numbers finds them only in docstring proof narratives
  (pre-existing). The per-episode edits are DATA repairs (the three scene
  JSONs), each disclosed in FP-8/FP-9/FP-10 — pose data, not scoring code.
- Determinism at code level unchanged: `make_draws` still seeds per
  variant via `SeedSequence([VARIANT_SEED, crc32(task__ep), k])`
  (build_scenes.py:142), VARIANT_SEED = 20261122 (twin_params.py:135);
  the fix pass touched neither the seeding nor the RNG consumption order
  for existing bodies (removing clip_block removes a consumer that was
  last in the cable bodies list order — cable's own draws are the first
  consumer and are unaffected in sequence; regardless, draws are
  re-derived per variant from the seed, so regeneration is deterministic
  by construction).

---

## Findings carried forward (all non-blocking)

| ID | Severity | Finding |
|---|---|---|
| D-01 | Minor (latent) | `scenes/fixtures.json` not covered by `param_table_hash()`; currently inert because hashed per-episode pose JSONs shadow every operative fixture name (verified across all 100 episodes), but a future global-fixture edit could change scenes silently. Add to HASH_DATA_FILES. |
| D-02 | Minor (scope note) | External inputs not hashed: arm model + meshes, recorded trajectory .npz files, split.json, dataset info.json (fps source). FP-3's corrected coverage claim is accurate as listed, so no overclaim remains; pin these by reference at suite freeze. |
| D-03 | Moderate (bookkeeping only) | ep784 nested `v11_repair.chosen_source = "fk_anchor_only"` contradicts the operative fixpass grounding source recorded in `v11_repair.fixpass_fp_g`; top-level method/pose are correct and no code reads the field, but the field is a provenance trap of the same kind FP-I fixed for ep697. Relabel or add an operative-source pointer. |

## Per-check summary

1. V11-01: **CLOSED** (draws + build + evaluator all verified live; probe
   reproduces 0.0993 → 0.1736 exactly; no fixture jitter remains anywhere).
2. V11-02: **CLOSED** (named constants in twin_params.py:155-165, consumed
   at replay.py:490-491, values identical to pre-fix literals).
3. V11-03: **CLOSED** (hash recomputed = `38e5a9e7dd1d0330`; coverage
   matches FP-C incl. patch files, builder2_stage glob, calibration data,
   100-pose manifest; missing-file test on a copy reproduces the claimed
   intermediate hash `ec2675e21523a946` with no crash). Residuals D-01/D-02.
4. V11-04: **CLOSED** (FP-13 present and accurate; FP-* 19:22 stamps
   plausible vs mtimes — equal, not future-dated).
5. Minors: **CLOSED** (SECONDARY_THRESH_M consumed, no literals; "(500 Hz)"
   gone; FP-6/FP-7 accurate against verification-code.md F-08/F-07).
6. Pose data: **VALUES VERIFIED** for ep784/ep779/ep697; ep784 provenance
   layering is a trap at the nested-field level (D-03, bookkeeping only).
7. No new damage: **PASS** (no episode/variant special-casing in code;
   seeding/determinism unchanged).

**CERTIFIED-WITH-FINDINGS — code class, twin-suite v1.1 final.**
