# Verification — Twin-Suite v2 Spec-Only Generator + Demonstration Set (Artifact Class (b), Amendment 2 §A2.1)

**Verdict: CERTIFIED-WITH-FINDINGS**

**Date:** 2026-11-22 · **Verifier:** independent context (builder ≠ verifier, pre-reg §9)
**Builder report:** `twin-suite-v2-docs/spec-only-v2.md`
**Governing text:** `twin-suite-v2-preregistration-20261122.md`, Amendment 2 §A2.1 / §A2.2
**Artifacts verified (DGX, read-only + scratch re-runs):**
`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/spec_only/` —
`generate_spec_only.py`, `specs/specs.json`, `specs/*.md`, `scenes/` (100 XMLs),
`suite_manifest.json`, `validate_spec_only.py`, `settle_check.json`, `runs/settle1/` guard record.
Cross-sources: `twin-suite-v2-docs/parameter-audit-v2.md` (§3.1 table) and the frozen v1.1
`code/twin_params.py` (DGX `zk0-study001-twin-20261122/`).

File identities confirmed against the builder report's printed hashes:
generator SHA-256 `d5b73567…`, `specs.json` `09d89574…`, `suite_manifest.json` `6e9178d1…`
(full hashes recomputed by the verifier, prefixes match).

---

## 1. No episode-data leakage — PASS (decisive check)

- `generate_spec_only.py` read end-to-end (362 lines). Its **only file reads** are
  `specs/specs.json` (line 23) and the arm include path string
  `scenes-v2/arm/so101_new_calib_padded.xml` embedded in the XML `<include>`.
  Imports: `argparse, hashlib, json, math, random, pathlib` only. No network, no
  dataset, no pose/split/records access of any kind.
- Grep of the generator, `specs.json`, and `validate_spec_only.py` for
  `episode|split.json|load_episode|records/|held-out|pose` and for every held-out
  episode id from pre-reg Appendix A: **no hits** except the docstring/comment
  declarations that no episode data is used, and `episode_data_used: false` written
  into the manifest. No episode id, recorded pose, or per-episode value appears
  anywhere in code or specs.
- Sampling provenance confirmed in code: placements come from
  `random.Random(20261126)` (`MASTER_SEED` read from `specs.json`) drawing
  `rng.uniform(...)` over the regions declared in `specs.json`, in the documented
  draw order (tasks alphabetical; scene 0–3; placement, then variants k=1..4).
  Variant draws: jitter uniform ±σ (σ per task), mass uniform over
  `variant_protocol.mass_ranges_kg`, friction multiplier uniform [0.8, 1.2]. v0 is
  nominal. Nothing is read from, fitted to, or indexed by any recording.
- The one code-source (non-episode) input beyond the spec file — fixture positions
  and the cable coil layout — was traced to the frozen v1.1 code, §5 below.

## 2. Spec fidelity vs the §3.1 parameter audit — PASS

Every object dimension / mass / friction in `specs.json` was checked against
`parameter-audit-v2.md`; all match the audited nominal values, with the audit's
labels carried:

| Task | Checked values (all match audit) |
|---|---|
| Global | mat 550×400×3 mm [MEASURED-EST, G8/OI-1], walls 60 mm [G10], timestep 0.0005 s [G2], gravity −9.81, restitution 0, floor/mat/wall frictions (G7/G9/G11) as in the generator's common XML |
| block_stack | cube edge 30 mm [B1], mass 20 g [B2], friction (0.6, 0.005, 0.0001) [DEFAULT, B4] |
| eye_drops_to_basket | carton 90×35×35 mm [E1], 30 g [PROP, E2], friction 0.5 [E3]; platform top 200×160×4 mm, height 90 mm [E4], friction 0.4 [E5]; basket 260×180 mm, wall 70 mm / ≈3 mm [E6], friction 0.4 [E7] |
| ring_insert | board Ø170×15 mm [R1], 6 recesses not cut [R2], friction 0.5 [R3]; peg Ø16 mm × 70 mm [R4], friction 0.4 [R5]; ring outer Ø45 / inner Ø22 / thick 12 mm, 12 segments [R6], mass 10 g [R7], friction 0.5 [R8] |
| tool_insert | tray 385×270×45 mm [PROP, T1], friction 0.6 [T2]; slot 110×32 mm [DEFAULT, T3], depth 5 mm [T4 with its qualified "ESTIMATE, never measured" label preserved]; tool 100 mm × Ø10 mm [T6], 70 g [T7], friction 0.4 [T8] |
| cable_clip | v1 table §6 lineage only, labeled as such (no v2 audit label claimed): cable 24 × 30 mm capsules, Ø5 mm, total 45 g, friction (0.8, 0.005, 0.001); clip block 110×70×55 mm; USB 25×12×8 mm / 6 g — all matching `twin_params` cable constants |

Variant mass ranges match the audit's frozen-code sampling ranges
(cube 10–35 g, carton 20–45 g, ring 4–15 g, tool 50–100 g, cable 40–60 g);
jitter σ matches pre-reg §3.3 (block 2.5, ring 2.5, eye_drops 3.5, tool 3.5 cm),
cable 0.020 m declared [ESTIMATE] under G-C1.

Success-criteria text: the five quoted criteria in the builder report §4 are
**character-identical** to the criteria printed in the five `specs/*.md` files,
and each spec correctly notes the secondary position criterion (3.0 cm; ring
2.0 cm; cable exempt) as episode-suite-only / not applicable spec-only.

## 3. Determinism — PASS (re-run by verifier)

The generator was re-run in a scratch sibling copy on the DGX (same parent, so
the arm-include path string is unchanged), output to a fresh directory:

- Regenerated `suite_manifest.json` is **exactly equal** (`==`) to the shipped manifest.
- All **100/100** regenerated XML SHA-256 hashes match both the manifest entries
  and the shipped scene files. 0 mismatches.
- The shipped scenes' per-file hashes in the manifest also verify directly
  against the files on disk (same comparison).

## 4. Settle check — PASS (recomputed by verifier)

`validate_spec_only.py` was re-run by the verifier in the scratch copy
(MuJoCo 4.0.1, sim venv on the DGX):

- Re-run output: **"settle check: 100/100 clean; failures: 0"**, and the
  regenerated `settle_check.json` is **identical** to the shipped one.
- 100/100 variants load + settle 1.0 s, finite state, **0 solver failures**.
- Drift maxima recomputed from the settle data: cubes 0.08 mm, carton 0.02 mm,
  ring 0.26 mm, cable seg_0 4.9 mm — all ≤5 mm as reported.
- Loose-tool roll confirmed exactly as reported: `tool_insert__spec00_v0`
  2.05 cm (≈2.1 cm claimed); spec03 variants v0–v4 = 1.62 / 5.95 / 6.33 / 5.32 /
  1.89 cm (claimed range 1.6–6.3 cm ✓). Every other tool variant drifts
  ≤4.4 mm, so the report's "except the loose tool" framing is accurate, and
  the report correctly flags that the manifest ground truth is the
  **pre-settle** sampled state — a solvability-executor convention to settle
  in A2.2, disclosed by the builder, not hidden.

## 5. Gap register honesty — PASS with minor findings F2–F3

Traced declarations:

- **Fixture positions** verified character-for-character against the frozen
  `twin_params.DEFAULT_FIXTURES`: platform (0.32, −0.06), basket (0.30, 0.15),
  board (0.30, 0.0), tray (0.40, 0.0), slot (0.35, 0.06), clip block (0.30, 0.05)
  — all match `specs.json`, all labeled [DEFAULT] (gaps G-E1/G-R2/G-T2/G-C3).
- **Cable layout**: generator's `default_cable_points` (2 turns, r=55 mm,
  centre (0.27, −0.07)) matches the declared v1 code default (G-C1).
- **Basket overhang** (G-E2): basket centre y=0.15 + 90 mm half-depth = 0.24 >
  mat edge 0.20 — kept verbatim and flagged in both `specs.json` and the spec
  doc, as claimed. Not silently moved. ✓
- **OI-1 / OI-2** carried at audited values with labels; **cube masses** drawn
  10–35 g with the audited 5 g foam corner explicitly flagged in `specs.json`
  (`mass_range_note`) as envelope-only, not a variant draw. ✓
- Placement distributions for all five tasks are printed with ranges in
  `specs.json` and each spec doc, labeled [DEFAULT] declared choices
  (G-B2/G-E3/G-R3/G-T3/G-C1). Colours (G-B1/G-R1), moved-cube identity (G-B3),
  carton orientation (G-E4), omitted seated tools (G-T1), cable's lack of a v2
  audit (G-C2) — all declared in both the report's gap table and the specs
  themselves. The verifier found **no episode-derived value filling any gap**.

Minor undeclared traceability items found by the verifier (neither is
episode data; both are inherited v1.1 scene construction, but neither is
traced in `specs.json` or assigned a gap ID):

- **F2 — Platform web/base sub-geometry.** The generator hardcodes the
  platform web (10 mm wide × td/4 deep) and a 4 mm base plate; `specs.json`
  carries only the platform top size + height. The audit's E4 note declares
  web/base "not separately parameterized" for the frozen code, and the frozen
  v1.1 XMLs do contain `platform_web`/`platform_base` geoms — so the lineage
  is real, but the spec file a Tier C validator receives does not state these
  dimensions.
- **F3 — Cable seg_0 free-joint damping 0.05.** `specs.json` declares
  `joint_damping: 0.2` (correct for the ball joints, and the cable spec doc
  says so), but the generator hardcodes damping 0.05 on the seg_0 free joint
  with no spec entry. (All other free bodies use damping 0.01, matching G5.)

## 6. Manifest completeness — PASS with minor finding F1

- Per task: `denominator_nominal: 4`, **`denominator_validated: 0`** for all
  five tasks; the manifest header states solvability validation is NOT
  included and denominators are 0 until the A2.2 workstream runs. **No
  solvability is claimed anywhere** — report, manifest, README, and spec docs
  agree. The builder report's "0 / 4 per task" denominator statement is
  correct, including for cable_clip (whose Tier C set is a demonstration;
  its gate remains the v1.1 suite).
- Per-scene entries carry: sampled placement, all variant draws
  (jitter / mass / friction multiplier per variant), per-file SHA-256 for all
  100 files, ground-truth initial state, target specification (goal object,
  fixture / slot / peg / channel coordinates — sufficient for a solvability
  executor), and a parameter-label pointer into `specs.json`.
- **F1 — Seed is suite-level only.** The master seed 20261126 is registered
  in the manifest header and in every XML's header comment, but individual
  scene entries carry no seed field. Functionally sufficient (one RNG stream,
  documented draw order, determinism proven in §3 above), but the literal
  "seed per scene entry" is not present.
- Observation (not a finding): per-variant ground-truth initial states are
  not stored separately; the stored ground truth is the nominal scene state,
  and each variant's exact initial state is deterministically recoverable
  from the recorded per-variant draws — which the verifier's regeneration
  (§3) proves. A solvability executor should apply the draws, and for
  tool_insert account for the settle roll (§4).

## Disclosed deviation noted, not counted against certification

§A2.1's Stage 1 deliverable sentence names a demonstration set
"solvability-validated"; this build delivers the generator + demonstration
set with solvability explicitly deferred to the A2.2 reference-controller
workstream and validated denominators of 0/4 throughout. The deferral is
disclosed in the builder report (§2, §7), the manifest, and the README —
nothing is silently claimed. Certification here covers the generator and
the demonstration set **as a generated, settle-checked, unvalidated set**,
which is what the artifacts are.

## Guard note

The builder report's §6 account (guard mirror hard gate hit;
`JOB_GUARD_ALLOW_NO_MIRROR=1` override; telemetry on-box) is corroborated by
the on-box record: `runs/settle1/guard/guard.log` records "mirror disabled:
JOB_GUARD_MIRROR_TARGET unset — telemetry stays ON-BOX", the launch log
records the explicit override warning, and the guard exit code is 0.
The verifier's own re-runs were plain scratch computations, not guard jobs;
the scratch copy was removed after verification and no v1.1 or spec_only
file was modified.

---

## Findings summary

| ID | Severity | Finding |
|---|---|---|
| F1 | Minor | Master seed registered at suite level only; no per-scene seed field in manifest entries (determinism independently proven, so no functional gap). |
| F2 | Minor | Platform web/base sub-geometry dimensions hardcoded in the generator, absent from `specs.json`, no gap ID (inherited v1.1 construction; audit E4 declares the sub-geometry unparameterized). |
| F3 | Minor | Cable seg_0 free-joint damping 0.05 hardcoded, untraced in `specs.json` (declared 0.2 applies to the ball joints). |

No major findings. Items 1 (leakage), 3 (determinism), and 4 (settle) pass
outright; items 2, 5, 6 pass with the minor traceability findings above.

**Verdict: CERTIFIED-WITH-FINDINGS** — the spec-only generator and
demonstration set are certified as a Tier C (§A2.1) generated,
settle-checked set with validated denominators 0/4 per task and no
solvability claimed, subject to findings F1–F3 being recorded (a future
suite version should carry the seed per scene entry and trace the F2/F3
construction values in `specs.json`).

---

## Delta verification — hygiene pass F1–F3 (2026-11-23)

**Verdict: CERTIFIED**

**Date:** 2026-11-23 · **Verifier:** independent context (builder ≠ verifier),
separate from both the builder and the original verifier above.
**Scope:** delta only — closure of findings F1–F3 as described in §8 of
`spec-only-v2.md`. The original verification above is unaltered; its
CERTIFIED-WITH-FINDINGS verdict stands for the pre-pass artifacts.
**Artifacts checked (DGX, read-only):**
`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/spec_only/` —
`generate_spec_only.py`, `specs/specs.json`, `suite_manifest.json`,
`runs/hygiene1/scene_hashes_pre.txt` + `scene_hashes_post.txt`, the two
amended spec docs; cross-traced in the frozen v1 tree
`/home/openclaw/muse-work/zk0-study001-twin-20261122/code/`
(`build_scenes.py`, `twin_params.py`).

Post-pass file identities recomputed by this verifier, matching §8's
printed prefixes: generator `c96fcec4…`, `specs.json` `d1045b54…`,
`suite_manifest.json` `dfd67858…` (full hashes recomputed; prefixes match).

### 1. Hash identity — PASS

- `scene_hashes_pre.txt` and `scene_hashes_post.txt` each contain 100
  entries; `diff` is clean — the lists are identical, 100/100.
- SHA-256 of each list is
  `5e3452e53883e0af54b09d3b738d3ccee2208617299d150a8c966fab0e3a240b`,
  exactly the aggregate claimed in §8, for both lists.
- Spot recomputation by this verifier: SHA-256 recomputed from disk for
  10 scene XMLs spread across all five tasks (indices 0, 11, 23, 37, 49,
  55, 67, 79, 88, 99 of the sorted list) — 10/10 match the listed hashes.
  For completeness all 100 on-disk files were also recomputed: 100/100
  match the pre list, and the manifest's per-variant `sha256` entries
  match the pre list 100/100. No generated scene value changed.

### 2. F1 — Per-scene seed — PASS (closed)

- All 20 scene entries in `suite_manifest.json` carry a `seed` field.
  Checked programmatically against the documented derivation
  `seed = 20261126 + task_order_index × 4 + scene_index` with alphabetical
  task order (block_stack=0, cable_clip=1, eye_drops_to_basket=2,
  ring_insert=3, tool_insert=4): 20/20 match, running 20261126–20261145
  in manifest order.
- The manifest header records `per_scene_seed_derivation` with the same
  formula and the note that the seed is a recorded identifier only
  (generation still uses the single `Random(20261126)` stream); the
  generator docstring and the manifest-construction code
  (`"seed": MASTER_SEED + ti * SCENES_PER_TASK + si`) say the same.
- Otherwise-unchanged: the pre-pass manifest file itself was not retained
  on the DGX, so the builder's literal structural diff cannot be replayed
  byte-for-byte. It is corroborated on every checkable consequence: the
  manifest's structure matches the original verification's §6 description
  plus exactly the two additions (header `per_scene_seed_derivation`,
  per-scene `seed`); denominators remain nominal 4 / validated 0 per task;
  and all 100 variant hashes in the regenerated manifest equal the pre
  list (§1), which pins placements, draws, and scene bytes. Recorded as
  an evidence note, not a finding.

### 3. F2 — Gap G-E5, platform web/base sub-geometry — PASS (closed)

Registered consistently in all four places §8 names:

- `specs.json`, `tasks.eye_drops_to_basket.fixtures.platform.sub_geometry`:
  web half-extents [0.01, 0.04, 0.045] m (full 0.02 × 0.08 × 0.09 m),
  base half-extents [0.1, 0.08, 0.002] m (full 0.20 × 0.16 × 0.004 m) at
  z=0.002 — arithmetic verified against tw=0.20, td=0.16, h=0.09
  (web = 0.01 × td/4 × h/2; base = tw/2 × td/2 × 0.002).
- `specs.json` top-level `spec_gaps_register_additions_2026-11-23` carries
  G-E5 with the same dimensions.
- The eye_drops spec doc's gap list carries G-E5, and the generator
  carries a G-E5 comment at the platform construction.
- Labeling as claimed: [DEFAULT] inherited v1.1 construction, untraced in
  the audited parameter table (audit E4 declares web/base "not separately
  parameterized"). This verifier confirmed the v1 source expressions are
  identical (`build_scenes.py`: `platform_web` size `0.01 {td/4} {h/2}`,
  `platform_base` size `{tw/2} {td/2} 0.002` at z=0.002), and a generated
  scene XML carries exactly those sizes.

### 4. F3 — Gap G-C4, cable seg_0 damping — PASS (closed)

Traced independently by this verifier in the v1 lineage:

- `zk0-study001-twin-20261122/code/build_scenes.py`, `cable_xml_build`
  (line 664): `'<joint name="cable_seg_0_free" type="free" damping="0.05"/>'`
  — hardcoded 0.05, as claimed.
- The ball joints in the same function use `P.CABLE_JOINT_DAMPING`, and
  `twin_params.py` (line 110) sets `CABLE_JOINT_DAMPING = 0.2` — the
  discrepancy is real and is exactly as §8 describes.
- Registration is consistent: `specs.json` records
  `seg0_free_joint_damping: 0.05` separately from `ball_joint_damping` /
  `joint_damping: 0.2`, with a [DEFAULT]/untraced G-C4 label; G-C4 also
  appears in the top-level additions register, in the cable_clip spec
  doc (both the parameter line and the gap list), and as a generator
  comment at the seg_0 joint construction.
- Value unchanged: every generated cable scene still carries
  `cable_seg_0_free … damping="0.05"` (one per file, alongside 23 ball
  joints at 0.2), consistent with the §1 byte-identity proof.

### 5. No v1.1 file modified — PASS

`find` over the entire frozen v1 tree
(`/home/openclaw/muse-work/zk0-study001-twin-20261122/`) for files newer
than 2026-11-23 00:00 returns 0 files; the tree's latest mtime is
2026-11-22 20:00. All hygiene-pass mtimes (2026-11-23 ~03:09) are inside
the v2 `spec_only/` workdir only. This verifier's own access was
read-only throughout.

### Delta verdict

**CERTIFIED** — findings F1, F2, and F3 are closed as described in §8 of
`spec-only-v2.md`: per-scene seeds are present, correctly derived, and
documented in the manifest header; G-E5 and G-C4 are registered
consistently across `specs.json` (including the additions register), the
affected spec docs, and the generator comments, with dimensions and
values as stated and correctly labeled [DEFAULT]/untraced; the F3 value
was not changed; all 100 scene XMLs are byte-identical pre/post pass
(aggregate `5e3452e5…` verified); and no v1.1 file was modified. The
underlying suite remains as originally certified: a Tier C generated,
settle-checked set with validated denominators 0/4 per task and no
solvability claimed. The one evidence note (§2: pre-pass manifest not
retained, so the structural diff is corroborated rather than replayed)
does not affect any checkable claim.
