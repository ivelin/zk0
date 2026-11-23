# Twin-Suite v2 — Stage 1.1 Parameter Audit: Independent Verification

**Artifact class:** (a) parameter audit (pre-reg §9 + Amendment 1 §A1.4 mapping).
**Verifier:** independent context, raw artifacts only; builder's conclusions not used.
**Date:** 2026-11-22. **Governing:** `twin-suite-v2-preregistration-20261122.md`
§3.1/§3.3 + Amendment 1 (folder root of the notes tree, not the v2-docs folder).

## Verdict: CERTIFIED-WITH-FINDINGS

The audit is complete and label-honest in substance: every v1-table physics
parameter for the four precision tasks and every v1.1 change is carried with
value, range or explicit point-only marking, label, and source; no label is
upgraded beyond its source; D1–D6 all independently confirm; OI-1/OI-2 are
genuinely open. Four findings (F1–F4) below — none blocks certification under
§3.1 (which expressly states the two record measurements do not block), but
F1/F2 should be fixed in the JSON / label before the envelope harness consumes
them, because the JSON is the machine-readable form a harness will trust.

---

## 1. Completeness — PASS (with F3, F4)

Method: row-by-row diff of v1 `physics-parameters.md` (§§1–5, 7, 9) +
`scene-inventory.md` tolerances against audit rows G1–G13, B1–B5, E1–E7,
R1–R9, T1–T8, then v1.1 changes from the v1.1 completion report §5 and
`twin-suite-v11-docs/builder3-v11-scenes.md` (changelog #8–#10, #14).

- All v1 rows map: floor/mat/walls/gripper friction, mat dims, tape (G7–G13);
  cube (B1–B4); carton/platform/basket (E1–E7); board/peg/ring (R1–R8);
  tray/slot/seated/loose tool (T1–T8); gravity, restitution (G1, G6).
- All v1.1 changes map: timestep (G2/#14), per-episode colours (B5, R9/#8),
  slot recess (T3, T4/#9), recess count (R2/#10), ring-mass derivation (R7).
- Range arithmetic spot-recomputed and correct throughout: mat ±20%
  (440–660 × 320–480), cube 30±10 mm, carton ±20%/axis, platform height
  90±15 mm, basket walls 60–80 mm, board Ø170±20, peg Ø16±4 / 70±15 mm,
  ring outer 45±10 / inner 22±5 / thick 12±3 mm, tool length 100±25 mm —
  all match scene-inventory tolerances exactly.
- Label-count arithmetic recomputed: Global 3/0/10, block 3/0/1,
  eye 3/1/3, ring 5/0/3, tool 3/2/3 → **17 [MEASURED-EST] / 3 [PROP] /
  20 [DEFAULT] / 40** — matches the audit's totals table; the exclusion of
  the visual-only colour rows (B5, R9) from the counts is declared in the audit.
- Nothing silently changed: the only value-level divergences from the v1
  table are exactly D1–D6, all disclosed (§3 below).

## 2. Label honesty — PASS (with F2)

- Friction and restitution: **every** friction row and G6 restitution remain
  [DEFAULT], matching v1. No friction value was relabeled.
- No upgrades found: mat dims, cube/carton/ring/tool masses, tray [PROP],
  seated-tool material [PROP] all carry the same labels as v1. The one label
  *change* besides F2 is conservative: slot footprint T3, [MEASURED-EST]
  shape/location in v1, is mapped to **[DEFAULT]** in v2 because the frozen
  code records `SLOT_SIZE` as `LOCAL-CHOICE` (verified in `twin_params.py`:
  `SLOT_SIZE = (0.110, 0.032)  # target slot footprint (x,y) LOCAL-CHOICE`).
- Frozen-code sampling rule verified verbatim on the DGX (read-only):
  `FRICTION_MULT_RANGE = (0.8, 1.2)   # +-20% where Builder A gives no range`,
  `MASS_RANGES = {cube_a/cube_b: (0.010, 0.035), carton: (0.020, 0.045),
  ring: (0.004, 0.015), tool: (0.050, 0.100)}`, `TIMESTEP = 0.0005`.
- Envelope friction ranges recomputed from the audit's stated rule
  (union of nominal×[0.8,1.2] with the v1-cited literature range):
  floor [0.50, 0.96] ✓, mat [0.70, 1.08] ✓, tool [0.30, 0.50] ✓,
  all ×-only rows (walls, gripper, cube, carton, platform, basket, board,
  peg, ring, tray) ✓. Every friction corner is flagged [DEFAULT]-derived
  in both the audit (§1, §8) and the JSON notes, as §3.1 requires.

## 3. Discrepancies D1–D6 — ALL CONFIRMED

- **D1 CONFIRMED (direct evidence).** DGX frozen doc
  `docs/physics-parameters.md` line 13 still reads "Timestep for twin
  scenes: 0.002 s"; `twin_params.py` has `TIMESTEP = 0.0005` (changelog #14
  comment); sampled scene XMLs (block ep411, eye ep1609, ring ep7, tool
  ep622) all carry `<option timestep="0.0005" gravity="0 0 -9.81"/>`.
  Audit G2 takes 0.0005 — correct direction.
- **D2 CONFIRMED.** B3 report §3a + v1.1 verification: count corrected
  5 → 6 (ep7 shows six empty recesses; ep33 five empty + ring in a sixth);
  v2 R2 takes 6, not cut — matches frozen docs.
- **D3 CONFIRMED (recomputed).** Ring volume = π(22.5² − 11²) mm² × 12 mm
  = **14.52 cm³** (v1 printed ≈5.5 cm³ — does not reproduce);
  × ≈0.7 g/cm³ ≈ 10.2 g. Frozen XML ring mass = 12 × 0.000833 kg = 0.010 kg.
  **Value (10 g) and range (4–15 g) unchanged** — derivation-only fix, as claimed.
- **D4 CONFIRMED.** Per-episode colours in v1.1 (29 body-instances, ring
  ep48 yellow→red at source) per completion report §5 / B3 §1; visual only.
- **D5 CONFIRMED.** v1 table §5 promised a 4-wall + floor slot recess;
  v1.1 B3 §2 records the v1 scenes had none (single solid tray box) and
  builds it in v1.1. Frozen tool ep622 XML contains `slot_floor` with
  half-extents 0.055 × 0.016 m = **110 × 32 mm** ✓. Depth **5 mm** is
  labelled **[ESTIMATE, never measured]** in B3 §2 and changelog #9
  (settle test 0.0459 m in-slot vs 0.0499 m flat, completion report §5) ✓.
- **D6 CONFIRMED.** v1 table §2 range reads 10–35 g while §9 states the
  material-contingent uncertainty as 5–35 g (both read directly in the v1
  doc); frozen code samples 10–35 g (above). v2 envelope uses the full
  5–35 g and flags the 5 g corner as unsampled-by-v1.1-machinery — correct
  and conservative.

Also spot-verified in frozen XML: weld `solref="0.005 1"`,
`solimp="0.95 0.99 0.001"` (G4) present; no contact solref/solimp set
beyond the weld line (G3's grep claim holds on the sampled scene).

## 4. Open items OI-1 / OI-2 — CONFIRMED OPEN

Independent spot-check, 2026-11-22:

- **HF dataset card** `armnet/armnetbench_v01_lerobot_so101` read in full:
  tasks listed with instructions only ("Stack the colouful blocks…");
  **no mat dimensions, no cube material/product/mass anywhere on the card.**
- **Paper** arXiv:2607.24481v1 (HTML) read + full-text finds: "WOWROBO"
  occurs **zero** times; no mat specification; the bill of materials
  (Table 2) lists only the A3 cardboard sheets as workspace material.
  Task Table 4 gives instructions only — no cube material/product/mass.
- **A3 figure CONFIRMED and NOT substituted:** §3.1 "Enclosure and
  workspace. A3 cardboard sheets (297 × 420 mm) bound the workspace" —
  it describes the cardboard enclosure, exactly as the audit's OI-1 states,
  and the audit keeps the mat at its own 550 × 400 × 3 mm [MEASURED-EST]
  estimate rather than adopting 297 × 420 mm. Correct handling.
- No value in the audit is dressed as measured when estimated, **except**
  the labelling nuance in F2: every estimate carries its method or an
  explicit [ESTIMATE] caveat in its row text.

## 5. envelope-corners-v2.json — PASS (with F1)

- Valid JSON (parsed programmatically). Per task **exactly 10 corners**:
  `nominal` + `pose_+x/−x/+y/−y` + `friction_low/high` +
  `mass_low/high` + `engine_drake` — the §3.3 structure (nominal baseline
  is MuJoCo; the swap corner is Drake).
- Pose σ values: block 0.025, ring 0.025, tool 0.035, eye_drops 0.035 m —
  exactly the frozen §3.3 values (block/ring 2.5, tool/eye 3.5 cm).
- Every friction and mass value in the JSON matches the audit table's
  ranges (global G7/G9/G11/G12 + per-task B4, E3/E5/E7, R3/R5/R8, T2/T8;
  masses B2 incl. the flagged 0.005 kg low corner, E2, R7, T7) — full
  cross-check, zero mismatches. Friction corners carry
  `depends_on_default_at_corner: true`; open items OI-1/OI-2 are embedded
  verbatim with the A3 non-substitution note.

## Findings

- **F1 (fix before harness use) — JSON point-only register is incomplete
  vs the audit's own §8/§9 claim.** The JSON's
  `point_only_held_fixed_in_all_corners` lists only G2, G3, G4, G5, G6 and
  torsional/rolling friction. Audit §8 additionally registers as point-only:
  **tray dimensions (T1), slot footprint (T3), slot depth (T4), and board
  thickness (R1)**. Audit §9 claims the JSON carries "the point-only
  register (§8) the harness must hold fixed" — as shipped, it does not.
  The audit document itself is complete; the machine-readable register is not.
- **F2 (label nuance) — T4 slot depth labelled [MEASURED-EST].** Its source
  (B3 §2) labels it "[ESTIMATE, never measured]". The audit row discloses
  this in full ("not a measurement of the physical tray"; derived so a
  seated tool's centre matches recorded `tool_final` z = 0.045 m), so no
  reader of the row is misled — but the bare label is stronger than the
  source's, and T4 appears in no JSON register (see F1), so a
  label-only consumer would over-trust it. Recommend a distinct marker or
  carrying the [ESTIMATE] caveat wherever T4's label is printed.
- **F3 (minor, completeness trace) — v1 §7 grasp thresholds not carried.**
  The attach/release thresholds (50% / 30% closed, 3.0 cm) appear nowhere
  in the audit. This is defensible — G4 notes the weld is a v1 replay
  device and the v2 gate actuates the arm per pre-reg §4 — but the audit
  never says the thresholds are dropped, so the lineage has a silent gap.
  One line in G4 would close it.
- **F4 (minor) — platform sub-geometry.** v1 §3 describes the platform as
  three boxes (top plate, web, base plate); E4 carries only the top plate
  dims + top height. The frozen code likewise defines only
  `PLATFORM_TOP`/`PLATFORM_HEIGHT`, so nothing is lost against the scenes
  as built — but the web/base dims are not stated as point-only/as-built.

## Claims-evidence lines

- D1 timestep · DGX frozen `docs/physics-parameters.md` L13 vs
  `code/twin_params.py` TIMESTEP vs sampled scene XML `<option>` · read 2026-11-22 via dgx.sh (read-only).
- Friction/mass sampling · DGX `twin_params.py` (`FRICTION_MULT_RANGE`,
  `MASS_RANGES`, `SLOT_SIZE`) · read 2026-11-22.
- OI-1/OI-2 absence · HF dataset card (full read) + arXiv:2607.24481v1 HTML
  (§3.1 A3 passage at paper lines 232–238; WOWROBO find = 0 hits) · checked 2026-11-22.
- D3 arithmetic · recomputed by verifier: π(22.5²−11²)×12 mm³ = 14.52 cm³.
- JSON structure/values · programmatic parse + full cross-check against
  audit tables · 2026-11-22.

---

## Delta Verification — F1–F4 Fix Pass (2026-11-22)

**Verifier:** independent context, separate from the builder fix pass;
raw artifacts only (`parameter-audit-v2.md` incl. its §10 change log,
and `envelope-corners-v2.json`, re-parsed programmatically).
**Scope:** exactly the delta for findings F1–F4 above; the original
findings text above is unaltered.

### Verdict: CERTIFIED

All four findings are fixed as specified, and no other change was found.

- **F1 — FIXED, verified.** The JSON `point_only_held_fixed_in_all_corners`
  register now has 12 entries: the original G2, G3, G4 (weld solref),
  G4 (weld solimp), G5, G6, torsional friction (§8) and rolling friction
  (§8), plus exactly one each of the previously omitted entries —
  T1 tray dimensions (`385 x 270 x 45 mm`, label `PROP`),
  T3 slot footprint (`110 x 32 mm`, label `DEFAULT`),
  T4 slot depth (`5 mm`, qualified label — see F2), and
  R1 board thickness (`15 mm`, label `MEASURED-EST`).
  Values and labels match audit §5/§4 rows and the §8 register; the
  register now matches §8 in count and content (the 12th entry is the
  G4 weld split into solref/solimp, as before). Audit §9 now enumerates
  that complete register (G2, G3, G4, G5, G6, torsional/rolling friction,
  T1, T3, T4, R1) and its claim that the JSON carries it is now true.
  The JSON still parses as valid JSON; per-task corner counts are still
  exactly 10 per task (nominal + pose_+x/−x/+y/−y + friction_low/high +
  mass_low/high + engine_drake) for all four tasks, and no corner values
  changed (pose σ block 0.025 / ring 0.025 / tool 0.035 / eye_drops
  0.035 m; all friction and mass corner values match the audit ranges
  and the values recorded in §5 of the original verification above).
- **F2 — FIXED, verified.** T4's label is the qualified form
  `[MEASURED-EST — depth: ESTIMATE, never measured]` consistently in
  the §5 tool_insert table (label cell and the labelling sentence in its
  source text) and in §8, and as `MEASURED-EST — depth: ESTIMATE, never
  measured` in the JSON register entry. T4 remains counted in the
  [MEASURED-EST] family, so the totals are unchanged; no other label
  was changed (spot-checked across the audit tables, §8, and the JSON).
- **F3 — FIXED, verified.** G4 in §1 now carries the lineage line:
  v1 §7 grasp thresholds (weld attach 50% / release 30% / 3.0 cm) are
  intentionally NOT carried into v2 because the v2 actuated-arm design
  replaces the weld replay device with physical grasp — the actuated-arm
  reason is stated, as required.
- **F4 — FIXED, verified.** E4 in §3 now carries the note that v1 §3's
  web/base sub-geometry is not separately parameterized because the
  frozen code defines only top plate + height (matches frozen scenes;
  declared simplification) — consistent with the frozen code's
  simplification recorded in the original F4 finding
  (`PLATFORM_TOP`/`PLATFORM_HEIGHT` only, cited in the E4 row itself).

### Nothing else changed — spot-checks

- Label totals unchanged: **17 [MEASURED-EST] / 3 [PROP] / 20 [DEFAULT] /
  40 rows**, with the per-scope counts as originally found (Global 3/0/10,
  block 3/0/1, eye 3/1/3, ring 5/0/3, tool 3/2/3).
- Values and ranges unchanged: audit nominals/ranges and all JSON
  friction/mass values cross-check against the original verification's
  recorded values with zero mismatches (e.g. floor friction [0.50, 0.96],
  mat [0.70, 1.08], tool [0.30, 0.50]; cube mass 0.005–0.035 kg with the
  OI-2 flag on the 5 g low corner; ring 0.004–0.015 kg; tool 0.050–0.100 kg).
- Open items unchanged: OI-1 (mat dimensions) and OI-2 (cube material)
  remain OPEN in audit §6 and are embedded in the JSON as before, with
  the A3 non-substitution note intact.

### Claims-evidence lines (delta pass)

- F1/F2 register and labels · programmatic parse of
  `envelope-corners-v2.json` (12 point-only entries; T1/T3/T4/R1 present
  exactly once each) cross-checked against `parameter-audit-v2.md`
  §4/§5 rows, §8 and §9 · 2026-11-22.
- F3/F4 text · `parameter-audit-v2.md` G4 row (§1) and E4 row (§3),
  read in full · 2026-11-22.
- No-other-change spot-check · per-task corner counts and all JSON
  friction/mass/pose values vs the values recorded in the original
  verification §1/§5 above; audit totals table (§5) · 2026-11-22.
