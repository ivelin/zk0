# Twin-Suite v2 — Stage 1.1 Parameter Audit (2026-11-22)

**Scope:** the four precision tasks — block_stack, eye_drops_to_basket,
ring_insert, tool_insert. Built under twin-suite v2 pre-registration
§3.1 (FROZEN, commit `bb11d20`; governing memo
`sim-gate-design-memo-20261122.md`). cable_clip is out of scope (Stage 0,
twin-suite v1.1 certified).

**Lineage:** v1 parameter table (`twin-suite-v1-docs/physics-parameters.md`)
→ v1.1 changes (completion report + `twin-suite-v11-docs/`, param-changelog
on the DGX) → frozen v1.1 scenes (param_hash `a4eb3f8c3611894d`), cross-checked
2026-11-22 against the actual scene XMLs and `code/twin_params.py` on the DGX
(`~/muse-work/zk0-study001-twin-20261122/`, read-only).

**Labels** (pre-reg §3.1): **[MEASURED-EST]** estimated from the recordings,
method stated · **[PROP]** physical-prop knowledge, prop named ·
**[DEFAULT]** MuJoCo/literature default or midpoint. No value here is a
silent invention. Where a parameter's stability role depends on a [DEFAULT]
value at its envelope corner, that is flagged (pre-reg §3.1, last bullet).

**Envelope-range rule (mechanical, applied uniformly):**
- Mass: the v1 table's sourced range, as sampled by the frozen code
  (`twin_params.MASS_RANGES`).
- Sliding friction: v1 gives point values only. The envelope range is
  **[min(nominal×0.8, literature low), max(nominal×1.2, literature high)]**
  where the v1 table cites a literature range; otherwise **nominal×[0.8, 1.2]**
  — the frozen code's own sampling rule (`FRICTION_MULT_RANGE = (0.8, 1.2)`,
  twin_params.py). Every friction range so derived inherits the [DEFAULT]
  label of its nominal. **Flag: every friction corner in this audit is a
  [DEFAULT] corner** — any v2 verdict whose envelope stability depends on a
  friction corner must be flagged as such in every report (pre-reg §3.1).
- Torsional/rolling friction, restitution, timestep, solver settings: v1
  gives point values with no sourced range → **point-only** (see §6 for the
  envelope consequence).
- Pose: per-task σ frozen in pre-reg §3.3 (block 2.5, ring 2.5, tool 3.5,
  eye_drops 3.5 cm/axis, v1.1 held-out measurements).

Units: SI (m, kg, s) unless mm/g stated. Friction = MuJoCo
(sliding, torsional, rolling).

---

## 1. Global / workspace parameters (all four tasks)

| # | Parameter | v2 value (nominal) | Envelope range | Label | Source |
|---|---|---|---|---|---|
| G1 | Gravity | −9.81 m/s² | point-only — no sourced range | [DEFAULT] | v1 table preamble; frozen XML `<option gravity="0 0 -9.81">` ✓ |
| G2 | Timestep | **0.0005 s (2000 Hz)** | point-only — no sourced range | [DEFAULT] | v1.1 changelog #14 (cable-chain integrator instability at 0.002; applied globally); `twin_params.TIMESTEP = 0.0005`; frozen XMLs `timestep="0.0005"` ✓. **v1 table said 0.002 — see Discrepancy D1.** |
| G3 | Contact solver (solref/solimp) | MuJoCo engine defaults (not set in scenes) | point-only — no sourced range | [DEFAULT] | Frozen XMLs set no contact solref/solimp (verified by grep, §7) |
| G4 | Weld (grasp-device) solref / solimp | solref (0.005, 1), solimp (0.95, 0.99, 0.001) | point-only — no sourced range | [DEFAULT] | Frozen XML `<equality><weld …>` in all sampled scenes ✓; grasp device per v1 table §7. (v2 note: the weld is a v1 replay device; the v2 gate actuates the arm per pre-reg §4 — weld rows are carried for lineage, not as gate parameters.) **Lineage: v1 §7 grasp thresholds (weld attach 50% / release 30% / 3.0 cm) are intentionally NOT carried into v2 because the v2 actuated-arm design replaces the weld replay device with physical grasp.** |
| G5 | Free-joint damping (manipulated objects) | 0.01 | point-only — no sourced range | [DEFAULT] | Frozen XMLs `<joint type="free" damping="0.01"/>` ✓ |
| G6 | Restitution (all geoms, all tasks) | 0.0 | point-only — no sourced range | [DEFAULT] | v1 table, every row |
| G7 | Floor (enclosure floor) friction | (0.8, 0.005, 0.0001) | sliding **[0.50, 0.96]** (lit. 0.5–0.9 ∪ ×[0.8,1.2]); torsional/rolling point-only | [DEFAULT] | v1 table §1 (cardboard/paper on wood laminate, lit. 0.5–0.9); frozen XML ✓ |
| G8 | WOWROBO mat — dimensions | 550 × 400 × 3 mm | **440–660 × 320–480 mm** (±20%); thickness 2–3 mm | [MEASURED-EST] | v1 table §1; scene-inventory §1 (pixel-ratio vs tray; thickness from front-view edge). **Records measurement attempted — OPEN ITEM OI-1 (§5).** Frozen XML: half-extents 0.275 × 0.200 × 0.0015 m ✓ (550×400×3) |
| G9 | WOWROBO mat — friction | (0.9, 0.005, 0.0001) | sliding **[0.70, 1.08]** (lit. 0.7–1.0+ ∪ ×[0.8,1.2]); torsional/rolling point-only | [DEFAULT] | v1 table §1 (rubber-mat lit. 0.7–1.0+); frozen XML ✓ |
| G10 | Enclosure walls — dimensions | 4 thin boxes; walls at mat boundary, height 60 mm as built | point-only as built (no sourced range stated in v1) | [MEASURED-EST] | v1 table §1 dims; frozen XML wall geoms (half-height 0.03 m) ✓ |
| G11 | Enclosure walls — friction | (0.5, 0.005, 0.0001) | sliding **[0.40, 0.60]** (×[0.8,1.2]); torsional/rolling point-only | [DEFAULT] | v1 table §1; frozen XML ✓ |
| G12 | Gripper finger/pad friction | (1.0, 0.005, 0.0001) | sliding **[0.80, 1.20]** (×[0.8,1.2]); torsional/rolling point-only | [DEFAULT] | v1 table §1 (appearance [MEASURED-EST]: smooth hard plastic, wrist close-ups) |
| G13 | White tape strips | visual only — no geom | n/a | [MEASURED-EST] | v1 table §1 (declared simplification, carried unchanged) |

Global label count: **[MEASURED-EST] 3** (G8, G10, G13) · **[PROP] 0** ·
**[DEFAULT] 10** (G1–G7, G9, G11, G12).

---

## 2. block_stack

| # | Parameter | v2 value (nominal) | Envelope range | Label | Source |
|---|---|---|---|---|---|
| B1 | Cube edge (×2 cubes) | 30 mm | **20–40 mm** (±10 mm) | [MEASURED-EST] | v1 table §2; scene-inventory §3 (pixel-ratio vs WOWROBO lettering, ep411/453). Frozen XML half-extent 0.015 m ✓ |
| B2 | Cube mass (each) | 20 g | **5–35 g full uncertainty** (see note); frozen-code sampling range 10–35 g | [MEASURED-EST] | v1 table §2 (27 cm³ × 0.7 g/cm³ painted wood ≈ 20 g; "if foam, ≈5 g — mass is the least certain value in this table"); v1 table §9 (5–35 g); `MASS_RANGES cube = (0.010, 0.035)`. Frozen XML mass 0.020 ✓. **Envelope consequence: the material-dependent low corner (5 g, foam) lies OUTSIDE the range the frozen variant machinery sampled (10–35 g). The v2 envelope corner list (§7 / JSON) uses 5–35 g and flags the 5 g corner as unsampled-by-v1.1-machinery and material-contingent (OI-2).** |
| B3 | Cube material | painted wood **or** dense foam — unresolved | n/a (drives B2) | [MEASURED-EST] appearance only | v1 table §2/§9 ("appearance ambiguous in frames"). **Records measurement attempted — OPEN ITEM OI-2 (§5). NOT relabeled.** |
| B4 | Cube friction | (0.6, 0.005, 0.0001) | sliding **[0.48, 0.72]** (×[0.8,1.2]; v1 cites a wood-on-rubber-mat literature range without numbers); torsional/rolling point-only | [DEFAULT] | v1 table §2; frozen XML ✓ |
| B5 | Cube colours (per episode) | purple fixed; second cube green or orange, per-episode from pose records | n/a (visual only; mass/friction colour-independent) | [MEASURED-EST] | v1.1 change: per-episode colours wired into XMLs (29 body-instances corrected across 20 episodes suite-wide for block+ring; B3 report) — **changed vs v1 table's static rendering (D4)** |

Task label count (B1–B4): **[MEASURED-EST] 3** (B1, B2, B3-appearance) ·
**[PROP] 0** · **[DEFAULT] 1** (B4). Open items: OI-2 (via B2/B3).

## 3. eye_drops_to_basket

| # | Parameter | v2 value (nominal) | Envelope range | Label | Source |
|---|---|---|---|---|---|
| E1 | Eye-drops carton dimensions | 90 × 35 × 35 mm | **±20% per axis** (72–108 × 28–42 × 28–42 mm) | [MEASURED-EST] | v1 table §3; scene-inventory §4. Frozen code `CARTON_SIZE = (0.090, 0.035, 0.035)` ✓ |
| E2 | Carton mass | 30 g | **20–45 g** | [PROP] (+ [MEASURED-EST] volume check) | v1 table §3 (retail eye-drops carton + 10–15 ml bottle ≈ 25–40 g typical); `MASS_RANGES carton = (0.020, 0.045)`; frozen XML mass 0.03 ✓ |
| E3 | Carton friction | (0.5, 0.005, 0.0001) | sliding **[0.40, 0.60]**; torsional/rolling point-only | [DEFAULT] | v1 table §3 |
| E4 | Platform dimensions | top plate 200 × 160 × 4 mm; top surface 90 mm above mat | height **75–105 mm** (±15 mm); plate ±20% (inventory class) | [MEASURED-EST] | v1 table §3; scene-inventory §4 (height vs carton length, front views). Frozen code `PLATFORM_TOP`, `PLATFORM_HEIGHT = 0.090` ✓ **Note: v1 §3's web/base sub-geometry is not separately parameterized because the frozen code defines only top plate + height (matches frozen scenes; declared simplification).** |
| E5 | Platform friction | (0.4, 0.005, 0.0001) | sliding **[0.32, 0.48]**; torsional/rolling point-only | [DEFAULT] | v1 table §3; frozen XML (platform_top/web/base) ✓ |
| E6 | Basket dimensions | 260 × 180 mm footprint, wall height 70 mm, wall ≈3 mm | walls **60–80 mm** high; footprint ±20% (inventory class) | [MEASURED-EST] | v1 table §3; scene-inventory §4. Frozen code `BASKET_OUTER/WALL_H/WALL_T` ✓. Ribs not modelled — declared simplification carried |
| E7 | Basket friction | (0.4, 0.005, 0.0001) | sliding **[0.32, 0.48]**; torsional/rolling point-only | [DEFAULT] | v1 table §3; frozen XML (basket_floor/w1–w4) ✓ |

Task label count: **[MEASURED-EST] 3** (E1, E4, E6) · **[PROP] 1** (E2) ·
**[DEFAULT] 3** (E3, E5, E7). Open items: none task-specific.

## 4. ring_insert

| # | Parameter | v2 value (nominal) | Envelope range | Label | Source |
|---|---|---|---|---|---|
| R1 | Board dimensions | cylinder Ø170 mm × 15 mm | Ø **150–190 mm** (±20 mm, inventory); thickness point-only as built | [MEASURED-EST] | v1 table §4; scene-inventory §5. Frozen XML half-extents (0.085, 0.0075) ✓ |
| R2 | Board recesses | **6 recesses, NOT cut (flat top)** | count fixed at 6; depth ≈2–3 mm (not modelled) | [MEASURED-EST] | **v1.1 change (D2):** count corrected 5 → 6 (B3 report §3a; changelog #10; settling evidence ep7/ep33). Depth: v1 table §4 declared simplification, carried |
| R3 | Board friction | (0.5, 0.005, 0.0001) | sliding **[0.40, 0.60]**; torsional/rolling point-only | [DEFAULT] | v1 table §4; frozen XML ✓ |
| R4 | Peg dimensions | Ø16 mm, 70 mm above board | Ø **12–20 mm** (±4 mm); height **55–85 mm** (±15 mm) | [MEASURED-EST] | v1 table §4; scene-inventory §5. Frozen XML (r 0.008, half-height 0.035) ✓ |
| R5 | Peg friction | (0.4, 0.005, 0.0001) | sliding **[0.32, 0.48]**; torsional/rolling point-only | [DEFAULT] | v1 table §4; frozen XML ✓ |
| R6 | Ring dimensions | outer Ø45, inner Ø22, thickness 12 mm | outer Ø **35–55 mm** (±10); inner Ø **17–27 mm** (±5); thickness **9–15 mm** (±3) | [MEASURED-EST] | v1 table §4; scene-inventory §5 (wrist close-up ep31). As built: 12-segment capsule ring, segment radius 0.00575 m (frozen XML ✓) — the v1 table's Phase-4 choice, logged |
| R7 | Ring mass | 10 g | **4–15 g** | [MEASURED-EST] | v1 table §4 **as corrected in v1.1 (D3):** volume = π(22.5²−11²) mm² × 12 mm ≈ 14.5 cm³ (v1 printed ≈5.5 cm³ — arithmetic error); × ≈0.7 g/cm³ ≈ 10 g; range spans densities ≈0.3–1.0 g/cm³. **Value and range unchanged from v1.** `MASS_RANGES ring = (0.004, 0.015)`; frozen XML 12 × 0.000833 kg = 0.010 ✓ |
| R8 | Ring friction | (0.5, 0.005, 0.0001) | sliding **[0.40, 0.60]**; torsional/rolling point-only | [DEFAULT] | v1 table §4; frozen XML ✓ |
| R9 | Ring colour (per episode) | yellow or red, per-episode from pose records (ep48 corrected yellow→red at source) | n/a (visual only) | [MEASURED-EST] | v1.1 change: per-episode colours (B3 report §1; changelog #8) — **changed vs v1 (D4)** |

Task label count (R1–R8): **[MEASURED-EST] 5** (R1, R2, R4, R6, R7) ·
**[PROP] 0** · **[DEFAULT] 3** (R3, R5, R8). Open items: none task-specific.

## 5. tool_insert

| # | Parameter | v2 value (nominal) | Envelope range | Label | Source |
|---|---|---|---|---|---|
| T1 | Tray dimensions | 385 × 270 × 45 mm | point-only — no sourced range (PROP class cross-checked against frames; no tolerance stated) | [PROP] | v1 table §5 (Sealey Premier EVA tray class), cross-checked [MEASURED-EST]; scene-inventory §6. Frozen code `TRAY_SIZE`; XML half-extents (0.1925, 0.135, 0.02 slab + rims to 45 mm) ✓ |
| T2 | Tray friction | (0.6, 0.005, 0.0001) | sliding **[0.48, 0.72]**; torsional/rolling point-only | [DEFAULT] | v1 table §5 (foam is high-grip); frozen XML (tray/slot_floor/rims) ✓ |
| T3 | Target-slot footprint | **110 × 32 mm** (as built) | point-only — no sourced range | [DEFAULT] | **v1.1 change (D5):** v1 scenes omitted the recess entirely (tray was one solid box) despite the v1 table promising a 4-wall + floor inset. v1.1 builds slab-to-40 mm + rim boxes + red slot_floor plate. Footprint = `twin_params.SLOT_SIZE`, recorded in code as LOCAL-CHOICE (B3 report §2); mapped to [DEFAULT] under the v2 three-label taxonomy |
| T4 | Target-slot recess depth | **5 mm** (effective pocket 4 mm above floor plate) | point-only — no sourced range | [MEASURED-EST — depth: ESTIMATE, never measured] | v1.1 B3 report §2 + changelog #9: **[ESTIMATE, never measured]** — set to the loose tool's radius so a seated tool's centre rests at tray-top height, matching the recorded `tool_final` z = 0.045 m. Settle test: in-slot centre z 0.0459 m vs 0.0499 m flat. Labelled [MEASURED-EST — depth: ESTIMATE, never measured] here only in the taxonomy sense "derived from recorded values"; it is **not** a measurement of the physical tray |
| T5 | Seated tools (≈11) — material | chrome-vanadium steel; static, merged with tray | n/a | [PROP] | v1 table §5 (Sealey Premier extension bars); identities/labels [MEASURED-EST] (tray print, scene-inventory §6) |
| T6 | Loose tool dimensions | length 100 mm, shaft Ø10 mm (capsule as built) | length **75–125 mm** (±25 mm); Ø point-only as built | [MEASURED-EST] | v1 table §5 (wrist close-up ep630); frozen XML capsule fromto ±0.05 m, r 0.005 ✓ |
| T7 | Loose tool mass | 70 g | **50–100 g** | [MEASURED-EST] | v1 table §5 (steel 7.85 g/cm³ × ≈9 cm³ ≈ 70 g); `MASS_RANGES tool = (0.050, 0.100)`; frozen XML mass 0.07 ✓ |
| T8 | Loose tool friction | (0.4, 0.005, 0.0001) | sliding **[0.30, 0.50]** (lit. 0.3–0.5 ∪ ×[0.8,1.2]); torsional/rolling point-only | [DEFAULT] | v1 table §5 (polished steel on foam/mat lit. 0.3–0.5); frozen XML ✓ |

Task label count: **[MEASURED-EST] 3** (T4, T6, T7) · **[PROP] 2** (T1, T5) ·
**[DEFAULT] 3** (T2, T3, T8). Open items: none task-specific.

### Suite totals (this audit's counting: one row = one parameter; friction
triple = one parameter; restitution/timestep/solver counted once, globally)

| Scope | [MEASURED-EST] | [PROP] | [DEFAULT] | Total |
|---|---|---|---|---|
| Global | 3 | 0 | 10 | 13 |
| block_stack | 3 | 0 | 1 | 4 |
| eye_drops_to_basket | 3 | 1 | 3 | 7 |
| ring_insert | 5 | 0 | 3 | 8 |
| tool_insert | 3 | 2 | 3 | 8 |
| **Four tasks + global** | **17** | **3** | **20** | **40** |

Friction and restitution remain entirely [DEFAULT] — inherited from v1,
unchanged by v1.1, and now the dominant label class. (v1's own count —
14/3/13 over its differently-grained table — is not directly comparable;
the counting rule above is this audit's.)

---

## 6. The two measurements from records (pre-reg §3.1)

Attempted 2026-11-22 from: (a) the HF dataset card of
`armnet/armnetbench_v01_lerobot_so101` (repo id and pinned revision
`1db69c82b589c7172ee31a6b8f7ef58d1fc2c663` per STUDY001_DATASET_REFERENCE.md;
card read in full); (b) the linked benchmark paper, Selvaraj, Uttini &
Kuosmanen, *ArmnetBench v0.1: Parallel Real-World Evaluation of Manipulation
Policies on a Low-Cost Arm Farm*, arXiv:2607.24481v1 (HTML read in full;
§3.1 cell specification, §4 task specification, bill of materials);
(c) the companion dataset paper, *Rollout then Recycle*, arXiv:2611.03168v1
(search-indexed full text); (d) the per-task source dataset card
`villekuosmanen/armnetbench_block_stack` (LeRobot-generated card: structure
metadata only); (e) all local records (scene inventory, v1/v1.1 docs and
verification reports — the only mat/cube statements anywhere in them are the
v1 estimates themselves).

### OI-1 — Mat dimensions: **OPEN ITEM** (not obtained from records)

- Neither the dataset card nor either paper states the WOWROBO mat's
  dimensions, thickness, or product identity. The paper specifies the
  *enclosure* only: "A3 cardboard sheets (297 × 420 mm) bound the workspace"
  (§3.1) — that is the cardboard, **not** the mat, and it is **not**
  substituted for the mat here. (Noted for the record: the v1 mat estimate,
  550 × 400 mm, is larger than an A3 sheet in both axes; the mat in the
  frames lies on the enclosure floor and is a distinct object with printed
  WOWROBO lettering. No inference is drawn from the A3 figure.)
- Parameter stays: **550 × 400 × 3 mm, range ±20%, [MEASURED-EST]** (G8).
- **What would resolve it:** a physical measurement (or product SKU/spec) of
  the mat used in the ArmnetBench single-arm cells.
- **Whose hands:** the ArmnetBench dataset creators / Armnet arm-farm
  operators (the physical setup is theirs, not zk0's). A query to the
  dataset authors, or a future revision of the dataset card / benchmark
  repo stating cell-workspace specifications, would close it. Per the
  governing memo these measurements "move calibration marginally; they do
  not block certification."

### OI-2 — Cube material (block_stack): **OPEN ITEM** (not obtained from records)

- No record states the block_stack cubes' material, product, or mass. The
  dataset card lists the task and instruction only; the paper's task table
  gives instructions only; the per-task source card is LeRobot structure
  metadata only. Local records state only the frames-based ambiguity
  (painted wood vs dense foam, v1 table §2/§9).
- Parameters stay: material **[MEASURED-EST] appearance-only, unresolved**
  (B3); mass **20 g nominal, [MEASURED-EST]**, with the envelope spanning
  the full material-contingent uncertainty **5–35 g** (B2) — the 5 g corner
  is the dense-foam case and is flagged wherever the envelope is consumed.
  Nothing was guessed and nothing was relabeled as measured.
- **What would resolve it:** the cube product/material (or a single cube's
  weighed mass) from the dataset creators.
- **Whose hands:** the ArmnetBench dataset creators / cell operators, as
  OI-1.

---

## 7. v1 table ↔ frozen v1.1 scene discrepancies

Cross-check method: diff of the DGX's frozen `docs/physics-parameters.md`
against the local v1 table; `code/twin_params.py` read in full; XML headers
and geom attributes sampled for one base scene per task (block ep411,
eye ep1609, ring ep7, tool ep622) plus block ep411 variant draws. All other
sampled values (frictions, masses, sizes, mat, timestep in XML) matched the
v1.1-intended values exactly.

| ID | Discrepancy | Direction / status |
|---|---|---|
| D1 | **Timestep: v1 table (and the frozen DGX copy of the parameter doc) says 0.002 s; frozen code and every sampled scene XML use 0.0005 s.** The v1.1 global fix (changelog #14) updated `twin_params.py` and the scenes but **not** the parameter doc's timestep line — the doc on the DGX still reads "Timestep for twin scenes: 0.002 s". | v2 table (G2) takes the frozen-scene value 0.0005 s. Doc-vs-scene inconsistency in the v1.1 record, flagged here; v1.1 files were not modified (read-only brief). |
| D2 | Ring board recess count: v1 table/scene-inventory said **5**; frozen v1.1 docs say **6** (not cut in either version). | v1.1 correction (changelog #10, B3 §3a). v2 takes 6 (R2). |
| D3 | Ring mass derivation: v1 printed volume ≈5.5 cm³ → ≈4–7 g (does not reproduce); v1.1 corrected derivation ≈14.5 cm³ → ≈10 g. | **Value (10 g) and range (4–15 g) unchanged** (R7). Derivation-only fix. |
| D4 | Per-episode colours: v1 rendered fixed colours (block a=purple/b=green; all rings yellow); frozen v1.1 XMLs carry per-episode colours from the pose records (incl. ring ep48 source-data correction yellow→red). | v1.1 change (changelog #8, B3 §1). Visual only; no physics parameter moves (B5, R9). |
| D5 | Tool tray slot recess: v1 table promised a modelled recess; **v1 scenes contained none** (single solid tray box). Frozen v1.1 scenes build the recess (110 × 32 mm, depth 5 mm [ESTIMATE]). | v1 table↔scene defect found by v1.1 verification (F3), fixed in v1.1 (T3, T4). |
| D6 | Cube mass sampling range: v1 table §2 range reads 10–35 g while §9 states the material-contingent uncertainty as 5–35 g; frozen code samples 10–35 g. | Internal v1 inconsistency, unresolved by v1.1. v2 envelope uses the full 5–35 g and flags the 5 g corner (B2, OI-2). |

No other discrepancies were found in the sample. Pose-jitter tiers, variant
seeds and the K=5 machinery are sampling machinery, not physics parameters,
and are inherited unchanged (pre-reg §4).

## 8. Point-only parameters — envelope consequence (standing flag)

The following carry **no sourced range** and therefore sit at their nominal
value at every envelope corner: timestep (G2), contact solver settings (G3),
weld solref/solimp (G4), joint damping (G5), restitution (G6), torsional and
rolling friction (every friction row), tray dimensions (T1, [PROP]), slot footprint
(T3, [DEFAULT]) and depth (T4, [MEASURED-EST — depth: ESTIMATE, never measured]),
board thickness (R1, [MEASURED-EST]). Envelope stability demonstrated
in v2 is stability **conditional on these point values**; a verdict that is
later shown to be sensitive to any of them is a new claim, not a sealed one.
Friction-corner dependence: **all** sliding-friction corners are [DEFAULT]-
derived (§1 rule) — per pre-reg §3.1, any verdict whose stability depends on
a friction corner is flagged as depending on a [DEFAULT] parameter at its
corner.

## 9. Envelope corners

Mechanically derived from the ranges above + pre-reg §3.3 pose σ; enumerated
in machine-readable form in **`envelope-corners-v2.json`** (same folder):
per task — pose axis-extreme offsets at ±σ, sliding-friction low/high
settings for every contact geom in the task, manipulated-object mass
low/high, and the engine swap (MuJoCo ↔ Drake), each as named
single-dimension corners over the nominal baseline, plus the complete
point-only register (§8) the harness must hold fixed — G2, G3, G4, G5, G6,
torsional/rolling friction, T1, T3, T4, and R1, all carried in the JSON's
`point_only_held_fixed_in_all_corners` (corrected 2026-11-22, fix F1).

## 10. Change log — fix pass 2026-11-22 (verification findings F1–F4)

Fix pass responding to `verification-v2-parameter-audit.md` (verdict
CERTIFIED-WITH-FINDINGS, 2026-11-22). Builder fix pass only; delta
verification is separate. No other changes were made.

- **F1 (JSON point-only register incomplete) — FIXED.**
  `envelope-corners-v2.json`: `point_only_held_fixed_in_all_corners`
  now carries, in addition to the existing G2, G3, G4 (weld solref and
  solimp), G5, G6, and torsional/rolling friction entries, the four
  §8 point-only entries it previously omitted — T1 tray dimensions
  (385 x 270 x 45 mm, [PROP]), T3 slot footprint (110 x 32 mm,
  [DEFAULT]), T4 slot depth (5 mm, qualified label — see F2), and R1
  board thickness (15 mm, [MEASURED-EST]). The register now exactly
  matches §8. §9 above was corrected to enumerate that complete
  register, so its claim about the JSON is now true of the fixed JSON.
- **F2 (T4 label overstated its source) — FIXED.** T4's label now reads
  **[MEASURED-EST — depth: ESTIMATE, never measured]** in the §5
  tool_insert table (label cell and the labelling sentence in its
  source text), in the §8 point-only register, and in the JSON
  register entry (`label: "MEASURED-EST — depth: ESTIMATE, never
  measured"`). T4 remains counted in the [MEASURED-EST] family in the
  label counts/totals — category unchanged, qualification added.
- **F3 (v1 §7 grasp thresholds silently dropped) — FIXED.** One
  explicit lineage line added to G4 in §1: v1 §7 grasp thresholds
  (weld attach 50% / release 30% / 3.0 cm) are intentionally NOT
  carried into v2 because the v2 actuated-arm design replaces the
  weld replay device with physical grasp.
- **F4 (platform web/base sub-geometry unstated) — FIXED.** Explicit
  note added to E4 in §3: v1 §3's web/base sub-geometry is not
  separately parameterized because the frozen code defines only
  top plate + height (matches frozen scenes; declared simplification).
- **Re-validation:** `envelope-corners-v2.json` re-parsed after
  editing (valid JSON); point-only register = 12 entries with exactly
  one each of T1, T3, T4, R1 added; per-task corner counts unchanged
  (10 per task: nominal + 4 pose + friction low/high + mass low/high
  + engine_drake). No values, ranges, labels (other than T4's
  qualification), corners, or open items were changed.
- **Files changed:** `parameter-audit-v2.md` (G4, E4, T4, §8, §9,
  this §10) and `envelope-corners-v2.json` (point-only register only).
