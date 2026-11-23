# Verification v1.1 — Objects & Assets (V-Objects+Assets, independent)

Verifier: fresh independent context. Re-derived from raw artifacts only
(scene JSONs, built XMLs, frames viewed directly, twin_params.py,
parameter table, MuJoCo re-run). Builder 3's report was treated as
claims to check, not evidence. CPU-only, niced, DGX training undisturbed.
Scope: scene CONTENT (dimensions, colours, tray geometry, materials).
Cable-clip placement/geometry is PROVISIONAL per brief (cable-only
scoring fix + regeneration pending).

## VERDICT: CERTIFIED-WITH-FINDINGS

All Builder 3 headline claims reproduce independently. One v1 finding
(F5, asset-inventory era label) remains uncorrected; two residual
notes below. No new unsourced dimension/material was found.

## 1. COLORS — PASS (Builder claim verified)

Independent recomputation: expected colour per body derived from each
episode's own scene JSON (block_stack: object keys cube_<colour>,
role-mapped manipulated→cube_a / base→cube_b; unroled episodes fall
back to key order, which I verified matches the pose-assignment code
path in load_episode_poses — the same fallback the builder uses, so
colour and pose cannot diverge) and ring_insert from objects.ring.color;
compared against rgba in the built base XMLs.

- block_stack: 11 roled episodes × 2 bodies checked directly + 9
  unroled episodes checked as colour-set + body assignment vs key
  order — 0 errors, all 20 episodes.
- ring_insert: 20/20 episodes, 0 errors (ring rgba matches JSON colour).
- Changed vs v1 hard-coding (cube_a=purple/cube_b=green, ring=yellow):
  I recount **29 body-instances across 20 episodes** — exactly Builder
  3's claim (25 block + 4 ring: ep48, ep188, ep240, ep242 red).
- ep48 correction VERIFIED AGAINST FRAME: I pulled
  frames/ring_insert__ep48/top_first.png and viewed it directly — the
  ring seated in the board's bottom recess is unambiguously RED.
  JSON now says "red" with the correction noted in-file. Correct.
- Orange/red rgba labelled as approximations: YES in code
  (build_scenes.py COLOR_RGBA comment: "visual-only rendering
  approximations for the recorded colour names; mass/friction are
  colour-independent") and in Builder 3's report §1/§6 and changelog
  entry #8 (Builder 3). The physics parameter table does not carry
  colour rgba at all, so no misleading label exists there. PASS.

## 2. TRAY RECESS — PASS (settle claim reproduced exactly)

- Presence: ALL 120 tool_insert XMLs (20 base + 100 variants) contain
  tray_geom slab + slot_floor + tray_rim_w/e/s/n. No exceptions.
- Geometry (ep622 base XML, read raw): slab half-height 0.020 → top at
  40 mm (= 45 − 5 mm depth); slot_floor size 0.055 × 0.016 half →
  **110 × 32 mm footprint**, centre z 0.0405, half-thick 0.0005 → floor
  top at 41.0 mm; rims centred z 0.0425, half 0.0025 → span 40–45 mm,
  tray top 45 mm unchanged. Footprint = twin_params.SLOT_SIZE
  (0.110, 0.032), tagged LOCAL-CHOICE in twin_params.py — a declared
  local choice, not dressed as a measurement. Recess centre in ep622:
  (−0.05516, +0.08360) tray-local — matches the recorded target_slot
  fixture pose as claimed (−0.0552, +0.0836). All 20 base XMLs share
  the identical slot_floor size.
- Shape note: built recess is a BOX, while the recorded slot is
  cross/T-shaped; the parameter table's own declared model is a box
  recess ("shallow box recess modelled as a 4-wall + floor inset"),
  so the build implements the declared simplification. Honest.
- Settle RE-RUN by me (MuJoCo 4.0.1, sim venv, ep622, tool dropped at
  slot centre vs flat tray control, 1500 steps):
  **slot z = 0.0459 m, flat z = 0.0499 m** — byte-for-byte the Builder's
  claimed values (0.0459 vs 0.0499). Planar drift 0.0 mm in slot.
  Claim reproduced, not merely plausible.
- 5 mm depth [ESTIMATE] labelling: present in ALL THREE required
  places — code (build_scenes.py comment "Depth = 5 mm [ESTIMATE, not
  measured…]"), parameter table (§5 target-slot row: "recess depth
  5 mm [ESTIMATE -- never measured; set to the loose tool radius…]"),
  and changelog (Builder 3 entry #9: "Depth 5 mm is an ESTIMATE, never
  measured"). PASS.

## 3. RECESS COUNT — PASS

grep "5 recess" across docs/ + scene JSONs: no stale claim remains.
Remaining hits are historical references only (Builder 3 report §3a
describing the fix; changelog "5 -> 6"). scene-inventory.md §5 now
reads "Recesses: 6" with settling evidence (ep7 six empty; ep33 five
empty + ring in sixth — and my own ep48 frame view independently
shows 5 empty recesses + red ring in a 6th = 6). physics-parameters.md
§4 board row: "6 recesses NOT cut (flat top)". Both docs corrected
as claimed.

## 4. RING MASS — PASS (arithmetic re-derived)

My own computation: π(R²−r²)·t = π×(22.5²−11²) mm² × 12 mm
= π × 385.25 × 12 = 14,524 mm³ ≈ **14.5 cm³**. × 0.7 g/cm³ = 10.17 g
≈ **10 g**. Range at densities 0.3–1.0 g/cm³: 4.36–14.52 g ≈ the
stated **4–15 g**. The corrected derivation in physics-parameters.md
§4 reproduces exactly; value and range unchanged from v1, as claimed.
Built XML check (ep7): 12 ring segments × 0.8333 g = 10.0 g total;
capsule radius 0.00575 → 11.5 mm thickness (table 12 mm ±, within
tolerance); peg geom Ø16 mm; board Ø170 × 15 mm. Consistent.

## 5. V1 FINDINGS RE-CHECK

- F1 (recess count): FIXED in both docs (§3 above).
- F2 (XML colours + ep48 JSON): FIXED (§1 above).
- F3 (missing slot recess): FIXED (§2 above). Residual: the ≈11
  seated tools still have no geometry — but the table declares them
  "merged with tray geom (visual meshes optional)", they never move
  in any episode, and this was explicit in v1; not a new defect.
- F4 (ring-mass derivation): FIXED (§4 above).
- F5 (asset-inventory "SIM-1/SIM-2 era sim validation" label):
  **NOT CORRECTED — STILL OVERSTATED.** asset-inventory.md line 8
  (arm-model row, testing-evidence column) still reads "SIM-1/SIM-2
  era sim validation per project record" verbatim. Builder 3 did not
  claim to fix it (its doc scope was recess count + ring mass), so
  this is a carried-over v1 finding, not a Builder 3 misclaim — but
  it remains uncorroborated provenance dressing and must be cited
  precisely or dropped before the inventory is quoted as baseline.
- Dimensions sample vs parameter table (my audit of built XMLs, all
  base episodes, not a sample): cube half-size 0.015 (30 mm), mass
  0.020 kg — all 20; board 0.085/0.0075 (Ø170×15); carton half
  0.045/0.0175/0.0175 (90×35×35); platform top 200×160×4 mm, basket
  floor 260×180, walls 70 mm high × 3 mm — all match table; cable
  block half 0.055/0.035/0.0275 (110×70×55), USB half
  0.0125/0.006/0.004 (25×12×8), 24 cable bodies; tray 385×270×45.
  No dimension discrepancy found in v1.1 builds.
- Parameter-source tags for Builder-3-touched items: SLOT_SIZE =
  LOCAL-CHOICE (honest), depth = [ESTIMATE] (honest), colours =
  recorded names + labelled rgba approximations (honest), ring mass
  = [MEASURED-EST] with now-correct derivation (honest). No tag was
  upgraded beyond its evidence.

## 6. UNSOURCED OBJECTS/ASSETS — none found

Every geom class in the built scenes traces to twin_params.py values
that trace to the parameter table with a source tag ([MEASURED-EST] /
[PROP] / [DEFAULT] / LOCAL-CHOICE / [ESTIMATE]). The one dimension
with no recorded source — recess depth — is the one explicitly
labelled [ESTIMATE] everywhere. No silent invention found.

## Findings

- V11-OA-1 (minor, documentation, carried from v1 F5): asset-inventory
  "SIM-1/SIM-2 era" label still uncorroborated and uncorrected.
- V11-OA-2 (note, provisional per brief): cable_clip content was not
  re-certified beyond dimension spot-checks (block/USB/segment count
  match table); cable geometry is provisional pending the pending
  cable-only scoring fix + regeneration.
- V11-OA-3 (note): seated-tool geometry absent in tool trays —
  declared simplification in the table ("visual meshes optional"),
  no physics claim rests on it.

Verdict stands: **CERTIFIED-WITH-FINDINGS** for the objects+assets
class — colours, tray recess, recess count, and ring-mass derivation
are all independently reproduced; the sole open item is the F5
provenance label in the asset inventory.
