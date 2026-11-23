# Verification — Objects & Assets (V-OBJECTS+ASSETS, independent)

Verifier: independent context, raw artifacts only. `docs/twin-validation-report.md`
and builder narrative conclusions were NOT read. Judged from: frames in
`frames/`, scene JSONs/XML in `scenes/`, `code/twin_params.py`, the three
Phase 1–2 docs under audit, and the sim source workdirs.

## VERDICT: CERTIFIED-WITH-FINDINGS

Geometry, counts, identities and parameter sourcing are substantially honest
and reproducible. Findings F1–F5 below are real but bounded: two are
documentation errors (one already corrected downstream in the scene JSONs),
two are visual-only colour defects in the generated XML, one is a missing
geometry the physics table promised. None changes a mass, friction or
dimension used by the physics. No checkpoint inference is made.

## Sample (verifier-chosen, 17 episodes)

- block_stack: ep411, ep453 (JSON/XML only), ep476 (JSON/XML), ep489, ep498
- ring_insert: ep7, ep11, ep30, ep33, ep48 (+ ep188/ep240/ep242 JSON/XML colour audit)
- eye_drops_to_basket: ep1609, ep1639
- tool_insert: ep622, ep684
- cable_clip: ep1896, ep1959
Frames viewed directly: ring ep7/ep30/ep33/ep48 (top/front), block ep411/ep489/
ep498, eye ep1609/ep1639, tool ep622/ep684, cable ep1896/ep1959.
All 100 scene JSONs and representative XML of every task were audited
programmatically (object keys, colours, geom sizes, masses, frictions).

## 1. Object fidelity

### SETTLED: ring board recess count = 6 (not 5)

- `ring_insert__ep7/top_final.png`: six empty circular recesses are directly
  countable around the peg (ring seated on peg at centre).
- `ring_insert__ep30/top_final.png`: same, six.
- `ring_insert__ep33/top_first.png` and `ring_insert__ep48/top_first.png`:
  five empty recesses + the ring seated in a sixth = 6 total.
- `docs/scene-inventory.md` §5 ("Recesses: 5 … corrects the coordinator's
  scouting note of 6") is WRONG. `docs/physics-parameters.md` §4 repeats "5
  recesses". The per-episode scene JSONs carry the correction ("6 recesses
  (corrects scene-inventory count of 5; counted in ring_insert__ep33 frames)")
  and are RIGHT. Physics impact: none today — recesses are declared NOT cut
  (flat board top in every ring XML, verified) — but the inventory and the
  parameter table must be corrected before they are quoted as the baseline,
  and any future recess geometry must use 6. **Finding F1.**

### SETTLED: colour variants

- Cubes: purple in all 20 episodes + green in 9 / orange in 11 (scene JSON
  census). Frames confirm both variants: ep411 green+purple, ep489/ep498
  orange+purple. Inventory §3 (purple always; green 411/453/476/479, orange
  489/498) matches the JSON census for every episode it names. The colour-variant
  claim is TRUE at JSON level.
- Ring: frames confirm yellow (ep7, ep11, ep30, ep33) and red (ep48 — clearly
  red in `ring_insert__ep48/top_first.png`, as inventory §5 says). BUT the
  scene JSON for ep48 says `"color": "yellow"` — wrong against its own frame.
  JSONs label ep188/ep240/ep242 red, ep48 yellow; inventory says red ep48.
  At least one JSON colour label (ep48) is frame-contradicted. **Finding F2 (part).**
- XML colour fidelity FAILS for both variant objects: every block_stack XML
  renders cube_b with the same green rgba `0.25 0.65 0.30` — including all 11
  orange episodes (ep489, ep498, ep499, ep525–527, ep545–549, ep557) — and
  every ring_insert XML renders the ring yellow `0.95 0.80 0.15`, including
  the red episodes. Colours are visual-only (mass/friction identical), so
  physics is unaffected, but rendered twins and any blinded visual judging
  would show the wrong colour in 11/20 block_stack and ≥3–4/20 ring episodes.
  **Finding F2.**

### Per-task identity / count / geometry (XML vs table vs frames)

- block_stack: 2 cubes/episode in all 20 JSONs/XMLs; XML box half-size
  0.015 m = 30 mm cube, mass 0.020 kg, friction 0.6 — all match the table.
  Final frames (ep498) show the stack outcome. Cubes in frames show fine
  horizontal layer lines (3D-printed appearance) — the table's "painted wood
  or dense foam — appearance ambiguous" misses this candidate material; its
  own flag that cube mass (5–35 g) is the least-certain value stands.
- eye_drops_to_basket: carton XML 90×35×35 mm, 30 g, friction 0.5 = table.
  Platform XML: top plate 200×160×4 mm at z≈90 mm, web + base = table.
  Basket XML: 260×180 mm footprint, 70 mm walls, 3 mm wall thickness = table
  (ribs not modelled, declared). Frames ep1609 (label face up) / ep1639
  (blue face up) confirm identity, the lying-on-side start, basket left /
  platform centre-right, and tape strips. MATCH.
- ring_insert: board XML cylinder Ø170×15 mm, peg Ø16 mm standing 70 mm
  above board top (geom spans z 15→85 mm), ring = 12-capsule composite with
  outer Ø45 / inner Ø22 / thickness 11.5 mm, total mass 0.010 kg — all match
  the table. MATCH except recess count (F1) and colour (F2).
- tool_insert: tray XML 385×270×45 mm, loose tool capsule 100 mm long,
  Ø10 mm, 70 g — match the table. Frames ep622 confirm the "Premier by
  SEALEY" tray, labelled extension-bar slots, exactly one empty cross/T slot,
  one loose chrome bar on the mat; ep684 final shows the slot filled.
  **Finding F3: the target slot geometry promised in physics-parameters §5
  ("shallow box recess modelled as a 4-wall + floor inset … at the empty
  slot's location") does NOT exist in any tool_insert XML** (grep across all
  20 base XMLs: no slot geom; the tray is one solid box). The scene JSONs do
  record a `target_slot` fixture pose, so the slot is known but unmodelled —
  the loose tool's final pose rests on the tray top surface, not in a recess.
  The ≈11 seated tools are likewise absent as geometry (table says "merged
  with tray geom (visual meshes optional)" — in fact there is nothing).
- cable_clip: clip block XML 110×70×55 mm + two black jaw boxes (channel
  16 mm between jaw inner faces vs table's "≈8 mm channel width" — jaw boxes
  are 8 mm apart edge-to-edge in y at ±8 mm centres with 4 mm half-width,
  i.e. a 8 mm gap; consistent on a centre-vs-edge reading, noted not flagged).
  Cable: 24 capsule segments × 30 mm, Ø5 mm, 1.875 g each = 45 g total, ball
  joints damping 0.2, friction 0.8 — exactly the table. USB box 25×12×8 mm,
  6 g = table. Frames ep1896/ep1959 confirm white block + black clip, coiled
  black cable with moulded connector, 5 tape strips, block re-placed per
  episode. MATCH.

### Independent dimension re-estimates (verifier's own, from frames)

Method: bounding-box pixel ratios within one frame (1024×576 top views;
x and y ratios averaged, perspective/anisotropy noted, ±~20%):

1. Ring outer Ø vs board Ø (ep7 top_final): ring/board ≈ 0.22 (x) / 0.29 (y),
   mean ≈ 0.25 → ≈43 mm at board 170 mm. Table: 45 ±10 mm. WITHIN.
2. Recess Ø vs board Ø (ep7 top_final): ≈0.25 → ≈42 mm; consistent with the
   ring (outer 45 mm) seating in a recess, as ep33/ep48 first frames show.
3. Peg height vs board Ø (ep7 front_first): peg above board ≈ 0.38 × board
   width in image → ≈60–70 mm. Table: 70 ±15 mm. WITHIN.
4. Cube edge vs mat (ep489 top_first): cube ≈ 0.055–0.07 of mat width
   (400 mm) → ≈25–35 mm after perspective. Table: 30 ±10 mm. WITHIN
   (corroborated by Builder B's homography cross-check in
   `scenes/scale_crosschecks.json` as cited in pose-estimation.md: cube
   median 37.5 mm, ring median 48.4 mm — both inside the table ranges).
5. Carton length vs platform top (ep1609/ep1639 top_first): carton ≈ 0.45 ×
   platform-top long axis (200 mm) → ≈90 mm. Table: 90 ±20%. WITHIN.
6. Loose tool length vs the tray's labelled 250 mm bar (ep622 top_first):
   pixel diagonals 136 px vs 359 px → ≈95 mm raw, ≈75–95 mm after
   foreground-perspective correction. Table: 100 ±25 mm. WITHIN (lower half).

No dimension claim was found outside its own stated uncertainty.

## 2. Parameter sources (8 values audited)

| # | Value | Tag | Verdict |
|---|---|---|---|
| 1 | Cube mass 20 g (27 cm³ × 0.7 g/cm³) | MEASURED-EST | HONEST. Arithmetic reproduces (18.9 g ≈ 20 g); material ambiguity and 5–35 g range declared in the table itself. |
| 2 | Mat 550×400 mm (1.45 × ~380 mm tray) | MEASURED-EST | HONEST. Method stated and roughly reproducible from ep622 top_first; ±20% declared. |
| 3 | Tray 385×270×45 mm, "Sealey Premier EVA tray class" | PROP | HONEST. Prop is real and frame-consistent ("Premier by SEALEY" legible ep622; Sealey Premier EVA module trays are a real product family in ~395×285 mm class sizes; extension bars in chrome-vanadium steel as claimed). Cross-checked MEASURED-EST as stated. |
| 4 | Loose tool mass 70 g (steel 7.85 g/cm³ × ≈9 cm³) | MEASURED-EST | HONEST. Capsule Ø10×100 mm ≈ 8–9 cm³ → ≈65–70 g; independently plausible (a Sealey Premier 75 mm 3/8" extension bar lists 0.07 kg nett). |
| 5 | Cable 72 cm / 45 g (24 × 30 mm; USB cable 40–60 g) | MEASURED-EST (+PROP cross-check) | HONEST. XML reproduces 24×30 mm and 45 g exactly; length inside the inventory's 60–90 cm frame estimate. |
| 6 | Ring mass 10 g, derivation "volume ≈5.5 cm³ × 0.7–1.2" | MEASURED-EST | **OVERSTATED DERIVATION (Finding F4).** The stated volume does not reproduce: π(R²−r²)·t for R=22.5, r=11, t=12 mm ≈ 14.5 cm³ (torus formula ≈11 cm³), not 5.5 cm³. The 10 g value itself is plausible on the correct volume (≈0.7 g/cm³ × 14.5 cm³ ≈ 10 g) and inside the declared 4–15 g range, so the value stands; the printed derivation is wrong and should be fixed. |
| 7 | Mat friction 0.9 / cube friction 0.6 etc. | DEFAULT | HONEST. All friction is openly declared unmeasured literature-default (§8 of the table admits this plainly); values sit inside the cited literature ranges for rubber mat / wood-on-mat. Torsional/rolling 0.005/0.0001 are standard MuJoCo-style defaults. |
| 8 | Gravity −9.81, timestep 0.002 s | DEFAULT | HONEST. 0.002 s is MuJoCo's actual default timestep. XMLs carry no contrary values. |
| + | Eye-drops carton mass 30 g | PROP | HONEST. Retail carton + 10–15 ml bottle at 25–40 g is plausible; volume cross-check declared. |
| + | Clip channel ≈8 mm | (declared modelling choice) | HONEST. Explicitly labelled a modelling choice with the unmodelled spring force declared — not dressed up as a measurement. |

No tag was found claiming measurement for a value that is plainly invented.
The one overstated item is F4's derivation arithmetic, not the value.

## 3. Asset provenance (asset-inventory.md, entry by entry)

- **SO-101 arm model** (`zk0-sim-v01-20261115/sim/model/so101_new_calib.xml`
  + 13 STLs): EXISTS at the claimed path (13 STL files counted in both the
  source dir and the twin `scenes/assets/` copy; sampled STL byte-identical,
  `diff` clean). Twin XMLs include the source file by absolute path (not a
  modified copy) — "REUSED as-is (included, unmodified)" is TRUE. Apache-2.0
  LICENSE present in `sim/model/`. Testing evidence: `sim/zk0_sim_pick_place.mp4`
  (1.8 MB, 2026-11-15) exists — the v0 pick-and-place demo did run;
  `zk0_sim_demo.py` contains the FK-solved HOVER/GRASP poses and the
  kinematic-carry Demo class exactly as described; `sim/README.md`
  corroborates the demo's scope. **Finding F5 (minor): the inventory's
  "SIM-1/SIM-2 era sim validation per project record" could NOT be
  corroborated — no record using the labels SIM-1/SIM-2 was found in the sim
  workdirs searched** (`zk0-sim-v01-20261115`, `sim-experience-audit-20261115`,
  whose run.log does show a SO-101 scene loading/stepping/rendering with
  RESULT: SUCCESS). The testing claim is substantively true; the SIM-1/SIM-2
  label is uncorroborated provenance dressing and should be cited precisely
  or dropped.
- **Alt arm copies** (v0 copy, eval2 `assets/SO101/`, menagerie copies):
  eval2 dir exists with the claimed variants (new/old calib, camera variant,
  `joints_properties.xml`); menagerie dir exists in v01. "Not used" is
  consistent with the twin XML include path. CORROBORATED.
- **Tabletop scene pattern** (`scene_tabletop.xml`): EXISTS; contains the
  40 mm cube (half-size 0.02, mass 0.04), table top at z=0, arm at origin —
  exactly as claimed. Twin scenes do follow the same convention (arm include
  at origin, mat top at z=0). CORROBORATED.
- **Cube pattern-reuse / size NEW (30 mm)**: TRUE — demo cube is 40 mm,
  twin cubes 30 mm, same box+free-joint mechanism.
- **Kinematic-carry / gripper convention**: `zk0_sim_demo.py` defines
  GRIPPER_OPEN=1.45 / CLOSED=0.10 rad verbatim. CORROBORATED.
- **Grasp/BAM-era dirs** (`zk0-simready-shelf-20261115`,
  `sim-experience-audit-20261115`, `sim-stack-check-20261115`): checked —
  no .xml scene files in shelf/audit dirs, consistent with "nothing reusable;
  absence on record". CORROBORATED.
- **Repo clone** (`zk0-study001-fl-20261121/train-fl-simulation.sh`): EXISTS;
  it is a Flower federation script, not a physics scene. CORROBORATED.
- **NEW items** (mat, enclosure, fixtures, carton/ring/tool/cable): no
  pre-existing assets for these props were found in the surveyed sim dirs,
  consistent with the NEW labelling; their XML parameters trace to the
  parameter table (verified in §1 above, except F3's missing slot).
- **Cameras/rendering** (MuJoCo 4.0.1, EGL): setup.log in the sim workdir
  records "mujoco installed: 4.0.1" and a successful render. CORROBORATED.

## Findings summary

- F1 (documentation, moderate): recess count is 6; scene-inventory §5 and
  physics-parameters §4 say 5. Scene JSONs already carry the correct count.
  Fix the two docs before the suite is quoted/archived as a baseline.
- F2 (visual fidelity, moderate): XML colours ignore the recorded variants —
  orange cubes render green (11/20 block_stack), red rings render yellow
  (≥3/20 ring_insert); ring ep48's JSON colour label (yellow) contradicts
  its frame (red). Physics unaffected; rendered/blinded judging would be.
- F3 (geometry, moderate): tool_insert target-slot recess promised in
  physics-parameters §5 is absent from all tool XMLs; tray is a solid box
  and seated tools have no geometry at all.
- F4 (derivation, minor): ring-mass volume derivation (5.5 cm³) does not
  reproduce (≈14.5 cm³ by the stated formula); the 10 g value remains
  plausible on the correct volume.
- F5 (provenance wording, minor): "SIM-1/SIM-2 era validation" label
  uncorroborated in the sim workdirs; the underlying demo/audit evidence
  is real (demo MP4, audit SUCCESS log).

Verdict stands at CERTIFIED-WITH-FINDINGS: object identity, counts (as built),
dimensions, masses and friction sourcing are honest within their stated
uncertainties, and reused-asset provenance is substantively true — subject to
F1–F5 being corrected in the record (and F2/F3 in the scenes if rendered or
slot-contact fidelity is claimed).
