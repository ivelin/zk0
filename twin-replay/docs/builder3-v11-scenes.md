# Twin-Suite v1.1 — Builder 3 Report: Scene Fidelity + Docs

Builder 3 (scene fidelity + docs), 2026-11-22 (DGX clock, CT). CPU-only,
`nice -n 10` throughout; no GPU use — DGX training (spec_exp chain) was
running at 96% GPU and was not disturbed. All changes logged
contemporaneously in `validation/param-changelog.md` entries #8–#10
(timestamped 2026-11-22 18:56 CT, the time of the edits).
Scope kept as briefed: edits are confined to the colour / tray-geometry
code paths of `code/build_scenes.py`. `param_table_hash()` and variant
sampling (`make_draws`) were NOT touched.

## 1. Colour variants wired into the scene XMLs — DONE

**Mechanism (global, data-driven, no per-episode code):**
`load_episode_poses()` now surfaces `poses["colors"] = {body: colour name}`
from each episode's own Builder B record — for block_stack from the object
keys (`cube_<colour>`, mapped to bodies by role: manipulated → `cube_a`,
base → `cube_b`, with key-order fallback for the unroled episodes, matching
the pose assignment exactly); for ring_insert from the ring object's
`"color"` field. `build()` renders through a `COLOR_RGBA` table. Purple /
green / yellow keep the exact v1 rgba values; orange (`0.90 0.45 0.10 1`)
and red (`0.80 0.15 0.15 1`) are visual-only rendering approximations for
the recorded colour names (mass/friction are colour-independent). Episodes
whose records carry no colour data fall back to the v1 colours.

**One data correction at source:** `scenes/ring_insert__ep48.json` said
`"color": "yellow"`, but the episode's own frame
(`ring_insert__ep48/top_first.png`) shows a red ring — as recorded in
scene-inventory.md §5 and settled by verification finding F2. The JSON now
says `"red"`, with the correction and its sources noted in the file's
`notes` field and in changelog #8. No other JSON colour label was
frame-contradicted (verifier census).

**Verification (independent recomputation from the JSONs vs the rebuilt
XMLs): 0 colour errors across all 40 block/ring episodes.** Rendered
colours changed vs v1 in 29 body-instances across 20 episodes:

- block_stack — 16 episodes, 25 body-instances:
  - Purple/green swapped between bodies (v1 always rendered a=purple,
    b=green): ep411, ep479, ep513, ep530, ep531 (both bodies corrected).
  - Orange cube rendered green in v1 → orange: ep489 (cube_b), ep498
    (cube_b), ep499 (cube_b), ep525 (cube_b), ep526 (cube_b), ep548
    (cube_b), ep549 (cube_b).
  - Orange manipulated cube + purple base (both bodies corrected):
    ep527, ep545, ep546, ep557.
  - Unchanged (v1 happened to match the record): ep453, ep476, ep514,
    ep539.
- ring_insert — 4 episodes: ep48, ep188, ep240, ep242 now render red
  (v1 rendered every ring yellow). The other 16 episodes are recorded
  yellow and are unchanged.

This matches the verifier's census (11/20 block episodes carry an orange
cube; ≥3–4/20 ring episodes are red) and extends it: the verifier counted
per-episode colour identity, while the swap cases above show v1 also put
the right colours on the wrong bodies in 9 block episodes.

## 2. Tool tray slot recess — DONE

v1 emitted the tray as one solid box (verification finding F3, against
physics-parameters.md §5's promise of a "4-wall + floor inset").
`build()`'s tool_insert path now builds the tray globally as:

- a full-footprint slab (`tray_geom`, 385×270 mm) rising to 40 mm, whose
  top face is the recess floor;
- a red `slot_floor` plate (1 mm) inset in the opening — red recess
  backing per scene-inventory.md §6 — floor top at z = 41.0 mm;
- four rim boxes (`tray_rim_w/e/s/n`) filling the top layer around the
  opening, tray top at 45 mm as before.

**Dimensions and sources:**
- Recess footprint 110 × 32 mm — `twin_params.SLOT_SIZE`, the pre-existing
  recorded LOCAL-CHOICE slot size (not invented here).
- Recess centre — the recorded `target_slot` fixture pose, in tray-local
  coordinates (−0.0552, +0.0836) m after the suite's standard FK frame
  correction; the opening lies fully inside the tray footprint.
- Recess depth 5 mm (rim depth; effective pocket depth 4 mm above the
  floor plate) — **[ESTIMATE, never measured]**: set to the loose tool's
  radius (`TOOL_R` = 5 mm) so a seated tool's centre rests at tray-top
  height, matching the recorded `tool_final` z = 0.045 m. Labelled as an
  estimate in the code comment, the parameter table, and changelog #9.
- Shape: the recorded recess is cross/T-shaped; the parameter table's own
  declared model is a box recess, so a box recess is what is built (the
  table's declared simplification, now actually implemented).

**Functional check (MuJoCo 3.8.1, CPU):** ep622, ep489 (recoloured) and
ep240 (recoloured) scenes load and step 1.0 s cleanly. A loose tool
dropped into the ep622 slot settles with centre z = **0.0459 m**
(predicted: floor top 0.0410 + tool radius 0.005 = 0.0460) and 0.0 mm
planar drift — it seats in the recess, 4.1 mm below the tray top. A
control drop on a flat part of the same tray settles at z = 0.0499 m
(tray top + radius = 0.0500), confirming the contrast is the recess, not
a global tray change. All 20 tool_insert XMLs (×5 variants) contain the
recess geometry. Replay criteria inputs are unchanged (`meta["slot_pos"]`
untouched; the recess is centred exactly on it).

## 3. Doc corrections — DONE

(a) **Ring board recess count 5 → 6**, in both Builder A docs that said 5:
`docs/scene-inventory.md` §5 (the "Recesses: 5 … exactly 5 … corrects the
coordinator's scouting note of 6" passage rewritten with the settling
evidence: ep7 top_final shows six empty recesses; ep33 shows five empty
plus the ring seated in a sixth) and `docs/physics-parameters.md` §4 board
row ("6 recesses NOT cut"). A grep across the docs confirms no other
"5 recesses" statement remains in Builder A's docs (the scene JSONs and
pose-estimation.md already carried 6). Settled source:
verification-objects-assets.md finding F1.

(b) **Ring-mass derivation arithmetic** in `docs/physics-parameters.md`
§4: the printed derivation (volume ≈ 5.5 cm³ × 0.7–1.2 g/cm³ ≈ 4–7 g) does
not reproduce. Corrected to: volume = π(R²−r²)·t = π×(22.5²−11²) mm² ×
12 mm ≈ **14.5 cm³**; × ≈0.7 g/cm³ ≈ **10 g**, with the 4–15 g range
spanning densities ≈0.3–1.0 g/cm³ on the correct volume. **The 10 g value
and the 4–15 g range are unchanged** — only the derivation arithmetic was
wrong (verification finding F4). The correction is noted inline in the
table so the change is visible in the record.

Also added to the §5 target-slot row: the as-built footprint/depth and
their sources (see §2), and a note that the recess is now implemented.

Note: `physics-parameters.md` is an input to `param_table_hash()`, so the
regenerated scene headers carry a new param_hash value. The hash function
itself is untouched (another builder owns its extension); the change in
its output is the direct, intended consequence of correcting the doc it
hashes, and is flagged in changelog #10.

## 4. Determinism — VERIFIED

Full regeneration (`python3 build_scenes.py`, niced, CPU) run twice after
the changes: **500 variant scenes built, 0 skipped, both runs.** The
aggregate SHA-256 over all 600 scene XMLs (500 variants + 100 base copies,
`sha256sum *.xml | sha256sum`) is identical across the two runs:

`85d3f51f61af2451578068d6f42aa15ffa70026ab3d455cee61d864ee2703391`

This covers the required sample (recoloured episodes ep489/ep240 and the
tool tray ep622) and every other scene — regeneration is byte-identical
given the same code + seed, as in v1.

## 5. Files changed

- `code/build_scenes.py` — colour table + `poses["colors"]` plumbing
  (colour code paths) and the tray recess construction (tool_insert
  geometry path) only. v1 backup at `/tmp/build_scenes.py.v1.bak` on the
  DGX (ephemeral; the v1 record remains in git / the v1 docs).
- `scenes/ring_insert__ep48.json` — colour data correction yellow → red
  (frame-sourced, noted in-file).
- `scenes/*.xml` — all 600 regenerated (colours + tray recess; header
  param_hash updated via the corrected parameter doc).
- `scenes/*_variants.json` — regenerated by the builder as usual (draws
  unchanged; variant sampling untouched).
- `docs/scene-inventory.md` — recess count correction (§5).
- `docs/physics-parameters.md` — recess count (§4 board row), ring-mass
  derivation (§4 ring row), target-slot as-built note (§5).
- `validation/param-changelog.md` — entries #8, #9, #10, contemporaneous.
- `docs/builder3-v11-scenes.md` — this report.
- Logs: `logs/builder3-regen1.log`, `logs/builder3-regen2.log`.

## 6. Blocked / open items

- Nothing blocked. Caveats for the verifier: (i) recess depth is an
  explicitly labelled estimate, not a measurement — the one dimension in
  this change set with no recorded source; (ii) orange/red rgba values are
  rendering approximations of the recorded colour names; (iii) no
  re-validation was run — these are scene-content changes under the v1.1
  minor-version rules, and re-validation + fresh independent verification
  are required before any certification claim (per the completion
  report's versioning rules). That verification is a separate context's
  job, not this builder's.
