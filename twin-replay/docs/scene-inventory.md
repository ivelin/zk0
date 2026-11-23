# Study 001 Twin-Replay — Scene Inventory (Phase 1, Builder A, 2026-11-22)

Evidence base: first/final frames extracted by `code/extract_frames.py` into
`frames/<task>__ep<id>/{front,top,wrist}_{first,final}.png` (600/600, 0 failures;
see `logs/extract_frames.log`). Cross-check vs the eval's curated `sample/`
frames: PASS (see §6). Every claim below cites frame files in `frames/`.
Episodes viewed in detail (≥6/task): block_stack 411, 453, 476, 479, 489, 498;
eye_drops_to_basket 1609, 1618, 1630, 1639, 1643, 1724;
ring_insert 7, 11, 30, 31, 33, 48; tool_insert 622, 629, 630, 683, 684, 697;
cable_clip 1896, 1903, 1905, 1939, 1959, 1960.

Recorded-outcome ground truth (coordinator addendum, from
`run/eval-traj-20261122/split/split.json`, which outranks frame inference):
block_stack, eye_drops_to_basket, ring_insert, tool_insert — all 20 episodes
`recorded_success=1` (pool_success_only=true); cable_clip — all 20 episodes
`recorded_success=0` (pool_success_only=false). Final frames corroborate this
in every episode viewed (stacked cubes, carton in basket, ring on peg, tool in
tray; cable never seated in clip). Split metadata also lists a
`source_policy_type` per episode (pi0 / pi0.5 / grootn1.7 / act / diffusion /
molmoact2); the recordings are real-arm executions.

## 1. Common workspace (all 5 tasks)

- **Enclosure**: open-top box of tan/kraft cardboard — floor plus four low
  walls; the back wall is lower/folded in front views and room elements
  (ceiling duct, monitor) are visible above it. Evidence:
  `eye_drops_to_basket__ep1630/front_first.png`,
  `cable_clip__ep1896/front_first.png`, all `top_first.png`.
- **Mat**: dark slate-blue/grey flexible mat lying on the enclosure floor,
  white border stripe, printed text "WOWROBO" across the middle (white
  letters for "WOW", green gradient for "ROBO"). Evidence: every
  `top_first.png`, e.g. `block_stack__ep453/top_first.png`.
  Size estimate: ≈55 × 40 cm ±20% [pixel-ratio estimate: in
  `tool_insert__ep622/top_first.png` the mat's long axis spans ≈1.45× the
  Sealey tray's long axis; tray ≈38 cm class — see §5; perspective tilt in
  the top view limits precision]. Mat thickness ≈2–3 mm (edge visible in
  `eye_drops_to_basket__ep1630/front_first.png`).
- **Arm**: black SO-101 arm enters from the near (bottom-of-top-view) edge,
  base outside or at the mat's near edge, roughly centred left-of-middle;
  the arm, its black cable loop and red/white/black servo wiring are in
  frame in nearly every view. Evidence: all `top_first.png`.
- **White tape strips**: in eye_drops_to_basket and cable_clip scenes, 4–6
  short white rectangular strips are stuck on the mat around the fixtures
  (fiducial/position markers). Evidence:
  `eye_drops_to_basket__ep1639/top_first.png`,
  `cable_clip__ep1959/top_first.png`. Absent in block_stack / ring_insert /
  tool_insert top views viewed.

## 2. Camera viewpoints (qualitative)

- **front** (1024×576): LOW camera at roughly mat level, outside the near
  edge, looking slightly up/across the mat; the arm is large in the left
  foreground and the back cardboard wall fills the background. Evidence:
  `block_stack__ep411/front_first.png`, `ring_insert__ep7/front_first.png`.
- **top** (1024×576): high camera above and behind the arm side, looking
  down at the mat at a steep but not perfectly vertical angle (far wall
  visible at top of frame, mild perspective foreshortening). Evidence: all
  `top_*.png`.
- **wrist** (1280×720): camera mounted on the arm at the wrist/gripper,
  looking down along the gripper; both black gripper fingers are visible at
  the bottom of the frame and the view moves with the arm. Evidence:
  `block_stack__ep411/wrist_first.png`, `tool_insert__ep630/wrist_first.png`.

## 3. block_stack

Objects (evidence: `block_stack__ep411/front_first.png`,
`__ep453/top_first.png`, `__ep489/top_first.png`, `__ep498/top_first.png`):
- **2 cubes**, matte, slightly textured (painted wood or dense foam
  appearance; edges sharp, faces flat in wrist close-up
  `block_stack__ep411/wrist_first.png`).
- Colours VARY by episode: one cube is **purple in every episode viewed**;
  the second cube is **green** (411, 453, 476, 479) or **orange** (489, 498).
  (This corrects the coordinator's scouting note "orange+purple" — that pair
  occurs, but green+purple is at least as common in the frozen set.)
- Cube edge ≈30 ±10 mm [pixel-ratio vs the WOWROBO lettering in wrist views,
  cross-checked across ep411/453; low-medium confidence — Phase 3 should
  refine per-episode from top views].
- No fixture. Cubes start at varying positions/orientations on the mat
  (compare ep411 vs ep489 top_first). Final frames show one cube stacked on
  the other (ep411 top_final: green cube alone visible on mat, purple
  occluded beneath the stack position; ep498 top_final consistent),
  corroborating recorded_success=1.
- Fixed: cube count, purple cube, mat/enclosure. Varies: second-cube colour,
  both cubes' start poses.

## 4. eye_drops_to_basket

Objects (evidence: `eye_drops_to_basket__ep1609/top_first.png`,
`__ep1618/top_first.png`, `__ep1630/front_first.png`, `__ep1639/top_first.png`):
- **Eye-drops carton ×1**: small rectangular cardboard carton, blue + white
  printed faces; ≈90 × 35 × 35 mm ±20% [size relative to the platform top,
  below]. Starts LYING ON ITS SIDE on the platform top; which printed face
  is up varies (blue face up in ep1639, white/label face up in ep1609/1618).
  Carton start position/orientation on the platform varies slightly.
- **Platform ×1 (fixture, fixed)**: white stand at centre-right of the mat —
  a flat horizontal top plate (≈200 × 160 mm) on a central vertical web with
  a flat base plate on the mat; top surface ≈90 ±15 mm above the mat
  [front views ep1609/ep1630, height vs carton length]. Looks like a
  bent-sheet / 3D-printed stand; all-white.
- **Basket ×1 (fixture, fixed)**: clear transparent ribbed plastic
  rectangular container at the LEFT of the mat, ≈260 × 180 mm footprint,
  walls ≈60–80 mm high, open top. Evidence: ep1609 top_first (left),
  wrist view `eye_drops_to_basket__ep1618/wrist_first.png` (ribbed wall).
- Final frames: carton inside the basket (ep1609 top_final, ep1724
  top_final), corroborating recorded_success=1.
- Fixed: platform position, basket position, tape strips. Varies: carton
  pose/face-up on the platform.

## 5. ring_insert

Objects (evidence: `ring_insert__ep7/top_first.png`, `__ep7/front_first.png`,
`__ep30/top_final.png`, `__ep33/top_first.png`, `__ep48/top_first.png`):
- **Board ×1 (fixture, fixed)**: flat circular cream/pale-wood disc,
  Ø ≈170 ±20 mm, ≈15 mm thick, resting on the mat right-of-centre.
- **Recesses: 5** shallow circular recesses in the board top, arranged in a
  ring around the centre. Counted directly in final frames where all are
  empty: ep7 top_final and ep30 top_final each show exactly 5. (This
  corrects the coordinator's scouting note of 6.)
- **Peg ×1 (fixed)**: light-wood cylindrical peg at board centre, rounded
  top, Ø ≈16 ±4 mm, standing ≈70 ±15 mm above the board top [front view
  ep7, height vs board Ø].
- **Ring ×1**: flat torus, outer Ø ≈45 ±10 mm, inner Ø ≈22 ±5 mm,
  thickness ≈12 ±3 mm [wrist close-up `ring_insert__ep31/wrist_first.png`
  vs gripper fingers]. Colour VARIES: yellow (ep7, 11, 33), red (ep48).
  Starts seated in/over one of the board's recesses (lower-left recess in
  ep33; left recess in ep7). Final frames: ring around the peg at board
  level (ep7/ep30 top_final), corroborating recorded_success=1.
- Fixed: board/peg position and orientation class. Varies: ring colour,
  which recess it starts in.

## 6. tool_insert

Objects (evidence: `tool_insert__ep622/top_first.png` + tray crop,
`__ep622/front_first.png`, `__ep630/wrist_first.png`, `__ep684/top_final.png`):
- **Tray ×1 (fixture, fixed)**: black foam (EVA) tool tray with red recess
  backing, in a red/black outer case; front face printed "Premier by
  SEALEY" (ep622 front_first). ≈385 × 270 × 45 mm [PROP: Sealey Premier
  EVA tray class; consistent with pixel scale vs the 250 mm bar, below].
  Occupies the far half of the mat in every episode viewed.
- **Tools in tray (fixed set)**: chrome socket extension bars and universal
  joints seated in labelled slots. Labels legible in ep622 top_first crop:
  1/2" Extension Bar 250mm; 3/8" Extension Bar 250mm; 1/2" Extension Bar
  125mm; 3/8" Extension Bar 150mm; 1/4" Extension Bar 150mm; 1/4" Extension
  Bar 75mm; 3/8" Extension Bar 75mm; 1/4" Extension Bar 50mm; (partial)
  "…Extension Bar 100mm"; Universal Joint 1/2", 3/8", 1/4". ≈11 pieces
  seated in first frames.
- **Empty slot (the target)**: exactly ONE slot is empty in first frames —
  a cross/T-shaped red recess at centre-left of the tray (ep622, ep683,
  ep697 top_first).
- **Loose tool ×1 (the manipulated object)**: ONE chrome steel bar lying on
  the mat in front of the tray; knurled grip section, drive/socket ends;
  ≈100 ±25 mm long, shaft Ø ≈10 mm [wrist close-up ep630 vs lettering].
  Start position/orientation on the mat varies (ep622 right-of-centre,
  ep683 centre, ep697 left, partly behind the arm).
- Final frames: loose tool gone from the mat and the tray's empty slot
  filled (ep684 top_final), corroborating recorded_success=1.
- Fixed: tray position, seated tool set, empty-slot identity. Varies:
  loose tool start pose.

## 7. cable_clip

Objects (evidence: `cable_clip__ep1896/top_first.png`,
`__ep1896/front_first.png`, `__ep1939/top_first.png`,
`__ep1959/top_first.png`, `__ep1960/top_final.png`,
`__ep1896/wrist_final.png` crop):
- **Clip block ×1 (fixture)**: white rectangular block ≈110 × 70 × 55 mm
  ±20%, with a **black clip** (two black jaw/strap elements) mounted on its
  top. The block sits on the mat; its position VARIES between episodes
  (centre-left in ep1896, right-of-centre in ep1939, centre in ep1959) —
  it is re-placed per episode, not fixed. Two small white feet/clips are
  visible at its base front in the ep1896 wrist_final crop.
- **Cable ×1 (manipulated object)**: black flexible cable, Ø ≈5 ±1 mm,
  ≈60–90 cm long [coiled loops in top views], with a moulded USB-type
  connector at one end (visible ep1896 top_first, ep1939 top_first — plug
  body ≈25 × 12 mm). Starts coiled/looped on the mat in varying layouts,
  partly under/near the arm; layout varies strongly per episode.
- **Tape strips**: 5 white strips on the mat in a row across the upper mat
  (ep1959 top_first) — fixed markers, present in all cable episodes viewed.
- Final frames: cable still coiled on the mat, NOT seated in the clip
  (ep1896 top_final, ep1960 top_final — gripper at the block, cable on the
  mat), corroborating recorded_success=0 for all cable_clip episodes.
- Fixed: tape strips, mat/enclosure. Varies: block position, cable layout.

## 8. Frame-extraction verification (Task A1)

- Method: `code/extract_frames.py` mirrors the eval's `scripts/frames.py`
  mapping (per-episode, per-camera chunk/file + from/to timestamps from
  `meta/episodes/chunk-000/file-000.parquet`; first = from_timestamp,
  final = to_timestamp − 1/20 s). PyAV is absent from all current venvs, so
  decoding used the imageio-ffmpeg binary bundled in the sim venv
  (frame-accurate input seek). Only the 100 split.json episodes were read;
  no sealed-task episodes were touched.
- Result: **600/600 frames extracted, 0 failures** (`logs/extract_frames.log`).
- Cross-check vs `sample/` (required check, 5 episodes × 6 images = 30
  images, incl. the 2-episode minimum: block_stack ep489 and
  eye_drops_to_basket ep1810): **PASS** — mean absolute pixel difference
  1.13–1.24 (0–255 scale) on every image, 0.0% of pixels differing by >10;
  i.e. identical frames up to a small uniform decode offset (PyAV vs
  ffmpeg colour conversion). Full numbers: `logs/crosscheck.log`.
- Secondary comparison vs `judgments/frames/` (600 images): all 100 first
  frames agree (MAD ≈1.21 median for every camera). Final frames agree for
  most front/top images but DIVERGE for a subset — worst in wrist_final
  (median MAD 1.23 but max 83.1; e.g. eye_drops ep1783, cable_clip ep2085).
  Interpretation (flagged for the verifier, not resolved here): the
  judgments final-frame extraction appears to have landed on a different
  final frame for fast-moving (arm-mounted) views in some episodes; the
  curated `sample/` agreement above is the authoritative pass for this
  extraction, and `sample/` finals match this extraction exactly.
