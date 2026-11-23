# Study 001 Twin-Replay — Phase 3: Pose Estimation (Builder B, 2026-11-22)

Deliverables: `scenes/<task>__ep<id>.json` x100 (initial + final poses,
confidence, `sigma_xy_cm` / `sigma_yaw_deg` per Amendment 1),
`scenes/fixtures.json`, `scenes/overlays/<task>__ep<id>.png` x100 (final
markers on `top_first`; green=high, yellow=medium, red=low),
`scenes/calibration_v2.json`. Code: `code/detect.py` (v1, superseded),
`code/detect2.py`/`detect3.py` (segmentation), `code/detect4.py` (eye/tool
regions), `code/detect5.py` (board/block attempts), `code/finalize.py`,
`code/patch_finalize.py`, `code/patch2.py`, `code/fk_tiepoints.py`,
`code/calibration.py`, `code/mat_corners.py`. Pixel-space detections are
preserved in `scenes/detections_px.json` / `detections4_px.json` /
`detections5_px.json` so the verifier can re-derive every pose.

## 1. World frame and calibration (`calib_v2_mat_homography`)

Frame (per brief): origin at arm base centre, x forward, y left, z up,
tabletop z=0.

Method: planar homography from the mat. The top camera is steep but tilted
with visible wide-angle barrel distortion; a homography is an approximation
(residual distortion is folded into the accuracy estimate, §2).

- Mat corners in top-view pixels, read by Builder B from
  `block_stack__ep411/top_first.png` (cleanest frame; cross-checked on
  ep622): NL (42,498), NR (858,508), FR (928,148), FL (338,64). Automated
  corner extraction (mask extremes / convex-hull quad) was tried first and
  REJECTED: fixture occlusion and the washed-out far corner bias it by
  50-100 px (see `scenes/mat_corners.json`, `mat_quad.json`). Corner
  uncertainty ±15 px.
- Mat dimensions from the scene inventory: 0.55 m (long axis, along the
  near/far edges = world y) x 0.40 m (near-to-far = world x), ±20%.
  Homography maps pixels to mat-local (u = near-to-far 0..0.40,
  v = NL-to-NR 0..0.55).
- Mat-to-world placement: world x = u + 0.05 (mat near edge sits ~5 cm in
  front of the arm base centre: in top views the base structure lies just
  outside the near edge; front views agree); world y = v - 0.155 (the arm
  sagittal plane: the arm base centroid and the wrist plate at home both
  project to mat v ≈ 0.153-0.156, measured in ep411 — a pure image-geometry
  read, independent of any scale assumption).
- FK cross-check ATTEMPTED AND REJECTED: MuJoCo FK of the recorded `gt`
  home pose (so101_new_calib.xml; gripper body median over 100 episodes
  (0.111, 0.004, 0.081) m) is inconsistent with the image geometry by
  ~14 cm in x (the wrist plate in the image sits at mat u ≈ 0.21, i.e.
  world x ≈ 0.26 under this calibration, not 0.111). The LeRobot degree
  conventions evidently do not map 1:1 onto the model qpos (offset/sign
  per joint not established). FK was therefore NOT used as an anchor.
  Phase 4 should note: the same mismatch will affect kinematic replay if
  raw `gt` degrees are fed to this model unverified. Tie points preserved
  in `scenes/fk_tiepoints.json`.

## 2. Calibration accuracy (evidence)

- Known-size cross-checks through the homography (`scenes/scale_crosschecks.json`):
  cube major extent (PCA, 40 cube detections) median 37.5 mm vs inventory
  30 ±10 mm — inside the inventory range; ring major extent median
  48.4 mm vs inventory outer diameter 45 ±10 mm — inside range, +7.6%
  vs nominal. Both checks are consistent with the mat dimensions being
  correct to roughly ±10%, better than the ±20% inventory bound, but
  they do not prove it (both reference sizes are themselves estimates).
- Error budget (1-sigma, planar): centroid noise 5-10 px ≈ 0.5-1.5 cm
  (scale varies ~13 px/cm near to ~7 px/cm far); corner uncertainty
  ±15 px → homography warp ±1.5-2.5 cm (worst at far corners);
  barrel-distortion residual after homography ≈ ±1 cm mid-mat;
  placement offsets (x0 ±2 cm, v0 ±1 cm) are COMMON-MODE translations:
  they shift all poses together and cancel in initial-vs-final and
  inter-object comparisons within an episode. Scale error (mat dims)
  grows with distance from the near edge: at u=0.30 m a ±10% scale error
  is ±3 cm in x.
- Bottom line: absolute pose accuracy ≈ ±2.5 cm (1-sigma, near/mid mat)
  to ≈ ±4 cm (far edge), dominated by mat-dimension and corner
  uncertainty. Relative accuracy within an episode (e.g. final vs the
  3.0 cm secondary criterion, same method both ends) ≈ ±1.5-2 cm.
  If Phase 4 validation shows a systematic miss, re-measuring the mat
  physically is the single highest-value calibration upgrade.

## 3. Detection methods per task

Common machinery: colour segmentation in the top view INTERSECTED with a
per-task median-image difference mask (median over the task's 20 first or
final frames). The median cancels all fixed scene content — crucially the
printed WOWROBO lettering, which defeated naive colour masks in v1 (the
green "ROBO" letters produced a constant-position false "green cube" in
all 20 block episodes; v1 outputs were discarded). Every episode's final
markers were visually verified by Builder B against the frames
(montage pass over all 100 overlays + full-res reads of every ambiguous
case listed below); failures were replaced by visual-grounding picks or
manual picks, each labeled in the JSON `method` field.

- block_stack: purple/green/orange saturation-gated masks ∩ diff,
  area 250-6000 px; second pass without diff (excluding median-print
  blobs) for cubes whose position coincides with the median. Manual
  corrections after visual verification: ep411 purple (occluded region /
  median coincidence), ep530 purple, ep539 both cubes, ep546 purple.
  Manipulated cube identified mechanically: final stack location from
  top_final detections; the cube whose initial position matches the stack
  (<70 px) is the base, the other is manipulated. Finals: stack (x,y) =
  mean of final cube detections (high if two detections coincide <55 px).
- eye_drops_to_basket: carton on the white platform defeats colour
  segmentation in ~half the episodes (white face up on white plate;
  platform-edge difference artifacts). Final method: interior-blob
  segmentation restricted to the platform plate (detect4; 6 episodes,
  high) + visual-grounding picks for the remaining 14 first frames and
  18 final frames (medium). Grounding was validated against Builder B
  full-res reads (agreement <25 px on checked episodes). Finals ep1618 /
  ep1760: arm ends over the basket and the carton is occluded in
  top_final — LOW, no pose.
- ring_insert: yellow/red mask ∩ diff; 20/20 initial, 20/20 final.
  CORRECTION to scene-inventory: the board is NOT fixed — it is re-placed
  between episodes (final-ring peg positions spread std ~65x45 px, two
  loose clusters). Per-episode peg = the ring's final position (ring is
  seated on the peg in every recorded final); board centre = peg +
  (+28,-7) px (median offset measured on ep7/ep33). Recess count is 6,
  not 5 (counted directly in ring_insert__ep33 first/final frames);
  recess offsets relative to peg taken from ep33 geometry; board rotation
  per episode is not estimated, so `start_recess` identity is approximate
  (distance to nearest recess centre is recorded).
- tool_insert: chrome-on-dark-mat segmentation is unreliable (tray bars
  and arm hardware are the same material). Final method: detect4
  metal∩diff outside the tray (7 episodes) + visual-grounding picks
  (10 episodes), all medium. LOW (occluded by arm/wrist plate in
  top_first; methods disagree): ep630, ep871, ep876. Final pose for all
  episodes = the target slot (fixture; recorded demos all succeed), medium.
- cable_clip: white-block segmentation REJECTED (block highlights merge
  with the mat border stripe — every raw blob ran to the mat edge — and
  split by lighting; raw centroids were biased up to ~300 px). Block pose
  = visual-grounding pick for all 20 episodes (medium; cross-check on
  ep1939 vs Builder B full-res read: <5 px). The block VARIES per episode
  (grounded centres span most of the mat). Cable: coarse descriptor only
  (coil centre + extent bbox from dark non-arm pixels ∩ diff) — full
  cable state is not pinnable from a single top view, as the brief
  anticipated; confidence low, sigma 8 cm on the coil centre. Per the
  pre-registration cable_clip is scored on outcome class only.

## 4. Confidence rubric and sigma derivation (Amendment 1)

- high: clean segmentation, unoccluded, blob area in the expected range,
  visually verified on the overlay. sigma_xy = 2.5 cm.
- medium: partial occlusion, second-pass / grounding / manual pick, or
  fixture-derived pose. sigma_xy = 3.5 cm (cable block 4.0 cm — grounding
  scatter on a large low-contrast object; ring_final on peg 2.0 cm — the
  peg mechanically constrains the ring centre; tool_final slot 2.5 cm).
- low: object cannot be pinned (occluded / methods disagree). Pose is
  null and sigma is null; the episode leaves the relevant denominator
  per the pre-registration.
- sigma_xy derivation: root-sum-square of the §2 budget terms relevant to
  the method (centroid/pick noise 0.5-2 cm by method; homography warp
  ~1.5-2.5 cm; common-mode placement excluded from sigma because variant
  jitter should sample per-episode error, not a shared translation —
  stated here so Builder C can add it back globally if desired).
  sigma_yaw: from PCA axis stability — elongated objects (tool) 15°,
  carton/block PCA 20°, near-square cubes 25° (modulo 90° face
  symmetry); null when yaw itself is null (symmetric / not estimated).

## 5. Pose counts by confidence (object entries, initial + final)

- block_stack: high 33, medium 29, low 0 (62 entries: 40 initial cubes
  + 22 final stack entries — finals recorded for base/manipulated pair).
  Initial episodes at medium-or-better: 20/20.
- eye_drops_to_basket: high 6, medium 32, low 2 (40 entries). Initial:
  20/20 medium-or-better. Low entries are finals only (see §6).
- ring_insert: high 25, medium 15, low 0 (40 entries). Initial 20/20.
- tool_insert: medium 37, low 3 (40 entries). Initial medium-or-better:
  17/20 (ep630, ep871, ep876 excluded from the validation denominator).
- cable_clip: block medium 20/20; cable descriptor low 20/20 (by design —
  outcome-class scoring only).

## 6. Named low-confidence poses (no pose emitted)

- tool_insert__ep630, tool_insert__ep871, tool_insert__ep876 — initial
  tool pose: loose tool occluded by the arm/wrist plate in top_first.
  These 3 episodes leave the twin-validation denominator (initial pose).
- eye_drops_to_basket__ep1618, eye_drops_to_basket__ep1760 — FINAL carton
  pose only (arm occludes the basket in top_final). Initial poses are
  medium; the episodes stay in the validation denominator but have no
  secondary-criterion final position.
- cable_clip cable layout (all 20) — descriptor only, by design.

## 7. Systematic risks (for the verifier and Phase 4)

1. Mat dimensions ±20% (inventory) bound all absolute positions; cross-
   checks suggest ~±10% but cannot prove it. A physical mat measurement
   collapses this risk.
2. Barrel distortion is approximated by a homography; residuals are
   largest mid-edge. Variant jitter (Amendment 1) partially covers this.
3. Board/peg and cable block are re-placed per episode — any Phase 4
   scene template that fixes them task-wide will be wrong for most
   episodes. (This also corrects the scene inventory's "fixed" labels
   for the ring board; platform/basket/tray WERE constant in every
   episode checked — platform and basket hand-read on ep1609 and
   confirmed visually on ep1639/ep1778/ep1630 frames; tray centre from
   per-episode tray masks, spread <10 px.)
4. FK of recorded `gt` in so101_new_calib.xml does not reproduce image
   geometry (see §1) — kinematic replay needs a joint-convention check
   before use.
5. z values are from the inventory (platform top 90 mm, board 15 mm,
   tray 45 mm, block 55 mm) — not measured per episode.
6. Yaw is the weakest channel: PCA yaw on near-square cubes is modulo
   90°; grounding/manual poses carry yaw = null rather than a guess.
