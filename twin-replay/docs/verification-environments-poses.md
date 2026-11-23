# Verification — Environments + Poses (V-ENVIRONMENTS+POSES, independent)

Verifier: independent context, raw artifacts only (frames, overlays, scene/detection/
calibration JSONs, variant JSONs, generation + finalize code, validation_results.json).
Builder B's pose-estimation.md and Builder A's inventory were treated as claims to test.
docs/twin-validation-report.md was NOT read. Date: 2026-11-22. No checkpoint inference performed.

## VERDICT: NOT-CERTIFIED

The pose layer contains gross, outcome-determinative errors in tool_insert (7 of 17
in-denominator episodes carry a fabricated initial tool pose), a finalize layer that
cannot be re-derived from its own preserved detections on 27/62 block entries, and a
calibration whose stated world frame is not the arm/FK world frame (a reflection +
rotation + offset separates them; median disagreement between B's map and the FK-affine
map over the mat grid is 29.4 cm). Ring poses, variant generation, and camera/layout
descriptions verify cleanly (below) but do not offset the failures above.

---

## 1. Pose sample audit (verifier's own frame reads)

Method: verifier read `frames/<ep>/top_first.png` and `scenes/overlays/<ep>.png`
directly, located the manipulated object independently (own visual read, cross-checked
with a visual-grounding locator), converted both the verifier pick and the scene-JSON
pose to pixels through the inverse of B's documented homography (calibration_v2.json),
and measured the error in cm in both B's frame (H_B) and the FK-affine frame
(calibration_fk_affine.json, the frame Builder C actually used).

### tool_insert — allegation CONFIRMED, and understated for 4 of 7 episodes

| Episode | JSON pose inverts to px | Verifier's tool location (px) | Error (H_B) | Error (affine) |
|---|---|---|---|---|
| ep697 | (769, 470) | ~(240, 350) left of arm, partly behind it | 41.9 cm | 34.0 cm |
| ep779 | (768, 473) | ~(220, 360) left, partly behind arm (±40 px) | 42.6 cm | 35.2 cm |
| ep784 | (768, 474) | ~(340, 320) small bar below wrist plate | 37.8 cm | 27.9 cm |
| ep885 | (237, 406) | ~(500, 452) vertical bar on mat | 18.6 cm | 16.9 cm |
| ep888 | (232, 395) | ~(475, 392) bar right of wrist | 15.5 cm | 16.0 cm |
| ep892 | (236, 399) | ~(605, 505) bar at bottom-right of mat | 28.6 cm | 23.8 cm |
| ep903 | (237, 399) | ~(360, 302) visible end above wrist plate (centre occluded; error is a lower bound) | 9.7 cm | 11.7 cm |
| ep622 (control, grounding method) | (550, 434) | (526, 421) | 2.3 cm | 1.6 cm |

Findings:
- All 7 flagged episodes are wrong. The allegation's 14–22 cm range holds only for
  ep885/ep888; ep697/ep779/ep784/ep892 are worse (24–43 cm depending on frame).
- Failure signature is mechanical, not noise: ep697/ep779/ep784 share the IDENTICAL
  detection pixel (~(768, 473)) — a fixed scene feature at the mat's right edge where
  the overlay marker visibly sits on bare cardboard with no object present.
  ep885/ep888/ep892/ep903 cluster at ~(235, 400) — on the arm/wrist hardware itself.
  In both clusters the metal∩diff segmentation locked onto fixed chrome (tray/arm)
  instead of the loose tool, whose true position varies per episode.
- All 7 carry confidence "medium", sigma_xy 3.5 cm — i.e. they entered the validation
  denominator with errors 3–12× their stated sigma. The 3 episodes B did mark low
  (ep630/871/876) show B could recognise occlusion failure; these 7 are the same
  failure class, unrecognised.
- Control: the visual-grounding episodes verify (ep622: 1.6–2.3 cm). The defect is
  specific to the detect4 segmentation path for tool_insert.

### Other tasks (verifier overlay/frame reads)
- ring_insert ep7: ring marker dead-on the yellow ring; ring_final marker at the peg
  base. Poses agree with detections exactly (§2). PASS.
- block_stack ep411: green-cube marker on the green cube. Purple-cube JSON pose is
  within ~2–3 cm of the visible cube — but see §2: it is the preserved DETECTION
  that is off here, not the JSON.
- eye_drops / cable_clip: poses are grounding/descriptor grade by construction
  (cable is outcome-class only, by design); no additional gross error found in the
  frames read (ep1609, ep1896). Not certified either — grounding picks carry no
  re-derivable pixel record in detections_px.json (detections4 first/final are
  empty {} for most eye episodes).

## 2. Internal consistency: finalize JSONs vs B's own detection pixels

Method: for every block_stack and ring_insert object entry, mapped detections_px.json
centroids through B's H and compared to the scene JSON pose (Euclidean, cm).

- block_stack: n=62 entries. 35 agree to 0.01 cm (pure re-derivation); 27 disagree
  by >1 cm — and the distribution is bimodal: every one of the 27 is >2 cm,
  24 are >3 cm, 15 >5 cm, 4 >10 cm (max 13.1 cm, ep411 cube_purple). Median over
  all 62 is 0.01 cm; mean 2.64 cm. The allegation is CONFIRMED in count
  (27 mismatches; Builder C's changelog independently states 27/80 counting a
  wider entry set).
- Direction matters: the mismatches are NOT uniformly "JSON wrong". On ep411
  cube_purple the verifier's frame read puts the cube at ~(465, 330) px: the JSON
  manual pick inverts to (490, 330) (~2–3 cm away — acceptable), while the
  preserved detection sits at (447, 482), on empty mat ~13 cm away. Other
  mismatches are final-stack constructions (base cube "under stack" entries,
  e.g. ep548 finals at 9.8–11.6 cm) where JSON and detection encode different
  conventions. Either way, B's claim that "pixel-space detections are preserved
  so the verifier can re-derive every pose" is FALSE for 27/62 (44%) of block
  entries: the JSON layer and the pixel layer are two different pose sets with
  no per-entry record of which is authoritative or why they diverge.
- ring_insert: n=40, median 0.00 cm, max 0.01 cm, 0 mismatches. Fully consistent.

## 3. Calibration

- B's ±2.5 cm (1σ, near/mid mat) is an error BUDGET, not a measurement. The only
  preserved evidence, scale_crosschecks.json, gives cube extent median 37.5 mm
  vs inventory nominal 30 mm (+25%; inside the wide ±10 mm inventory range) and
  ring extent 48.4 mm vs 45 mm (+7.6%). These are consistent with a scale error
  of roughly +8–25% on object extents and cannot support a ±2.5 cm absolute
  claim; at u=0.30 m a +10% scale error alone is 3 cm.
- Decisive frame problem: B's homography frame is NOT the stated world frame
  (origin at arm base, x forward, y left). The FK-affine map that Builder C had
  to fit has a NEGATIVE determinant (reflection) and its outputs disagree with
  B's map by a median of 29.4 cm (range 3.1–71.5 cm) over a mat grid. At most
  one of the two maps can express poses in the FK world frame; B's poses are
  therefore not accurate to ±2.5 cm in the frame B claims — they are
  self-consistent only up to an unknown reflected rigid transform. B's own §1
  records the ~14 cm FK inconsistency and sets it aside; that was the
  calibration's central defect, not a side note.
- FK affine (calibration_fk_affine.json): it IS a single global map — one 2×3
  affine applied to every pose in code/build_scenes.py (`correct_xy`), linear
  part singular values 0.076 / 0.060 cm/px, det < 0 (reflection), translation
  (0.192, 0.359) m. No per-episode or per-task terms.
- The claimed tie-point improvement (12.8 → 1.4 cm median, n=93) CANNOT be
  recomputed from the preserved artifacts: the 93 dwell pairs were never
  archived. scenes/fk_tiepoints.json contains only first/final frames, which
  are home poses (gripper ≈ (0.111, 0.005, 0.081) m for essentially every
  episode) — not grasp/place dwells. The verifier's attempted independent tie
  check (wrist-plate pixel in ep411 top_first vs FK wrist_xyz) is inconclusive
  by construction (plate pixel ≠ wrist-body origin; both maps land 13–17 cm
  from it) and is reported as such, not as evidence either way. Status of the
  1.4 cm claim: PLAUSIBLE (a reflection+rotation frame error of the observed
  size would produce a ~12.8 cm pre-correction median, and C's downstream
  grasp-gate behaviour is consistent with a large correction) but UNVERIFIED
  as archived. The dwell-pair list must be published for this claim to be
  certifiable.

## 4. Variants (Builder C draws, scenes/*_variants.json)

- K=5 per episode: CONFIRMED, 100/100 variant files have exactly 5 variants
  (k=0 nominal + 4 sampled).
- Jitter vs sigma: normalised draws (dx/σ, dy/σ over all 560 jittered bodies,
  1120 components) have mean −0.004, std 0.957, 3.7% beyond 2σ, 0.36% beyond
  3σ — statistically a proper N(0, σ) per Amendment 1 (which specifies sampling
  within B's error estimates, i.e. Gaussian, not a hard 1σ clip). The largest
  single draw is cable ep2130 k=2 at 4.0σ radial (32.2 cm on an 8 cm descriptor
  sigma) — a legitimate Gaussian tail draw, flagged here only for visibility.
  Caveat: the sigmas sampled are B's, so variant families inherit §1/§3's
  defects — e.g. tool variants jitter around poses that are 12–35 cm wrong.
- Mass draws: all within the physics-table ranges (block 10–35 g, eye 20–45 g,
  ring 4–15 g, tool 50–100 g, cable total 40–60 g — observed min/max match the
  ranges to rounding). Friction multipliers: all within [0.80, 1.20] (n=560,
  observed 0.801–1.200), the ±20% fallback range.
- Seed reproducibility: regenerated draws with build_scenes.make_draws
  (SeedSequence [20261122, crc32(episode), k]) for tool ep697, ring ep7, eye
  ep1609 — exact match on all 5 variants each. Block ep411: xy/mass/friction
  reproduce exactly; cube_a yaw draws differ because the verifier fed the raw
  finalize JSON rather than the builder's internal pose dict (input-form
  difference, not a seed failure). Seed mechanism VERIFIED.
- The 12 excluded variants (validation_results.json, per-variant build_defect
  strings): eye_drops ep1780 k4, ep1783 k2, ep1808 k4, ep1814 k1 — all
  "carton: displaced 0.090 m / z=0.017 after pre-settle"; tool_insert ep885
  k0/k1/k3, ep888 k0/k4, ep892 k0/k4, ep903 k0 — all "tool: displaced
  0.034–0.038 m after pre-settle". Judgement: these ARE logged per variant with
  a stated mechanism, and pre-settle displacement is within Amendment 1's
  "initial instability" exclusion class, so the exclusions are formally
  compliant. Two qualifications: (a) the eye defect is identical (exactly
  9.0 cm = platform height, carton ends at mat z) in 4 different episodes —
  a systematic spawn/platform mismatch pattern, not four independent
  instabilities; (b) the tool exclusions fall EXACTLY on the §1 episodes whose
  initial poses are fabricated (ep885/888/892/903) — a tool spawned 12–35 cm
  from its true position, plausibly interpenetrating other geometry, sliding
  3.5 cm in pre-settle is a downstream symptom of the pose defect. They are
  build defects in form, but they are not independent of the pose failure,
  and 4 of the 8 are k=0 NOMINAL variants — those episodes effectively lost
  their nominal twin. No excluded variant shows an outcome-failure reason
  (no "failed to reach / wrong final position" exclusions were found).

## 5. Camera / layout claims (scene-inventory §2)

Spot-checked against frames (block ep411 front/top/wrist first + all top_first
frames read in §1): CONFIRMED on all three.
- front: low, roughly mat level, arm large in the left foreground, back wall
  filling the background — matches ep411 front_first exactly.
- top: high, steep but visibly tilted (far wall at top of frame, perspective
  foreshortening, barrel curvature at mat edges) — matches.
- wrist: arm-mounted, looking down along the gripper, both gripper fingers at
  the bottom of frame — matches ep411 wrist_first exactly.
One inventory error found in passing, already corrected by B and confirmed by
the verifier's count in ring ep7's frame: the ring board has 6 recess
positions (5 empty + 1 under the ring in ep7), not the inventory's 5.

## Required fixes before re-certification
1. Re-estimate tool_insert initial poses for ep697, 779, 784, 885, 888, 892,
   903 (grounding/manual with per-episode visual verification); re-run their
   variant families and validation. Audit the remaining detect4-segmentation
   episodes for the same fixed-feature signature (identical/near-identical
   detection pixels across episodes).
2. Publish a per-entry authoritative-source record reconciling finalize JSONs
   with detections_px (27 block entries), or regenerate JSONs from the pixels.
3. Archive the 93 FK dwell pairs so the affine fit (12.8 → 1.4 cm) is
   recomputable; restate B's accuracy claim as mat-frame self-consistency,
   not world-frame accuracy, until a physical mat measurement exists.
