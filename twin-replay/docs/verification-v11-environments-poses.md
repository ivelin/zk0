# Verification v1.1 — Environments + Poses (V-Environments+Poses, independent)

Verifier: fresh independent context. Trust placed only in raw artifacts re-derived in this
session: episode frames read directly, preserved detection pixels, scene JSONs, the v1 backup
set (`validation/builder2_backup_v1/`), the dwell archive, and the stage-1/2 evidence JSONs.
Builder 2's report was treated as claims to test. CPU-only, `nice -n 10`, DGX training undisturbed.
Date: 2026-11-22.

## VERDICT: CERTIFIED-WITH-FINDINGS

The v1 defects are substantively repaired and every quantitative claim I could recompute
reproduces — several of them exactly. Five of the seven repaired tool poses verify visually
against their frames. The certification carries findings because the two weakest repairs do
not both hold up: **ep784's FK-only repair is contradicted by the visible tool at ~2× its
claimed σ (FINDING F1, unsound at claimed σ)**, and **ep779's cannot be visually confirmed
or refuted (F2, indeterminate)**. These two episodes should not be quoted at σ = 3.5 cm;
ep784 should be re-anchored to its visible fragment or moved OUT, and ep779 should be
labelled FK-only/indeterminate wherever it is used. No other pose in the suite is affected.

---

## Check 1 — The two FK-only repairs (priority)

Method: I read `frames/tool_insert__ep{779,784}/top_first.png` at full resolution and in an
enlarged crop of the arm/wrist region, formed my own pixel pick for the loose tool, and
converted picks to Builder B's world frame through `calibration_v2.json` (the same homography
the repairs are expressed in) for cm distances to the repaired poses.

### ep784 — verdict: **UNSOUND at the claimed σ** (repair direction right, value off)

- Visual evidence: the loose tool **is visible** in `top_first`. In the enlarged crop it is a
  silver T/socket piece sitting just left/below the black wrist plate, clear of the tray and
  of the arm body. My own centre pick: **≈(340, 305) px**. V-Env's independent v1 pick was
  (340, 320) px — two independent reads agree to 15 px.
- Converted: my pick → world (0.2089, −0.0520); V-Env's pick → (0.1957, −0.0464). The repaired
  pose is (0.1808, 0.0192). **Distance from the repaired pose: 7.65 cm (my pick) / 6.73 cm
  (V-Env's pick).**
- The claimed uncertainty for this repair is σ_axis ≈ 3.0 cm, radial RMS ≈ 4.3 cm
  (scene JSON `sigma_xy_cm` = 3.5). The visual evidence puts the true tool **1.6–1.8× the
  claimed radial RMS** away. Caveat honestly stated: the tool's centre is partly occluded by
  the wrist plate, so a centre pick carries ±2–3 cm of extrapolation — even at the generous
  end the offset is ≳4–5 cm, at or beyond the claimed radial σ.
- Why this happened, from the builder's own record: the one grounding candidate that read the
  correct fragment (V-Env's (340, 320)) sat 89.2 px from the FK anchor and was discarded by
  the uniform 75 px gate; Builder 2's own candidate (500, 291) is, on my frame read, a tool
  still seated in the tray — not the loose tool. The gate thus rejected the only correct
  visual read and kept the anchor. The repair moved the pose 31.1 cm in the right general
  direction (away from the fabricated chrome-lock), but the surviving value is not within
  its stated σ of the visible tool.

### ep779 — verdict: **INDETERMINATE** (consistent with the frames, not confirmable)

- Visual evidence: in `top_first` (full frame + enlarged crop) **no loose tool is visible
  anywhere**. Every silver bar in the tray is seated in its red slot; the mat outside the
  tray shows no loose tool; the only place a tool can be is behind/under the arm–wrist–gripper
  assembly, which is exactly where the FK anchor lands: anchor px (417.2, 363.5) converts to
  world (0.1616, 0.0201) vs the repaired pose (0.1620, 0.0201) — the pose is the anchor,
  faithfully transcribed (0.04 cm).
- Builder 2's candidate (499, 286) is a tray-seated tool on my read; V-Env's (220, 360) lands
  on bare mat lettering with nothing visible there. Neither grounding read is credible, which
  is consistent with the builder's account of why rule step (3) fired.
- So: the repaired pose is *plausible and uncontradicted* — the tool must be occluded, and the
  anchor sits in the occluded region — but no visual evidence in the frames can confirm it
  within σ ≈ 3–4.3 cm, and none should be claimed. It must travel as an FK-only estimate with
  indeterminate visual status, not as a verified pose.

## Check 2 — The other five repairs (all five spot-checked against frames)

| Episode | My frame read | Chosen pick | Verdict |
|---|---|---|---|
| ep697 | Loose silver socket clearly visible left of the arm head, ≈(230, 350) px | consensus (242, 346.5) | **Sound** — within ~15 px of the visible tool |
| ep885 | Loose vertical bar standing on the mat right of the arm, centre ≈(515, 442) px | B2 grounding (531, 442) | **Sound** — within ~16 px |
| ep888 | Loose horizontal bar right of the wrist plate, centre ≈(505, 410) px | B2 grounding (506, 411) | **Sound** — dead on |
| ep892 | Loose diagonal bar at bottom-right of the mat, centre ≈(617, 503) px | consensus (617, 503.5) | **Sound** — dead on |
| ep903 | Tool mostly behind the wrist plate; visible silver end above the plate ≈(360, 302) px | consensus (357, 307.5) | **Sound** — matches the visible fragment; centre extrapolated under occlusion, as V-Env already noted (error is a lower bound by construction) |

## Check 3 — Identical-pixel audit re-run

Independently re-scanned **all** preserved pixel entries: `detections_px.json` 246 +
`detections4_px.json` 24 + `detections5_px.json` 0 = **270 entries — the claimed count
reproduces exactly**.

- detect4 tool `first` clusters reproduce with the claimed spreads: ep697/779/784 at
  ≈(768, 473) px, max pairwise **4.5 px**; ep885/888/892/903 at ≈(235, 400) px, max pairwise
  **11.9 px**. Cross-episode pairs ≤3 px in the tool `first` entries occur **only within the
  known 7-episode set** (det4: 892/903 at 1.08 px, 779/784 at 1.46 px; det3 likewise, plus
  697/779 at 2.97 px).
- The remaining ≤3 px pairs are exactly the classes the builder flagged: eye-carton finals
  clustered at the fixed basket, cable white-blob pairs from the rejected segmentation, and
  block-cube finals at shared stack locations. None is an operative initial pose.
- One point the builder's report does not spell out, verified here so it cannot be mistaken
  for a miss: det3 (non-operative for tool) also places ep830 (207, 382) and ep852 (233, 387)
  near the second chrome cluster. Both episodes' published poses are visual-grounding picks
  (checked in their scene JSONs), not these detections, so they are not operative chrome-locks.
- **Conclusion confirmed: operative chrome-lock hits = exactly the 7 known episodes.**

## Check 4 — Block reconciliation

- Re-derived **all 62** reconciliation distances from the preserved detection pixels through
  the homography: every entry matches the published `dist_json_vs_detpx_cm` to ≤0.15 cm;
  the 35 AGREE entries have max distance **0.01 cm** (claim: ≤0.01 cm — confirmed).
- Sampled 8 FINAL_STACK_CONSTRUCTION entries (ep411/453/479/513, purple + green finals):
  recomputing the stack location as the mean of the episode's `top_final` cube detections
  reproduces the JSON finals to **0.03–0.21 cm** — the "different quantity" explanation is
  real, not a relabelling.
- MANUAL_PICK_INITIAL entries: `finalize.py` does contain the `HAND_BLOCK_FIRST` manual-pick
  table and applies it exactly as the reconciliation describes.
- "No values changed": diffed every block object pose against `validation/builder2_backup_v1/`
  — **62/62 identical**.
- **Claim confirmed per sampled entry and in aggregate: published JSON authoritative, no pose
  values changed; the v1 defect was provenance, and the provenance is now supplied and correct.**

## Check 5 — Dwell archive

`scenes/dwell_tiepoints_v11.json` actually contains the pairs this time: **275 excursion
dwells, 135 paired** (grasp 66 / place 69; per task×kind: block 11+11, ring 20+20, tool 18+20,
eye 17+18 — all matching the report). Each paired entry carries episode/task/kind, the three
selector dwell frames, FK body + site + wrist xyz, the object pixel and its FK-frame xy, and
residuals. Recomputed from the archive itself:

- Raw FK-body ↔ object median, grasp: **12.83 cm (n = 66) — exact match to the claim.**
- Affine px→FK(site) refit, train = even / test = odd, held-out median: **3.21 cm
  (block+ring) / 3.43 cm (all tasks) — exact match to both claimed figures.**
- Oracle-dwell (S3) site residuals from `builder2_stage1c.json`: **4.17 cm grasp / 2.66 cm
  place — exact match.**
- The v1 "→ 1.4 cm" figure does not reproduce under any unfiltered held-out refit I ran;
  the builder's retraction of it as an in-sample, <6 cm-filtered selection artifact stands.

## Check 6 — Fix-4 honesty

- Held-out protocol is real in code: `builder2_stage1b.py` / `builder2_stage1d.py` split
  `epi % 2 == 0` (train) vs `== 1` (test), offsets/medians fitted on train only.
- I recomputed the S7 + constant-offset held-out σ per task/kind from
  `builder2_stage1d.json` (median train offset, per-axis RMS on test): **all eight reported
  values reproduce to the printed digit** — block 4.90/7.20, ring 3.11/3.19, tool 3.03/2.48,
  eye 4.61/4.59, with the reported n_test (8/8, 11/11, 9/9, 8/9).
- No FK substitution: diffing all backup-covered scene JSONs (102 object entries) against
  v1, the **only** pose changes are the 7 tool repairs. Ring poses still re-derive from
  preserved pixels to ≤0.006 cm (unchanged image pipeline). Across the full suite,
  **134 object entries carry `fk_anchor` fields — the claimed count, exactly** — and outside
  the 7 repairs those fields are additive cross-checks only.
- The one exception is by design and disclosed: for ep779/ep784 the pose *is* the FK anchor
  (see F4 below for a labelling defect in how that is recorded).

## Check 7 — Confidence labels and σ values

- Suite-wide labels follow the v1 rubric (high 2.5, medium 3.5, medium 2.5 for
  slot/peg-constrained finals, low = null pose for ep630/871/876), and that rubric is now
  corroborated rather than merely budgeted: the dwell-based held-out measurement gives
  ≈2.6 cm/axis pooled for the image pipeline, consistent with block/ring ≈2.5 and
  tool/eye ≈3.5. The OUT episodes (ep630, ep871, ep876) remain OUT with null poses — the
  ep871 decision (single candidate 92 px from its anchor) is documented in
  `builder2_stage2.json` and is consistent with the uniform rule.
- Findings on the two FK-only repairs' labels are F1/F5 below: a grounding-grade
  `sigma_xy_cm = 3.5` / `medium` label does not honestly describe an FK-only estimate whose
  own held-out radial RMS is 4.3 cm — and for ep784 not even that σ survives the frame read.

---

## Findings

- **F1 (material, ep784):** FK-only repair is 6.7–7.7 cm from the visible tool on two
  independent frame reads vs claimed radial RMS ≈4.3 cm. Verdict: **unsound at claimed σ**.
  Recommended: re-anchor ep784 to the visible fragment (≈(340, 305–320) px, grounding-grade)
  or move it OUT; do not re-validate it at the current value/σ.
- **F2 (material, ep779):** No visual confirmation is possible — the tool is not visible in
  `top_first` and both grounding candidates are non-credible on frame read. Verdict:
  **indeterminate**. The FK value may stand only if labelled FK-only/indeterminate, not as a
  frame-verified pose.
- **F3 (minor, process):** The adjudication rule genuinely changed between stage 2 and
  stage 2b (stage 2 had ep697 as FK-only and ep892/ep903 as single-candidate picks). The
  change is honestly logged as B2-4 with a sound statistical rationale (a ~55 px-σ adjudicator
  cannot veto an 8–25 px consensus), but B2-4's text says only "ep697, ep903 repair source
  changes" — ep892's source also changed (single V-Env pick → consensus mean). The final
  values are unaffected in quality (all three verify in Check 2); the changelog sentence
  understates its own change set.
- **F4 (minor, labelling):** In ep779/ep784's scene JSONs the `fk_anchor.note` boilerplate
  states the anchor "does NOT replace the image pose" — for these two entries it *did*
  replace the pose (pose xy == `fk_anchor.xy_world_b` to 0.0 cm). The `method` field states
  the truth; the note contradicts it and should be corrected before anyone quotes the note.
- **F5 (minor, σ semantics):** The repaired tool entries carry `sigma_xy_cm = 3.5`
  (the grounding rubric value) including the two FK-only repairs, whose governing uncertainty
  is the anchor's per-axis 3.03 cm / radial 4.29 cm. Under a per-axis reading 3.5 is roughly
  fair; under the radial reading a reader will naturally apply to `sigma_xy`, it understates
  the FK-only uncertainty. State the convention, or set the FK-only entries' σ to the anchor's.

## What was NOT in scope / not verified

Variant regeneration and re-validation of the 7 repaired episodes (Builder C's lane; scene
XMLs still embed v1 poses until `build_scenes.py` is re-run — the builder states this plainly
and it remains true). Physics, assets, and harness classes belong to the other verifiers.
Environments in the narrow sense (fixtures, tray/platform/basket positions, camera/layout
descriptions) were verified in v1 and nothing in the v1.1 pose work touches them; no new
environment defect was found in the frames read for this pass.
