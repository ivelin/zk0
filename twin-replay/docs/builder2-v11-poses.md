# Builder 2 (Poses) — Twin-Suite v1.1 Report (2026-11-22)

Workdir: `/home/openclaw/muse-work/zk0-study001-twin-20261122/`. CPU-only, `nice -n 10`,
no GPU touched (DGX training undisturbed). All changes logged contemporaneously in
`validation/param-changelog.md` entries B2-0…B2-5. Code: `code/builder2_stage1.py`,
`code/builder2_stage1b.py`, `code/builder2_stage1c.py`, `code/builder2_stage1d.py`,
`code/builder2_stage2.py`, `code/builder2_stage2b.py`. Determinism: stage 2b re-run
produces byte-identical scene JSONs (md5 verified); all fits are medians/least-squares
on fixed train/test splits (train = even episode id, test = odd episode id) — no RNG,
no per-episode tuning anywhere in this report.

## FIX 3(a) — The 7 fabricated tool_insert initial poses, repaired

Method (per the brief): the **grounding method of the verified controls** — visual-grounding
picks in pixel space on `top_first.png`, converted through the same `calib_v2` homography
`finalize.py` used for the 10 control episodes (TOOL_GROUND path). Control validation of my
own grounding pass: on ep622 my pick (557,434)px lands 7px from Builder B's v1 pick
(550,434)px, which V-Env verified at 1.6–2.3 cm. Two independent grounding candidate sets
were used (Builder 2's locator pass; V-Env's published reads) plus the FK grasp anchor as
adjudicator under one uniform rule (changelog B2-3/B2-4, stated before application):

1. both candidates within 25px of each other → consensus (mean px);
2. else the candidate within 75px of the FK anchor (nearest wins);
3. else the FK-anchored pose itself.

(FK anchor = first-grasp dwell FK, S7 selector §FIX 4, + per-task offset fitted on
even-id episodes only. The 75px gate ≈ 1.4× the anchor's own held-out radial RMS
for tool grasp, ~55px.)

| Episode | Old pose (fabricated, B frame) | New pose (B frame) | Moved | Source (rule step) | Candidates px (B2 / V-Env) | FK anchor px |
|---|---|---|---|---|---|---|
| ep697 | [0.0816, 0.3128, 0.005, −3.092] | [0.1682, −0.0981, 0.005, null] | 42.0 cm | consensus (1) | (244,343) / (240,350) | (332,345) |
| ep779 | [0.0792, 0.3123, 0.005, −3.094] | [0.1620, 0.0201, 0.005, null] | 30.4 cm | FK anchor (3) | (499,286) / (220,360) | (417,364) |
| ep784 | [0.0779, 0.3123, 0.005, −3.092] | [0.1808, 0.0192, 0.005, null] | 31.1 cm | FK anchor (3) | (500,291) / (340,320) | (426,343) |
| ep885 | [0.1202, −0.0794, 0.005, 2.636] | [0.0999, 0.1233, 0.005, null] | 20.4 cm | B2 grounding (2) | (531,442) / (500,452) | (550,429) |
| ep888 | [0.1287, −0.0860, 0.005, 2.603] | [0.1252, 0.0965, 0.005, null] | 18.3 cm | B2 grounding (2) | (506,411) / (475,392) | (524,401) |
| ep892 | [0.1259, −0.0825, 0.005, 2.584] | [0.0512, 0.2019, 0.005, null] | 29.4 cm | consensus (1) | (629,502) / (605,505) | (558,458) |
| ep903 | [0.1258, −0.0819, 0.005, 2.592] | [0.2077, −0.0399, 0.005, null] | 9.2 cm | consensus (1) | (354,313) / (360,302) | (404,327) |

Notes: yaw is null on all 7 (grounding carries no yaw — same convention as the v1 control
picks; the fabricated det4 yaws are discarded). Confidence stays **medium**, σ_xy = 3.5 cm
(v1 grounding rubric). ep779/ep784 are the weakest repairs: the two grounding candidates
flatly disagree (the tool is largely behind the arm; each read plausibly picked a different
visible fragment) and the FK anchor sits between them — the FK-anchored values carry the
anchor's σ (axis ≈3.0 cm, radial RMS ≈4.3 cm). The fabricated values are preserved in each
JSON under `pose_v1_fabricated`, the full adjudication record under `v11_repair`.
Move magnitudes (9–42 cm) are consistent with V-Env's independent error estimates
(12–35 cm in the affine frame).

**Pre-registered OUT episodes — unchanged.** ep630 and ep876: grounding returns no pick
and no FK pair exists → still unobservable, stay OUT. ep871: grounding returns a candidate
(315,310)px but it sits 92px from its FK anchor (~1.7σ of the anchor) — **not** a genuine
observation at the agreement level the other repairs show, so ep871 **stays OUT**. No
denominator change; no amendment requested. (If a future verifier wants ep871 revisited,
the evidence pair is archived in `validation/builder2_stage2.json`.)

## FIX 3(b) — Identical-pixel audit (all tasks, all detection passes)

Checked: **270 preserved pixel entries** across `detections_px.json` (detect3),
`detections4_px.json` (detect4), `detections5_px.json` (detect5; empty for all episodes —
no entries), plus the published tool JSON poses inverted to pixels. Criterion: cross-episode
centroid distance ≤3px (exact/near-identical), with cluster spreads measured for every hit.

| Hit set | Pass | Spread | Disposition |
|---|---|---|---|
| tool first: ep697/779/784 ≈ (768,473)px | detect4 | max pairwise 4.5px | **CHROME LOCK — the known fabrication.** Repaired in 3(a). |
| tool first: ep885/888/892/903 ≈ (235,400)px | detect4 | max pairwise 11.9px | **CHROME LOCK — the known fabrication.** Repaired in 3(a). |
| tool first pairs (779,784), (892,903) ≤1.5px | detect3 (`detections_px`) | — | Same episodes via the other pass; detect3 tool entries were not the operative source for the published poses (finalize used det4/grounding). No new episodes. |
| cable block first: 10 pairs ≤2.6px (ep1896/1903, 1959/1960/1968/1969 cluster, 2020/2032, 2042/2046/2059, 2059/2068) | detect3 | ≤2.6px | **Not operative.** This is the white-blob segmentation Builder B already REJECTED (block highlights merge with the mat border stripe); published cable poses are grounding picks (patch_finalize.py). No repair needed; detections retained as the rejected record. |
| eye carton final: cluster around the basket (ep1724/1741/1746/1808 within ≤1.4px; 1757/1760/1778 ≤2.8px) | detect3 + detect4 | ≤2.8px | **Benign by task design.** All demos place the carton into the same fixed basket; finals cluster physically. Published finals are grounding picks, not these detections. |
| block cube final pairs ≤2.4px (ep411/514, 525/526/548, 546/549 purple finals; ep499 first/ep527 final orange) | detect3 | ≤2.4px | **Low-impact, finals only.** Stack locations in episodes whose base cubes were placed at near-identical spots; finals enter scenes only through the stack construction (§3c). Initial poses — the safety-critical quantity — show no cross-episode duplicates. Flagged for the verifier; no pose changed. |

**Conclusion: the chrome-lock signature occurs in exactly the 7 known tool episodes and
nowhere else in any operative detection pass. No additional episodes required repair.**

## FIX 3(c) — Block_stack reconciliation (27 entries >2 cm)

Machine-readable full table: `scenes/block_reconciliation_v11.json` (all 62 entries with
detection px, both poses, class, authoritative source, rationale). Of 62 block entries,
35 re-derive from preserved detection pixels to ≤0.01 cm. The 27 that do not split into
exactly two classes — **in all 27 the published JSON pose is the authoritative source and
no pose value changes**; the v1 defect was missing provenance, which this table now supplies.

**Class 1 — MANUAL_PICK_INITIAL (5 entries): JSON authoritative.** The preserved detection
is the *failed* auto detection that Builder B replaced with a visual-verification manual
pick (`finalize.py` HAND_BLOCK_FIRST). V-Env's own frame read on ep411 supports the pick
(JSON ~2–3 cm from the visible cube; detection ~13 cm away on empty mat).

**Class 2 — FINAL_STACK_CONSTRUCTION (22 entries): JSON authoritative.** The JSON final is
the *stack location* (mean of the top_final cube detections); a per-colour final detection
measures an individual colour blob in `top_final` (often only the visible top cube or the
mat lettering) — a different quantity, not a competing pose. Re-deriving the stack mean
from the preserved pixels reproduces the JSON finals (residuals are rounding-level in the
reconciliation file). Builder C's `build_scenes.py` independently derives its stack point
from the same preserved pixels (detection nearest the base initial), so both consumers are
now traceable to the pixel layer.

| Episode | Entry | Dist (cm) | Class | Authoritative source |
|---|---|---|---|---|
| block_stack__ep411 | cube_purple | 13.09 | MANUAL_PICK_INITIAL | published JSON pose (manual pick) |
| block_stack__ep411 | cube_purple_final | 4.92 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep411 | cube_green_final | 5.35 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep453 | cube_purple_final | 7.32 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep453 | cube_green_final | 7.05 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep479 | cube_purple_final | 3.71 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep479 | cube_green_final | 3.64 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep513 | cube_purple_final | 3.23 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep513 | cube_green_final | 3.36 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep527 | cube_purple_final | 7.96 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep527 | cube_orange_final | 3.57 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep530 | cube_purple | 8.26 | MANUAL_PICK_INITIAL | published JSON pose (manual pick) |
| block_stack__ep530 | cube_purple_final | 5.73 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep530 | cube_green_final | 6.31 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep531 | cube_purple_final | 2.11 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep531 | cube_green_final | 2.13 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep539 | cube_purple | 11.06 | MANUAL_PICK_INITIAL | published JSON pose (manual pick) |
| block_stack__ep539 | cube_green | 2.48 | MANUAL_PICK_INITIAL | published JSON pose (manual pick) |
| block_stack__ep545 | cube_purple_final | 4.8 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep545 | cube_orange_final | 5.17 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep546 | cube_purple | 12.12 | MANUAL_PICK_INITIAL | published JSON pose (manual pick) |
| block_stack__ep546 | cube_purple_final | 7.08 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep546 | cube_orange_final | 5.38 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep548 | cube_purple_final | 9.8 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep548 | cube_orange_final | 11.64 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep557 | cube_purple_final | 3.07 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |
| block_stack__ep557 | cube_orange_final | 3.02 | FINAL_STACK_CONSTRUCTION | published JSON pose (stack location) |

## FIX 3(d) — Dwell tie-point archive

**`scenes/dwell_tiepoints_v11.json`** — 275 excursion dwells across 80 non-cable episodes
(cable_clip excluded: the arm grasps the cable, which has no point reference), of which
**135 are paired** (first-grasp → initial object, last-place → final object): block 11+11,
ring 20+20, tool 18+20, eye 17+18. (Block pairing is 11/20 because 9 block JSONs carry no
`manipulated` role — finalize's stack identification failed there, so no grasp target can
be named from the preserved record; stated, not patched over.) Every paired entry carries:
episode/task/kind, excursion span, dwell frames under three selectors (oracle S3, first-dip
S6, crossing+lag S7), recorded FK gripper **body** xyz, FK gripper **site** xyz, FK wrist xyz,
the paired object's detection/grounding pixel, its FK-frame xy, and residuals. **Every
residual statistic quoted below is recomputable from this file** (plus the stage scripts).

Re-statement of the v1 calibration claim from this archive (unfiltered pairs, my pairing):
raw FK-body↔object distance at argmax-closedness dwell — median **12.83 cm** (grasp, n=66),
reproducing Builder C's "12.8 cm" pre-correction figure. The "→ 1.4 cm" half does **not**
reproduce: refitting an affine px→FK(site) on train pairs gives held-out median **3.21 cm**
(block+ring) / **3.43 cm** (all tasks); the oracle-dwell site residuals are median 4.17 cm
(grasp) / 2.66 cm (place). The 1.4 cm figure was an in-sample residual on pairs pre-filtered
to <6 cm agreement — a selection artifact. The honest dwell-based consistency of the image
pipeline is ≈3.3 cm median radial (σ ≈ 2.6 cm/axis), consistent with Builder B's ±2.5 cm budget.

## FIX 4 — FK-anchored pose estimation: method, honest measurement, verdict

**Method as designed and tested.** Closedness from the recorded gripper channel exactly as
`replay.py` (per-episode p5/p99, inverted — low = holding). Excursions = closedness ≥0.5
runs ≥5 frames. The object's initial pose is anchored to the **first** excursion (grasp),
its final pose to the **last** (place): object_xy = FK_site_xy(dwell) + offset, with one
per-task offset fitted on train episodes only. Four dwell selectors were compared:
S1 argmax-closedness, S2 min site-z, S6 first/last z-dip, S7 crossing + per-task median lag
to the oracle dwell (lag fitted on train only — deployment-realistic). Two anchor models
(constant offset; full affine FK→object) were compared on top. **Protocol: train = even
episode ids, test = odd episode ids; every σ below is held-out, per task, no exceptions.**

**Achieved held-out σ (per-axis cm; radial RMS = σ·√2) — best selector (S7) + offset:**

| Task | Grasp σ (n_test) | Place σ (n_test) | Image-pipeline σ (v1 labels) |
|---|---|---|---|
| block_stack | 4.90 (8) | 7.20 (8) | 2.5–3.5 |
| ring_insert | 3.11 (11) | 3.19 (11) | 2.5 (final 2.0) |
| tool_insert | 3.03 (9) | 2.48 (9) | 3.5 |
| eye_drops_to_basket | 4.61 (8) | 4.59 (9) | 3.5 |

Other selectors are worse (S1/S2/S6 grasp σ 5.9–7.4 cm/axis pooled); the affine anchor model
does not beat the constant offset (pooled grasp σ 6.1–7.2 cm). Even the **oracle** dwell
(arg-min distance to the true object — not deployable) with a train-fitted offset yields
σ 2.6–4.4 cm/axis grasp: that is the floor of this data, and it is set by jaw-geometry
offset variation with wrist orientation plus the FK↔image frame disagreement (~3.3 cm
median), not by dwell selection.

**Verdict: the σ ≲ 1 cm target is UNREACHABLE from the recordings, and FK anchoring does
not beat the image pipeline on any task/kind under held-out measurement.** Per the brief's
honesty rule, FK-anchored estimates were therefore **not** substituted for image poses
(that would have degraded the suite). What FK anchoring *did* deliver: (i) adjudication of
the 7 tool repairs, including the only usable estimates for ep779/ep784; (ii) an independent
cross-check now stored in every anchored object entry (134 entries) as `fk_anchor`
{xy_world_b, sigma_xy_cm_heldout, agreement_with_image_pose_cm}; (iii) the archived,
recomputable tie-point record in §3(d). Best achieved σ per task remains the image
pipeline's: block ≈2.5, ring ≈2.5 (final 2.0), tool 3.5 (grounding; repairs included),
eye 3.5 — now corroborated by the dwell-based measurement (≈2.6 cm/axis pooled) rather
than asserted as a budget alone. What limits further improvement, in order: (1) the
FK↔image frame affine itself only agrees to ~3.3 cm median — a physical mat measurement
remains the single highest-value upgrade, as Builder B already flagged; (2) grasp-point
vs object-centre offset varies with wrist orientation by several cm; (3) dwell-frame
identifiability from the gripper channel alone (oracle-vs-deployable gap ≈1–3 cm).

## Files changed

- `scenes/tool_insert__ep{697,779,784,885,888,892,903}.json` — poses repaired (above);
  all other scene JSONs for block/ring/tool/eye gained only additive `fk_anchor` fields
  (134 object entries; no pose values touched). Pre-change copies: `validation/builder2_backup_v1/`.
- `scenes/dwell_tiepoints_v11.json` (new), `scenes/block_reconciliation_v11.json` (new).
- `code/builder2_stage1*.py`, `code/builder2_stage2*.py` (new).
- `validation/builder2_stage1_stats.json`, `builder2_stage1b.json`, `builder2_stage1c.json`,
  `builder2_stage1d.json`, `builder2_stage2.json`, `builder2_stage2b.json` (evidence).
- `validation/param-changelog.md` — entries B2-0…B2-5 appended (no other builder's entries touched).
- This report: `docs/builder2-v11-poses.md`.

## Downstream / blockers (not in my lane)

- The 7 repaired tool episodes (and any scene whose JSON changed) need variant regeneration
  + re-validation by the scene/replay builder before any certification claim; overlays were
  not regenerated. Scene XMLs still embed v1 poses until `build_scenes.py` is re-run.
- ep779/ep784 repairs rest on the FK anchor alone (σ ≈ 3–4.3 cm) — a verifier should
  independently confirm or refute them from the frames before tool_insert re-validation.
