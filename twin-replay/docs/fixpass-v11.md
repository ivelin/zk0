# Twin-Suite v1.1 — FIX-PASS Report (builder: fix-pass, single sequential writer)

Workdir: `/home/openclaw/muse-work/zk0-study001-twin-20261122/`
Applied: 2026-11-22, DGX clock — all changelog entries FP-1…FP-13 stamped
2026-11-22 19:22 CST, each from a fresh `date` read taken immediately before
its append. CPU-only, `nice -n 10`, DGX training undisturbed. No full
re-validation was run (a separate runner owns that on this landed state).
Values / thresholds / tolerances otherwise UNCHANGED; all changes global,
no per-episode conditionals added.

## Per-fix results

### FP-A (V11-01, BLOCKING) — cable clip reference — DONE
- `code/build_scenes.py:145` — `make_draws` cable_clip jittered bodies
  `["clip_block", "cable"]` → `["cable"]`. clip_block is a fixture/reference
  (like ring's peg+board, changelog #12's stated principle) and stays at
  nominal; the manipulated cable keeps its jitter.
- `code/replay.py:461-475` — `evaluate()` cable branch now reads the
  clip_block pose from the SIM BODY via `pos("clip_block")` (mirrors how
  ring reads the peg from the sim body, replay.py ring branch), falling
  back to the pose-file/fixtures only if the body is absent. Scoring
  always uses actual scene geometry.
- Smoke (cable ep1939, all 5 variants rebuilt): clip_block XML position is
  the nominal (0.27226967686214654, −0.08705115372165617, 0) in **every**
  variant v0–v4; `make_draws` jitter keys for k≥1 are exactly `["cable"]`.
- Functional proof evaluate() reads the sim pose: same scene evaluated
  unshifted vs with the clip_block body shifted +0.10 m in x in the XML
  only (pose file untouched) — `nearest_seg_to_block_xy_m` moved
  0.0993 → 0.1736 m. The evaluator tracks the sim body, not the pose file.

### FP-B (V11-02) — cable channel tolerances registered — DONE
- `code/twin_params.py:155-165` — new named constants with basis comment:
  `CABLE_CHAN_X_HALF_M = 0.012`, `CABLE_CHAN_Y_EXTRA_M = 0.002`,
  `CABLE_CHAN_Z_HALF_M = 0.012`. Basis documented: built jaw geometry
  (clip_jaw_a/b centres y = ±(CLIP_CHANNEL_W/2 + 0.004), half-extents
  (0.012, 0.004, 0.006) m) — x bound = jaw half-length, y bound reaches
  the jaw inner faces' outer edge, z bound = 2× jaw half-height.
- `code/replay.py:490-491` — cable test consumes the constants.
  **Values unchanged.**

### FP-C (V11-03) — param_table_hash coverage — DONE
- `code/build_scenes.py:32-64` — hash extended; missing files are skipped
  (no crash — verified live: with `code/patch2.py` temporarily renamed
  away the hash computed `ec2675e21523a946`; restored → `38e5a9e7dd1d0330`).
- Changelog FP-3 carries the CORRECTION of #13 (its "every code file the
  pipeline executes" was inaccurate; calibration.py is imported by nothing).
- **New param_hash: `38e5a9e7dd1d0330`** (was `4d8d350b6907427c`).
  Exact coverage, in hash order:
  1. `code/twin_params.py`, `code/build_scenes.py`, `code/replay.py`,
     `code/run_validation.py`, `code/calibration.py`, `code/finalize.py`,
     `code/patch_finalize.py`, `code/patch2.py`, `docs/physics-parameters.md`
  2. Glob `code/builder2_stage*.py` (sorted): builder2_stage1.py,
     builder2_stage1b.py, builder2_stage1c.py, builder2_stage1d.py,
     builder2_stage2.py, builder2_stage2b.py
  3. Data: `scenes/calibration_fk_affine.json`, `scenes/calibration_v2.json`,
     `scenes/detections_px.json`
  4. Sorted manifest (name + contents) of the 100 pose files
     `scenes/<task>__ep*.json` (excluding `*_variants.json` build logs)

### FP-D — evaluate() consumes SECONDARY_THRESH_M — DONE
- `code/replay.py:507,510` — `thr = P.SECONDARY_THRESH_M[task]` for
  block/eye/tool and ring; cable remains exempt (None). No hardcoded
  0.030/0.020 literals remain in the secondary block.

### FP-E — stale rate text — DONE
- `code/replay.py:5` — "(500 Hz)" → "(2000 Hz)" (actual: TIMESTEP 0.0005).
  Historical mentions of v1's 500 Hz in twin_params.py / the changelog are
  accurate as history and were left as-written (noted in FP-5).

### FP-F — v1 F-08 / F-07 changelog bookkeeping — DONE
- FP-6: correction of #3a — the block final-pose code actually selects,
  whenever the base cube's initial detection exists, the candidate nearest
  the base cube's initial pixel; the mean/<80 px rule is only the fallback
  when the base initial pixel is missing.
- FP-7: disclosure of the hand-pick tables — finalize.py HAND_BLOCK_FIRST
  (ep411/530/539×2/546), EYE_FIRST_GROUND (11), EYE_FINAL_GROUND (18),
  EYE_FINAL_LOW (2), TOOL_GROUND (10), TOOL_LOW (3), confidence overrides
  (ep411 cube_green, ep513 cube_purple), hand-read fixture pixels;
  patch_finalize.py CABLE_BLOCK_GROUND (all 20 cable episodes) and the
  per-episode ring peg/board derivation from the final-frame ring
  detection; patch2.py hand picks for ep1780/1783/1814.

### FP-G — ep784 re-anchor — DONE
- My fresh visual-grounding pick of the loose tool in
  `frames/tool_insert__ep784/top_first.png`: **(363, 306) px**
  (visual-grounding tool pick, corroborated by my enlarged-crop read
  ≈(367, 310) px; the silver T/socket piece left/below the wrist plate).
- Cross-check vs the verifier's picks: (340, 305) px → **23.0 px** away;
  (340, 320) px → **26.9 px** away; 23.8 px from their midpoint. Within
  the ~25 px agreement band — corroborating, not contradicting. No stop.
- Converted through the calib_v2 homography (finalize.py TOOL_GROUND
  px2world path; validated: it reproduces ep697's published pose from its
  consensus px exactly): **(363, 306) px → (0.2094, −0.0365)**.
- `scenes/tool_insert__ep784.json` tool: `pose_xyz_yaw` :=
  [0.2094, −0.0365, 0.005, null]; method := visual grounding;
  σ_xy := 3.5 cm (tool grounding σ); confidence medium. Superseded
  FK-anchor pose [0.1808, 0.0192, 0.005, null] preserved in-JSON as
  `pose_v11_fk_anchor_superseded`, with the reason recorded under
  `v11_repair.fixpass_fp_g` (the historical `rule_v2` record is intact).
  New pose is 6.26 cm from the superseded anchor — consistent with the
  verifier's 6.7–7.7 cm offset finding.

### FP-H — ep779 relabel + F4/F5 — DONE
- `scenes/tool_insert__ep779.json` tool: pose VALUE UNCHANGED
  [0.162, 0.0201, 0.005, null]. confidence medium → **low**;
  σ_xy 3.5 → **4.3 cm** (governing radial RMS); method now states
  FK-only, visual status **indeterminate**, not frame-verified.
- `fk_anchor.note` corrected in ep779 (anchor DOES supply the pose) and
  in ep784 (anchor DID supply the v1.1 repair pose; superseded by FP-G,
  now cross-check only). The tool_final cross-check notes were accurate
  and untouched.

### FP-I — ep697 label — DONE
- `scenes/tool_insert__ep697.json`: `v11_repair.chosen_source`
  "fk_anchor_only" → **"grounding_consensus"** (matches rule_v2 in the
  same file + Builder 2's report). Pose values untouched.

### FP-J — F3 rule-v2 change-set completion — DONE
- Changelog FP-11: ep892's repair source also changed under rule v2
  (single V-Env pick → consensus mean (617, 503.5) px); B2-4's text
  understated it. B2-4 left as-written.

### FP-K — V11-OA-1 asset inventory — DONE
- `docs/asset-inventory.md:8` — "SIM-1/SIM-2 era sim validation per
  project record" replaced with the precise citation: the Nov 15 demo +
  audit logs, per the twin-replay completion report's V-Objects note.

### Timestamps (V11-04) — DONE
- Changelog FP-13: B2-5 (stamped 19:12) and COORD-1 (stamped 19:14) were
  post-dated by a few minutes; per file mtime both were appended by
  19:10 DGX time on 2026-11-22. Originals left as-written.

## Changelog
13 entries appended, series FP-1…FP-13 (FP-1=FP-A … FP-13=timestamps;
FP-F split into FP-6/F-07 entries FP-6+FP-7, so FP-G=FP-8, FP-H=FP-9,
FP-I=FP-10, FP-J=FP-11, FP-K=FP-12). All stamped 2026-11-22 19:22 CST
from fresh DGX clock reads. No collision with the main series (#18 was
next free; the FP-* series was used per brief).

## Determinism evidence
Regenerated twice in-process (all 5 variants each); XML SHA-256[:16]
identical across both runs for every variant:
- block_stack ep411: df8eeab6658ea219, 5add40f83c50ed3b, cc1d1d9d90881082,
  d73273f616f4712b, f2a838c26d399c51 — identical ✓
- cable_clip ep1939: 988464292872ae5e, f8e2563f86a4f7df, 33c487d1eeb9770b,
  65bb435a4ad3fa92, 03486e9541c58135 — identical ✓
- tool_insert ep784: 61091562a04238bb, 1dafdffa70f5b000, 0deaacd978a1cd78,
  bf3383969c95b21d, 027ffaa2aa1529c4 — identical ✓
Regenerated XMLs for ep411, ep1939, ep784 were written to `scenes/`
(all variants + base). tool_insert ep779 was also regenerated and
verified identical across two runs (345eafb57fb14e01, cda5ead8d717a0fc,
b046bbce036c6fae, 404fe8df71550862, e8f03feec312799e) because FP-H
changed its σ (which feeds variant draws); its XMLs were written too.
All other episodes' XMLs still embed the previous param_hash in their
headers until the re-validation runner regenerates them — expected;
the hash in any newly built scene is now `38e5a9e7dd1d0330`.

## Not done / notes
- Full re-validation NOT run (per brief — separate runner).
- ep784's historical top-level `v11_repair.chosen_source` still reads
  "fk_anchor_only" — that field records the rule-v2 outcome and was left
  as provenance; the operative fixpass source is recorded in
  `v11_repair.fixpass_fp_g.chosen_source = "visual_grounding_fixpass"`.
- Dead JITTER tier table in twin_params.py (V-Code Check 6 residual) was
  not in the brief's fix list and was left untouched.
- Pre-fix backups of code/ + the three edited scene JSONs + changelog
  were kept on the DGX at /tmp (ephemeral) during the pass.
