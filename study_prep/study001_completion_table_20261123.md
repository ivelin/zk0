# Study 001 synthetic eval — JUDGED completion (offline replay) + trajectory error

Binary completion is JUDGED from open-loop predicted trajectories against the
registered rubric (eval-plan.md) — NOT physical completion, NOT sim rollout success.
Trajectory error = mean per-step L2 distance (action units, degrees) between predicted
and recorded actions on held-out episodes. cable_clip error is REFERENCE-FLAGGED:
its held-out references are recorded FAILED executions (no successful held-out
execution of cable_clip exists); it is excluded from pooled error aggregates.
Sequential checkpoint: included.

| checkpoint | block_stack | eye_drops_to_basket | ring_insert | tool_insert | cable_clip | pooled |
|---|---|---|---|---|---|---|
| base | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/100 (0%) |
| spec_block_stack | 13/20 | 6/20 | 2/20 | 5/20 | 0/20 | 26/100 (26%) |
| spec_eye_drops | 1/20 | 19/20 | 0/20 | 12/20 | 0/20 | 32/100 (32%) |
| spec_ring_insert | 0/20 | 4/20 | 19/20 | 4/20 | 0/20 | 27/100 (27%) |
| spec_tool_insert | 0/20 | 7/20 | 3/20 | 18/20 | 0/20 | 28/100 (28%) |
| spec_cable_clip | 0/20 | 6/20 | 0/20 | 8/20 | 0/20 | 14/100 (14%) |
| cexp | 17/20 | 18/20 | 19/20 | 16/20 | 0/20 | 70/100 (70%) |
| fl_r10_3cam | 0/20 | 3/20 | 2/20 | 8/20 | 0/20 | 13/100 (13%) |
| fl_r20_3cam | 0/20 | 4/20 | 10/20 | 8/20 | 0/20 | 22/100 (22%) |
| fl_r30_3cam | 0/20 | 7/20 | 10/20 | 9/20 | 0/20 | 26/100 (26%) |
| sequential | 1/20 | 9/20 | 10/20 | 12/20 | 0/20 | 32/100 (32%) |
| spec_exp_block_stack | 12/20 | 4/20 | 4/20 | 11/20 | 0/20 | 31/100 (31%) |
| spec_exp_eye_drops | 1/20 | 18/20 | 0/20 | 6/20 | 0/20 | 25/100 (25%) |
| spec_exp_ring_insert | 0/20 | 5/20 | 17/20 | 9/20 | 0/20 | 31/100 (31%) |
| spec_exp_tool_insert | 0/20 | 4/20 | 1/20 | 15/20 | 0/20 | 20/100 (20%) |
| spec_exp_cable_clip | 0/20 | 9/20 | 3/20 | 8/20 | 0/20 | 20/100 (20%) |

Mean trajectory error (lower = closer to the recorded execution):

| checkpoint | block_stack | eye_drops_to_basket | ring_insert | tool_insert | cable_clip | pooled | pooled excl. cable_clip |
|---|---|---|---|---|---|---|---|
| base | 91.23 | 99.81 | 98.12 | 110.81 | 104.05 | 100.8 | 99.99 |
| spec_block_stack | 17.62 | 63.91 | 36.05 | 72.18 | 63.4 | 50.63 | 47.44 |
| spec_eye_drops | 50.88 | 18.77 | 59.85 | 75.08 | 67.44 | 54.41 | 51.15 |
| spec_ring_insert | 38.99 | 57.15 | 17.36 | 49.28 | 70.32 | 46.62 | 40.69 |
| spec_tool_insert | 48.49 | 61.02 | 59.57 | 17.69 | 82.73 | 53.9 | 46.69 |
| spec_cable_clip | 52.51 | 61.79 | 65.15 | 42.76 | 35.05 | 51.45 | 55.55 |
| cexp | 19.27 | 21.16 | 16.71 | 18.8 | 30.11 | 21.21 | 18.98 |
| fl_r10_3cam | 40.32 | 58.75 | 40.93 | 31.34 | 39.95 | 42.26 | 42.84 |
| fl_r20_3cam | 35.47 | 58.93 | 30.57 | 27.35 | 36.7 | 37.81 | 38.08 |
| fl_r30_3cam | 33.75 | 58.3 | 27.1 | 26.8 | 35.42 | 36.27 | 36.49 |
| sequential | 53.73 | 55.53 | 45.37 | 34.2 | 32.06 | 44.18 | 47.21 |
| spec_exp_block_stack | 19.16 | 61.1 | 38.09 | 60.96 | 40.55 | 43.97 | 44.83 |
| spec_exp_eye_drops | 58.57 | 24.61 | 62.86 | 62.85 | 74.94 | 56.77 | 52.22 |
| spec_exp_ring_insert | 43.04 | 61.96 | 16.78 | 41.16 | 68.7 | 46.33 | 40.74 |
| spec_exp_tool_insert | 45.66 | 62.94 | 50.28 | 19.81 | 64.41 | 48.62 | 44.67 |
| spec_exp_cable_clip | 62.73 | 60.54 | 68.04 | 41.62 | 32.78 | 53.14 | 58.23 |

## Notes

- SUPERSEDED ROW (2026-11-23): the original fl_r30 row (25/100 pooled) was
  measured under unequal eval inputs and is SUPERSEDED. Its records and
  judgments are preserved at superseded_20261122/fl_r30_2view/ in this eval
  directory. Camera caveat (eval-plan Amendment A4): fl_r30's packaged config
  declared only front + top cameras (observation.images.front,
  observation.images.top) - NO wrist camera - while cexp declares three
  cameras (camera1/camera2/camera3) and the specialists declare
  front/top/wrist; each checkpoint was evaluated through exactly the inputs
  its own config declared. The corrected row fl_r30_3cam (26/100 pooled) is
  the same FL round-30 weights with the checkpoint-declaration repair,
  evaluated on identical 3-view inputs as every other row. Corrected
  comparison: FL 26/100 vs C-exp 70/100, Phi = -44 pp. Strict-reading bound:
  one corrected-run flip (fl_r30_3cam tool_insert ep859) reverts under a
  strict reading of sub-rule R4, giving FL 25/100 and Phi = -45 pp; the
  conclusion is unchanged either way.
- Borderline-review disclosure (original 8 checkpoints): 142 borderline
  packets were reviewed and 33 were flipped - ALL 33 flips went False->True
  (the visual review only ever added completions: fl_r30 +9, cexp +9,
  specialists +15 combined). Independent verification (2026-11-22)
  sensitivity check: substituting the verifier's calls on the sampled flips
  moves fl_r30 from 25 to 23 and cexp from 70 to 68 pooled - the headline
  comparison is unchanged.
- Step 4 rows (added 2026-11-23): sequential (the sequential-chain final
  checkpoint) and the five spec_exp_* checkpoints (expert-only specialists,
  3,000 steps each from the pinned base) were evaluated on the same frozen
  100 held-out episodes under the eval gate (eval_gate_report.json: PASS,
  49 checks, full 14-checkpoint set) and judged under the same rubric incl.
  Amendment A7. Borderline review for the six new checkpoints: 113 packets
  reviewed, 21 flipped False->True (R1 x6, R2 x6, R4 x9); no provisional
  completed was overturned. Review trail: judgments/review_step4/ (sheets,
  borderline_list.json, decisions.csv).
- FL rounds 10/20 diagnostic rows (added 2026-11-23, Step 4 remainder):
  fl_r10_3cam (13/100) and fl_r20_3cam (22/100) were evaluated on the same
  frozen 100 held-out episodes under the re-run eval gate
  (eval_gate_report.json: PASS, 72 checks, full 16-checkpoint set) and judged
  under the same rubric incl. Amendment A7. Borderline review for the two new
  checkpoints: 47 packets reviewed, 8 flipped False->True (R1 x1, R2 x2,
  R4 x5; one R4 flip rests on two over-gate jumps inside a single isolated
  excursion, the application documented in A7 sec.3); 1 provisional completed
  confirmed; no provisional completed overturned. Review trail:
  judgments/review_rounds1020/ (sheets, borderline_list.json,
  borderline_features.json, decisions.csv). Diagnostic readout: judged
  completion climbs monotonically 13 -> 22 -> 26 across rounds 10/20/30 -
  no mid-run peak; FL was still improving at round 30 on judged completion,
  trajectory error (42.84 -> 38.08 -> 36.49 pooled excl. cable_clip) and
  training loss alike.
