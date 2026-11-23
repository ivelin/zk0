# Study 001 — Step 4 REMAINDER record (FL rounds 10/20 diagnostic) — 2026-11-23

Executed under the founder's explicit go-ahead, given 2026-11-23 07:04 CT in the
zk0 chat (options card): **"Approve — run the merged eval pipeline on the DGX"**.
That approval covered executing the merged pipeline code (eval gate, trajectory
passes, judging) in the study worktree and the stranded git rebase+push below,
after the previous attempt stopped at an execution-risk check. All figures are
**JUDGED completion (offline replay)**.

## 1. Git repair

- The spec_exp completion commit `2043154` (six study_prep files) was rebased
  onto the gap-closure merge `a136886` in the clean worktree
  `/home/openclaw/muse-work/zk0-step4-merge-20261123` (cherry-pick, no
  conflicts) and pushed via DGX git: origin `study001/fl-substrate-20261121`
  `a136886..a6b323f`. New commit: **`a6b323f074637f1ca44676622ce8553833825e99`**
  (ls-remote verified).

## 2. Eval gate re-run (full 16-checkpoint set)

- Merged `eval_gate_check.py` (from `a136886`) over the 14 existing checkpoints
  + `fl_r10_3cam` + `fl_r20_3cam`, canonical report regenerated at
  `run/eval-traj-20261122/eval_gate_report.json`.
- **GATE VERDICT: PASS (72 PASS, 0 DIFF-WAIVED, 0 DIFF-FAIL)** — includes the
  G-12 standard-load proof per checkpoint and the G-6 active-records hygiene
  check; all 16 checkpoints declare identical input features. The prior
  14-checkpoint report is preserved at
  `rounds1020/eval_gate_report_14ckpt_20261123.json`.

## 3. Trajectory passes (guard-wrapped, breaker logonly)

- `fl_r10_3cam`: exit 0, 100 records + summary; image key map
  front→camera1, top→camera2, wrist→camera3 verified in the guard log.
- `fl_r20_3cam`: first two launches refused by the guard pre-flight
  (whole-box load1 14.89 and 14.07 vs the 14 threshold — the twin-v2 controls
  battery was running concurrently); relaunched when load fell and completed
  exit 0, 100 records + summary, same verified key map. The guard's ceiling
  refusal was honored, never bypassed.

## 4. Judging (scoped to the two new checkpoints)

- Scoped scripts `scripts/step4r/` (glob-filtered copies of the Step-4 set);
  the 1,400 existing FINAL judgments untouched.
- 200 packets; **47 borderline, all visually reviewed** (8 sheets,
  `judgments/review_rounds1020/`); **8 flips False→True** (R1 ×1, R2 ×2,
  R4 ×5), 1 provisional completed confirmed, no provisional completed
  overturned. One R4 flip (fl_r10 tool_insert ep683) rests on two over-gate
  jumps inside a single isolated excursion — the application documented in
  eval-plan Amendment A7 §3 — and is noted as such in its decision row.
  Sub-rule margins were calibrated against the prior batches' applied ranges
  (computed from the step4 + repair3cam decision files: all prior flips
  single-gate; R2 flips at gripper miss ≤0.068 vs b-only confirms ≥0.125;
  R1 flips at ratio ≤1.076). Two combined-marginal packets were confirmed not
  completed for lack of any combined-gate flip precedent (fl_r10 ring_insert
  ep177; fl_r20 eye_drops ep1746).

## 5. Diagnostic readout

JUDGED completion (offline replay), pooled /100 (block / eye / ring / tool / cable):

| checkpoint | block | eye | ring | tool | cable | pooled |
|---|---|---|---|---|---|---|
| fl_r10_3cam | 0 | 3 | 2 | 8 | 0 | **13** |
| fl_r20_3cam | 0 | 4 | 10 | 8 | 0 | **22** |
| fl_r30_3cam | 0 | 7 | 10 | 9 | 0 | **26** |

Trajectory error (pooled excl. cable_clip) falls 42.84 → 38.08 → 36.49;
median client training loss fell 0.322 → 0.246 → 0.222 over the same rounds.

**§6 conditional:** PEAKED requires a mid-run advantage ≥10 pp — max(r10, r20)
− r30 = −4 pp. WEAK-PEAK (5–9 pp advantage) — no. NEVER-TOOK-OFF requires
r10/r20 within ±5 pp of 26 with an early loss plateau — r10 is 13 pp below and
the loss curve shows no plateau. **The branch is therefore AMBIGUOUS under the
registered definitions → default GPU order Q-A → Q-B → Q-D**, with the actual
signature logged: completion climbs monotonically (13 → 22 → 26), i.e. FL was
still improving at round 30; it neither peaked nor flatlined.
**Q-STEPS trigger:** r20→r30 climb is +4 pp — inside §6's own ±5 pp flat band,
so the trigger's "real margin" is **not** met; Q-STEPS stays at its provisional
priority pending the continuation re-sort's own application.

## 6. Independent verification — CERTIFY-WITH-FINDINGS (2026-11-23)

Separate-context verifier; work files in
`run/eval-traj-20261122/rounds1020/verification_step4r/`:

1. **Table recomputation — exact agreement, zero discrepancies** across all
   16 rows × 13 columns, recomputed from the 1,600 FINAL judgment JSONs and
   record summaries (fl_r10 13/100, fl_r20 22/100, fl_r30 26/100, cexp 70/100
   all reproduce).
2. **Gate coverage — PASS**: 72/72 checks, all 16 checkpoints covered.
   *Minor finding:* the gate report carries no weights/config hash field, so
   checkpoint correspondence was established by direct hashing instead —
   r10 `aa2e02e0…1089` and r20 `6b472ba9…ac0a`, both exact matches to the
   prep record and to the as-saved round weights.
3. **Blind re-judge — 24/24 agreement** (seed 20261123; 12 per new checkpoint,
   stratified; 10 borderline + 14 non-borderline; calls recorded before
   reading stored finals; contested packets resolved identically, incl. the
   R2 flip at ring ep165 and the repeated-jump confirm at tool ep883).
4. **Review-trail integrity — clean**: 47 borderline ↔ 47 decision rows 1:1;
   exactly 8 flips reflected; the other 153 new packets FINAL with
   final == provisional; all 1,400 prior judgment files untouched (mtimes
   predate the run) and their rows match the preserved 14-checkpoint table.
5. **Prior rows certified**: the 14 founder-sequence rows (incl. sequential
   and spec_exp ×5) are now independently certified by the same recomputation
   — their pending certification is closed.

## 7. Eval record commit

- Copies committed to `study001/fl-substrate-20261121` under `study_prep/`:
  the 16-checkpoint gate report, the updated completion table (md + csv),
  the judgments index (1,600 FINAL judgments, per-file SHA-256), and this
  record. Commit hash in the delivery report.
