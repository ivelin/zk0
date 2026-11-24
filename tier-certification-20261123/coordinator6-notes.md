# Successor coordinator #6 notes — Gemini certification stages 3–5 resume run

Run: scheduled job gemini-cert-stages-resume-20261130, executed early on
founder order 2026-11-23 ~20:06 CT (RUN CONTEXT: Gemini group verified
33% remaining; do not wait for the Nov 30 reset). This worker ran in
PARALLEL with coordinator #5's own execution of the same founder order;
the two drivers raced on the same DGX prompt/out files for stage 3.
Outputs were identical (same .raw files); #5 merged runs 1–3 and
committed 1abe44d. This note records THIS run's verified delta and the
reconciled resume point. No verdict was resolved; everything stays
PENDING HUMAN APPROVAL.

## Quota (verified at source, agy -p "/usage", model gemini-3.8-flash-high)
- Start of this run: Gemini group 33% remaining, reset 2026-11-30T21:41:45Z
  (Claude/GPT group 100%).
- Burn rate observed (coordinator #5's accounting, confirmed by this
  run's checks): ~9.3% of the weekly group per 25-episode PNG run.
- After run 4 this run's driver usage check read 0%; a direct re-check
  confirmed 0%. Driver stopped per the <15% stop rule (QUOTA_STOP on
  the DGX: "quota-low stop at 0% after gemini_train250-cable_clip-2").
  Guard rundir-gemini-stages345 exited rc=0 at 20:19:57 CT.

## Stage 3 — 1h training set (250 eps): 100/250 merged after this run
- Runs completed (all rc=0, 25/25 verdict lines each, model
  gemini-3.8-flash-high recorded per episode in register):
  gemini_train250-block_stack-1, block_stack-2, cable_clip-1 (merged by
  #5, commit 1abe44d) + gemini_train250-cable_clip-2 (merged by THIS
  run into local register.csv; backup register.csv.bak-pre-gemini-stages345).
- Totals for the 100: Gemini 99 PASS / 1 FAIL (ep604, concurring with
  primary + Grok). Agreement: vs primary 100/100; vs Grok 99/100
  (ep573: Gemini PASS = primary, Grok FAIL — already contested, stays
  contested; NO new contested episodes from this run).
- block_stack-1 raw contains duplicate verdict lines for ep580–584
  (both emissions PASS, evidence wording differs slightly); merge took
  first occurrence in this run, last in #5's — verdicts identical.
- Register Gemini coverage after this run's merge: 101/2,059 rows
  (76 committed in 1abe44d + 25 cable_clip-2 pending the commit below).
- Sheets: train250 PNGs 250/250 on DGX (gemini/sheets-train250/).

## Stages 4–5 — NOT started (quota exhausted at 0%)
- Canonical prompts are coordinator #5's gen_gemini_stage45.py set
  (local gemini-staging/prompts/: gemini_spa-01..48,
  gemini_rc-cube-drawer-01..04, gemini_rc-screwdriver-01..04,
  gemini_rc-stack-bowls-01..04, gemini_rc-stack-cubes-01..04,
  gemini_bb-push-green-01..02, gemini_bb-flip-left-01; run lists
  gemini-staging/runs-gemini_*.txt). Their sheets (spa/rc/rc2 PNG)
  still need shipping per coordinator5-notes.md.
- This run additionally generated a grok-prompt-derived prompt set
  (85 runs) and PNG-converted ALL register sheets on the DGX
  (gemini/sheets-all/: 1,809/1,809 PNG). That set is a fallback only;
  do not run both sets for the same episodes (duplicate quota burn).

## EXACT RESUME POINT (next quota cycle, reset 2026-11-30T21:41:45Z)
1. Stage 3 remaining — 6 runs / 150 eps: gemini_train250-eye_drops_to_basket-1,
   eye_drops_to_basket-2, ring_insert-1, ring_insert-2, tool_insert-1,
   tool_insert-2. Prompts + PNG sheets already on the DGX; driver skips .done.
2. Stage 4 — Spa 1,200 + RC cube-drawer/screwdriver 200 (56 runs, #5's prompts).
3. Stage 5 — rc-stack-bowls/cubes 200 + bb-push-green 39 + bb-flip-left 20
   (11 runs, #5's prompts) + ArmnetBench held-out 150 (6 runs; this run's
   grok-derived gemini_arm-heldout-* prompts + gemini/sheets-all/ PNGs).
- At ~9.3%/run, one weekly cycle (~100% after reset, stop at <15%)
  covers roughly 9 runs — expect multiple cycles; sequence strictly
  in stage order and re-check /usage between chunks.
