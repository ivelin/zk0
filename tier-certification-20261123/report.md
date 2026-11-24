# Tier certification program (Wave 0) — report

Status language: nothing in this program is called "certified". Every
episode remains **PENDING HUMAN APPROVAL** until the founder approves
the review packs and the Gemini leg closes. **Gemini/agy leg: OWED**
— agy is absent on all machines checked; the DGX `gemini` CLI exists
but is unauthenticated; no credentials were improvised.

## DF §8 criteria freeze confirmations

- **Test_Tube_Nesting_Task** (DecisionFacts, rev 8c161ed271870ede4780631a018dbad9d697b870):
  task strings read from the downloaded `meta/tasks.parquet` on
  2026-11-23 by coordinator #3, BEFORE any judging of this source:
  - task 0: "Pick and lift the plate, transport it, and seat it flat
    in the nest with all four corners down."
  - task 1: "Pick the test tube and properly place it in the tube tray."
  These match criteria-frozen.md §8(b) as drafted (plate seating /
  test tube in tray). §8 is FROZEN for this source as of this log
  entry; judging may begin once the download completes.
  Dataset facts at the pinned revision: 160 episodes, 30 fps,
  cameras observation.images.cam_front + observation.images.cam_top,
  264,190 frames.
- **Pipe_And_Joint_Sorting_Task**: task-string confirmation still
  pending — the download has not landed yet (one transient failure,
  one queued retry). §8 freeze for this source completes only when
  its strings are read and logged here, before any judging.

## Results — coordinator #3 completion summary (2026-11-23)

**Everything in the register remains PENDING HUMAN APPROVAL. No
episode in this program is certified until the founder's human-eyes
approval lands. No training has used or may use this data.**

Register: 1,800 rows. Grok coverage 1,800/1,800. Gemini coverage:
held-out eval 100 (in its audit record) + 2 register rows; stages 3–5
pending the 2026-11-30 quota reset (see Gemini status). Contested
register-wide: 51 episodes — none resolved by any coordinator; all go
to the founder with every read attached.

### Per-source results (landed sources, pipeline complete)

- **ArmnetBench training 250** (1h program carry-in): primary
  249 PASS / 1 FAIL (ep604, excluded). Cleanliness 246 CLEAN /
  3 RECOVERED / 1 INCIDENT. Blind re-judge 49/50 (98.0%). Grok
  249/250 (99.6%). Gemini: ep573 read PASS (stage 1); the full 250
  are Gemini stage 3, pending the quota reset. ep573 founder-
  adjudicated PASS 2026-11-23 (Grok's FAIL stands as the dissenting
  read). Contested: ep573.
- **Spa-Bench 1,200**: primary 1,193 PASS / 7 FAIL (ep110, 120, 159,
  497, 518, 859, 891). Cleanliness 1,151 C / 49 I / 0 R. Blind
  re-judge 239/240 (99.6%). Grok 1,166/1,200 (97.2%). Contested 35:
  ep859 (primary FAIL / re-judge PASS / Gemini PASS) + 34 primary-
  PASS episodes Grok read FAIL or INDETERMINATE, incl. a contiguous
  block ep204–224 (one task family — flagged for the founder as a
  possible batch-level Grok artifact or a genuinely failed demo
  family; the reads stand as recorded either way).
- **RoboColosseum cube-drawer 100**: primary 100 PASS. Grok 90/100 —
  10 INDETERMINATE, all contested. Re-judge (RC combined) 40/40.
- **RoboColosseum remove-screwdriver 100**: primary 100 PASS. Grok
  100/100. RC combined cleanliness: 179 C / 20 I / 1 R (ep19
  screwdriver dropped mid-carry, re-grasped, placed).
- **ArmnetBench held-out tasks 150** (eye_drops_to_shelf 50,
  tool_removal 50, cable_unclip 50; criteria §6 frozen before
  judging): primary 146 PASS / 4 FAIL (toolrem ep1205; unclip
  ep2479, ep2486, ep2498). Cleanliness 127 C / 23 I / 0 R — the
  unclip INCIDENTs are the charger/adapter unit tipped or displaced
  by the cable pull (20 of 50 unclip episodes). Blind re-judge
  29/30 (96.7%). Grok 145/150 (96.7%). Contested 5: ep1205 (primary
  FAIL / re-judge PASS / Grok PASS), ep2465 (primary PASS / Grok
  FAIL), ep2479 + ep2486 + ep2498 (primary FAIL / Grok PASS).

### Held-out eval set (separate record)

Primary 77 PASS / 22 FAIL / 1 INDETERMINATE; blind verifiers 95% and
100% on the seeded sample; Grok 80/100 agreement; **Gemini leg
complete for this set**: 100/100 reads (73 PASS / 27 FAIL), vs
primary 93/100, vs Grok 82/100. Contested after all three model
families: 22 episodes. Full record + Gemini addendum:
heldout-episode-certification-audit-20261123.md (Study 001 notes).

### Gemini leg status (agy v1.5.15, model pinned gemini-3.8-flash-high)

Stages 1 (contested-first: held-out 20 + register ep859 + ep573) and
2 (held-out 100) COMPLETE, all runs rc=0, verdicts merged with the
model name recorded per episode. The leg STOPPED on quota discipline
after stage 2: Gemini weekly group at 32% remaining, shared with
Cos's machine, reset 2026-11-30T21:41:45Z. Stages 3 (1h training
250 — prompts generated, sheets to ship), 4 (Spa-Bench +
RoboColosseum) and 5 (new sources as they land) resume after the
reset. A #bot-hub note is owed about this week's Gemini consumption.
ep2020 note: Gemini produced three mutually inconsistent reads of
this borderline cable episode across the micro-test, the contested
batch and the full-set batch (FAIL-equivalent / PASS / FAIL) — all
recorded in the held-out audit addendum.

### Review packs (review-pack/, sheets + README per source)

spa-bench (7 FAIL, 35 contested, seeded PASS samples by cleanliness
subset), rc-cube-drawer (10 contested), rc-screwdriver (samples),
armnetbench-heldout (4 FAIL, 5 contested, samples). Seed 20261123,
cap 12 per subset. The ArmnetBench training-250 pack derives from
the 1h program's records.

## Gemini leg — stages run early on founder order (2026-11-23 ~20:05 CT)

The founder ordered the remaining Gemini stages run immediately
instead of waiting for the Nov 30 quota reset (live check: Gemini
group ~32-33% remaining). Coordinator #5 ran the leg in stage
order. **Stage 3 is PARTIALLY complete; the leg stopped on the
10%-remaining quota floor** (/usage read 5% after three runs;
~9.3% of the weekly group per 25-episode PNG run; reset
2026-11-30T21:41:45Z):

- Stage 3 (1h training 250): 3/10 runs done (block_stack 50 +
  cable_clip 25 = 75 episodes merged, model-attributed). Gemini:
  74 PASS / 1 FAIL (ep604 — concurring with primary and Grok).
  Agreement vs primary 75/75; vs Grok 74/75 (ep573: Gemini PASS,
  with primary + founder over Grok's FAIL). Remaining: 7 runs /
  175 episodes (cable_clip-2, eye ×2, ring ×2, tool ×2) — prompts
  and PNG sheets already on the DGX.
- Stage 4 (Spa-Bench 1,200 + RoboColosseum 200): NOT started.
  Prompts generated (gen_gemini_stage45.py); sheets not yet shipped.
- Stage 5 (rc2 sources, 259 episodes): NOT started. Prompts
  generated; sheets not yet shipped.
- The Nov 30 scheduled run remains the fallback for everything
  above; coordinator5-notes.md carries the exact resume point.
  Register Gemini coverage: 76/2,059. Nothing about the contested
  flags changed — Gemini overrules nobody.

## Update — coordinator #5 (2026-11-23 evening)

Register: 2,059 rows, all PENDING HUMAN APPROVAL. Grok coverage
1,900/2,059. Register-wide contested: 123 (none resolved).

- **RoboColosseum stack-bowls 100**: primary 100 PASS; cleanliness
  100 CLEAN; re-judge 20/20 (100%). Grok leg merged: agreement
  50/100 — runs 01/02 read PASS on ep0-49; runs 03/04 read FAIL on
  ALL of ep50-99, with evidence describing the same end state the
  PASS runs called a completed nested stack. Flagged for the
  founder as a probable batch-level Grok artifact (same pattern as
  the Spa ep204-224 block); the reads stand as recorded and all 50
  episodes are contested. Review pack built (review-pack/
  rc-stack-bowls/: 50 contested, 12 seeded PASS+CLEAN samples).
- **RoboColosseum stack-cubes 100**: primary 100 PASS; cleanliness
  86 C / 13 I / 1 R; re-judge 20/20 (100%); 2 contested (third
  independent read FAIL vs primary PASS on ep72/73). Grok leg
  BLOCKED (below); review pack pending its Grok leg.
- **BusyBox push-green 39**: primary 39 PASS; 39 CLEAN; re-judge
  8/8 (100%). Grok leg BLOCKED; pack pending.
- **BusyBox flip-left 20**: primary PASS 20/20, but blind re-judge
  FAIL 4/4 sampled and two further independent reads FAIL 20/20 —
  all 20 contested for the founder (primary stands). Cleanliness
  UNGRADED after two attempts (both graders returned pass/fail
  reads instead of grades; reads preserved in verdicts/). Grok leg
  BLOCKED; pack pending.
- **Grok leg blocker**: the 7 remaining rc2 runs (stack-cubes ×4,
  push-green ×2, flip-left ×1) failed twice — the original run and
  the one authorized re-run — with HTTP 402 "Grok Build usage
  balance exhausted". No further retries were made. The Grok legs
  (159 episodes) and the three review packs resume when the Grok
  Build balance is restored.
- **Downloads**: rc-upright-bottle still in flight under HF
  rate-limit backoff (316 MB, 12 partial files; monitor only).
  BusyBox multitask (132) and Cup Nesting (200) queue behind it.
  DF test-tube: verified 7 video files still missing (6 cam_front +
  1 cam_top) against its episodes manifest; its remainder fetch and
  DF sorting's single queued retry run sequentially once crawler
  load drops. DF test-tube §8 criteria are FROZEN (above); DF
  sorting §8 freeze still waits on its task strings.

### Blocked sources (~742 rows) — downloads

HF rate-limit backoff stalled the crawlers for most of 2026-11-23
(sockets CLOSE-WAIT, sizes static for hours; no restarts, no retry
storms, per program rule). At report time: rc-stack-cubes (208M)
and rc-upright-bottle (217M) downloads in flight; BusyBox multitask
(132), BusyBox singles (59), Cup Nesting (200), DF sorting (141)
not landed; DF test-tube (160) partially landed (§8 criteria FROZEN
above; judging waits on the complete download). DF sorting's §8
freeze is still pending its task strings; its one queued retry
remains queued. These sources pipeline identically as they land —
see coordinator3-notes.md for the successor checklist.

### Process notes (for the audit trail)

- **Held-out verdict-file correction (2026-11-23 ~18:50 CT)**: the
  first-written arm_heldout verdict files claimed 150/150 PASS and
  150/150 CLEAN; the judge workers' delivered verdicts say 146/4
  and 127 C / 23 I. Files rewritten verbatim from the handoffs
  (originals preserved in verdicts/superseded-20261123/), register
  rows rebuilt via add_source.py, counts re-verified. Details in
  coordinator3-notes.md.
- Four Grok remainder batches (spa-coun-07: 14 eps, spa-obje-17: 19,
  spa-ordi-06: 2, spa-size-06: 15) were re-run once under a mistaken
  truncation assumption — they are remainder batches whose verdict
  counts exactly match their prompt episode counts. First attempts
  are canonical in the register; second attempts are archived on
  the DGX (grok/out/*.raw.attempt2). One read differs between
  attempts: spa ep786 (ordi-06) FAIL in attempt 1, PASS in
  attempt 2; attempt 1 stands.
- Rotation record: primary + re-judge reads are Muse judge
  subagents; second full reads are Grok Build CLI v1.0.88; tie-break
  reads are Gemini (gemini-3.8-flash-high via agy). Model attribution
  is recorded per episode in the register (primary_model /
  verifier_model / grok_model / gemini_model columns). Items still
  contested after Gemini get their next read from a fourth family
  (Muse Code, via the parent) or the founder — never from a family
  already involved in that item.
