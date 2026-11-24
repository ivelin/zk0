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

## Protocol addition — contested-episode narration (founder, 2026-11-23 21:50 CT)
From this date, judge reads for contested episodes must include, beyond the verdict: (a) a chronological trajectory walkthrough and (b) anomalies/insights observed — the founder uses these to verify the judges actually perceive the videos. Recorded as Amendment 3 in criteria-frozen.md and added to the pending source briefs (busybox-multitask, cup-nesting, df-testtube, rc-upright). Applies to all NEW contested-episode judging (Wave 1+ sources and any later Wave-0 judging); delivered verdicts are not re-opened. A separate narration pass over the held-out contested set is run from ~/workspace/founder-tiebreak-20261123/narrations/; this program's register work is unchanged by that pass.

## FOUNDER RULING — GROK TRUST SUSPENDED (2026-11-23 21:53 CT)
Verbatim: "We can no longer trust Grok until it gets a major AI model upgrade. Let's check back next month." Trigger: Grok's lone PASS reads on the contested held-out cable episodes (2020/2042/2046/2092) described the cable as seated when the full videos show it loose on the mat; the founder overruled Grok on all four from the full videos. Applied to this program the same evening (criteria-frozen.md Amendment 4): Grok is no longer a certifying leg; all 1,900 register rows carrying a Grok read are flagged "grok read DISTRUSTED (founder ruling 2026-11-23)" in their notes (reads preserved, not deleted); no new Grok runs will be made — certification proof now rests on Muse + Gemini + the founder's eyes; contested resolution no longer cites Grok agreement, and 99 rows whose only dissent was Grok's were cleared of contested with the note "Grok dissent — distrusted per founder ruling 2026-11-23" (contested 123 → 24; the remaining 24 involve non-Grok disagreements). The pending Grok legs for rc-stack-cubes, bb-push-green and bb-flip-left are cancelled; those sources certify on Muse + Gemini + founder and their review packs proceed without Grok. Trust re-check: 2026-12-23, restoration only on the founder's yes. Standing duty for the remaining Gemini merges: if Gemini dissents from primary on a row cleared under this ruling, the row is re-flagged contested.

## AGY MODEL MENU (verified live at AI Pro tier, agy models on the DGX, 2026-11-23 ~21:58 CT) + ROUND-ROBIN RULE (founder, 2026-11-23 21:54 CT)
Selectable at the current tier: Gemini 4 Neon (low/medium/high/xhigh) · Gemini 4 Carbon (low/medium/high) · Gemini 4 Argon (low/medium/high) · Gemini 4 Hydrogen (low/medium/high/xhigh) · Orchestrator (Neon/Carbon/Argon) · Gemini 3.8 Flash (low/medium/high) · Claude Opus 5.5 (low/medium/high) · Claude Sonnet 5.5 (low/medium/high) · GPT-OSS 120B (medium). The founder's names map exactly: neon = gemini-4-neon, carbon = gemini-4-carbon, opus 5.5 = claude-opus-5-5.
ROUND-ROBIN for the agy leg, effective 2026-11-23 (each at its top setting), cycling per 25-episode judging run: 1) gemini-4-neon-xhigh → 2) gemini-4-carbon-high → 3) claude-opus-5-5-high → repeat. Implemented in gemini/driver-gemini-rr.sh: the model for a run is assigned deterministically by its line number in the runs file ((line-1) mod 3), done-skips included, and written to gemini/out/<run>.model so the exact model is recorded per episode in the register at merge (auditable). gemini-3.8-flash is skipped from the rotation (superseded generation; its earlier reads stand, model-stamped "gemini-3.8-flash-high (agy 1.5.15)"). Quota groups: Neon/Carbon draw on the Gemini Models group; Opus 5.5 on the Claude/GPT group — the rotation also spreads load across both groups; /usage is checked between stages.

## GRIPPER-RELEASE STANDARD (founder, 2026-11-23 22:05 CT) + retroactive flagging
New program-wide standard (criteria-frozen.md Amendment 5): a certified PASS requires the gripper VISIBLY OPEN / released at episode end; an episode ending with the gripper still closed on the object is INDETERMINATE, regardless of how the object sits. Origin: the founder revised his held-out block_stack ep489 ruling from PASS (22:03 CT) to INDETERMINATE (22:05 CT) — "Gemini was correct. The gripper was still closed at the end... We need to see grippers open ideally." Both rulings are kept in the record, the revision marked as superseding.
Retroactive scan (original verdicts untouched; separate release_standard_flag column): register — 3 PASS rows flagged (armnetbench:573 block_stack [the founder-adjudicated ep573 — queued for a re-look under the new standard; its adjudication is not overwritten], spa-bench:1072 and spa-bench:1133, both Relative Size, "gripper closed"); training-250 subset of the register: 1 (ep573). Held-out eval 100 (verdicts_master.csv): 6 flagged — ep489 (founder-ruled), block 453/499/514 (the primary judge's own borderline flags), block 557 and tool 784 ("gripper still" in evidence). Full list: release-standard-queue-20261123.md — this is the founder-review queue for the standard. Method limit, stated plainly: the scan reads judging rationale text only; an episode whose final frames show a closed gripper without any textual mention is not caught — the queue is a starting set, not a complete census.
New judging: the release clause was appended to all 109 un-run agy prompt files (stages 4–5) on the DGX; the in-flight run at the change (gemini_spa-14, carbon-high) kept its original prompt and is covered by the scan approach. All future primary/cleanliness briefs inherit the standard via criteria-frozen.md Amendment 5.

## CABLE_CLIP RE-SPLIT (founder decision, 2026-11-23 22:11 CT)
The original held-out cable set (20/20 FAIL) is removed from the active eval set (marked in verdicts_master.csv; FAIL records stay in the audit history). New held-out cable eval set drawn from the certified training pool: eligibility = unanimous PASS across all legs + not release-flagged + not contested + not the qualified PASS ep2201 → eligible pool 49 of 50 (only ep2201 excluded). Fixed-seed draw (seed 20261123, random.sample over the index-sorted pool): 2164, 2169, 2170, 2183, 2190, 2193, 2194, 2195, 2200, 2208 — all cleanliness CLEAN (grades recorded in the manifest). The remaining 40 episodes are the training set for the certified-data redo; rationale on record: the redo trains only on the 40, so the move does not contaminate it. Manifest: cable-resplit-manifest-20261123.md (seed, eligible pool, drawn 10, remaining 40). Register rows for the drawn 10 carry a MOVED-TO-HELD-OUT note. Scope: cable_clip only.
Spot-check #4 (same ruling batch): training cable ep2159 — founder PASS, concurs unanimous (632/632); register human_status updated.

## UNANIMOUS-TRUST RULE (founder, 2026-11-23 22:22 CT) — applied
Muse (Muse Wonder) and Gemini are the approved judges (Grok set aside until the December re-check). Unanimous Muse+Gemini agreement is TRUSTED and certified without founder review; the founder sees only (a) Muse≠Gemini disputes (INDETERMINATE on either side counts) and (b) gripper-release-standard flags. Held-out 100 (recomputed from the audit columns; matches the founder's count): 92 unanimous-trusted — 78 rows marked "CERTIFIED — unanimous Muse+Gemini agreement (founder rule 2026-11-23)", 8 already founder spot-check approved, 4 already founder-adjudicated FAIL, and the 2 release-flagged unanimous passes (block 453, block 514) held in the founder queue; 8 disputes — 489 (adjudicated INDETERMINATE) and 2020 (adjudicated FAIL, removed from eval) closed; OPEN founder queue = block 499, block 557, tool 784, tool 824, eye 1643, eye 1760 + release-flagged block 453, block 514 (8 items). Training 250 (stage-3 Gemini complete, so the split applies now): 244 CERTIFIED by unanimous agreement (including ep604 as a certified FAIL), 4 founder spot-check approvals stand (cable 2159/2179/2189/2206), 2 remain pending — ep573 (release-standard flag) and ep2201 (verifier INDETERMINATE; the qualified PASS). Register sources receive the same split as their Gemini coverage lands; this is now a standing duty at every Gemini merge.
Release-flag precision (updated 2026-11-23 22:25 CT): founder-reviewed held-out flags — ep489 TRUE positive (gripper still closed; ruling revised to INDETERMINATE), ep514 FALSE positive (founder: "Clean pass. Gripper open at the end." — flag cleared in the record). The flag is a review trigger only; the founder's eyes are the arbiter. Remaining founder queue (held-out): disputes block 499, block 557, tool 784, tool 824, eye 1643, eye 1760 + release-flagged block 453.
Judge-precision log (2026-11-23 22:27 CT): ep784 — Gemini's FAIL rested on a gripper-state claim ("gripper remains holding the tool… without releasing it") that the full video contradicts; the founder ruled PASS ("Clean pass."), the primary read stands, and the release flag sourced from that claim is a false positive (cleared). Running precision tally for gripper-state-based challenges: Gemini gripper reads — ep489 CORRECT (founder revised to INDETERMINATE on it), ep784 WRONG. Release flags founder-reviewed: 1 true positive (ep489), 2 false positives (ep514, ep784). Single-judge gripper-state claims remain a dispute-queue matter by design.
Judge-precision log (2026-11-23 22:31 CT): ep557 — Gemini's FAIL rested on a final-position claim ("orange block slipped off and rests on the table beside the purple block") that the full video contradicts (orange on purple, released by 92%, stable at 99%); founder PASS, primary stands, release flag cleared (false positive #3). PATTERN (founder's note, logged verbatim in substance): Gemini has misread 2 of the 3 disputes the founder adjudicated tonight — ep784 (gripper-state) and ep557 (final-position); on ep499 its gripper observation was correct. Release flags founder-reviewed: 1 true positive (ep489), 3 false positives (ep514, ep784, ep557).
Judge-precision log (2026-11-23 22:32 CT): ep824 — third Gemini 3.8 Flash misread tonight (784 gripper-state, 557 final-position, 824 final-position: "tool dropped on toolbox surface leaving empty slot unfilled" vs the video's seated tool at 99%, gripper empty); only ep499's Flash gripper observation held up under founder review. Founder directive (22:32 CT): Flash 3.8 is not good enough — the held-out 100 Gemini leg is re-run on the advanced round-robin models, with the Flash reads preserved in their own column set and the founder's adjudications final and unre-opened.
Judge-precision log (2026-11-23 22:35 CT): ep1643 — a MUSE PRIMARY (Muse Wonder) misread, logged symmetrically with the Flash misreads: the primary's FAIL rationale ("tray tipped, then clearly empty… platform empty, box gone") was wrong on the full video; the founder ruled PASS, siding with Gemini Flash ("bottle resting inside basket with gripper released"). Asymmetry on record: the primary judged from 6-timepoint contact sheets while the founder saw full video — sheet sampling can miss an end state that settles between sampled frames. Misread tally tonight: Gemini Flash 3 (784, 557, 824), Muse primary 1 (1643), Gemini Flash gripper-correct 1 (499).

## JUDGING-INPUT STANDARD (founder, 2026-11-23 22:39–22:40 CT)
Full video if the advanced models can ingest it; otherwise dense sheets: start frames (scene setup) + rising density toward the end + final frames always included (~16 timepoints). Uniform 6-frame sheets are retired program-wide. Input mode is recorded per episode alongside the model stamp. Video test (blind, gemini-4-neon-xhigh): ep824 — agy accepted the full mp4 and returned PASS with accurate end-state evidence (founder ground truth PASS), 146 s/call. Coordinator's application: video for the dispute + held-out re-judge tiers; dense sheets for bulk register legs (volume), stated for the founder's countermand in the morning report.
VIDEO TEST — FULL RESULT (2026-11-23 ~22:47 CT): both blind test calls on gemini-4-neon-xhigh read the founder's ground truth correctly. ep824: PASS in 146 s (tool seated flush, release seen ~0:12–0:13, gripper empty at end). ep1643: PASS in 189 s (bottle released into the basket ~5.0–5.2 s, resting inside through the final frame; gripper "completely empty and visibly open" at end) — siding with Flash over the Muse sheet misread. CONCLUSION: agy accepts native mp4 input; video judging is CORRECT on both known-misread episodes, including the final gripper state. Video re-judge of the 350 (held-out 100 + training 250) launched 2026-11-23 ~22:52 CT: three parallel drivers, per-episode calls alternating gemini-4-neon-xhigh / gemini-4-carbon-high (Gemini-family qualifying second judges per Amendment 8), composites built frame-exact from the dataset (front + top inset, frame counts verified per episode). Founder adjudications remain final — the re-judge adds reads; it does not re-open them.
NAMING (founder, 2026-11-23 22:46 CT): the primary leg is Muse (Muse Wonder), not Anthropic Claude; labels corrected program-wide (Amendment 8). Opus 5.5 reads remain third reads pending the founder's ruling on whether they count as an independent second judge.
