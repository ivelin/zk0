# Held-Out Episode Certification Audit — Study 001 (2026-11-23)

**Order:** founder's top-priority rule, 2026-11-23 15:34 CT — every
episode, training AND held-out, must be individually verified and
certified for actually passing its task before use in training or
eval; proof by multiple frontier models (Muse, Grok, Gemini, etc.)
and approved by human eyes; no exceptions, no compromises, no
shortcuts.
**Status of this audit: certification is NOT final.** Every episode
below is marked PENDING HUMAN APPROVAL — the human-eyes leg (the
founder's review of the pack in §9) has not yet been given. The
Gemini leg is likewise outstanding (§5). No episode in this set may
be treated as certified until both legs close.
**Auditor:** a separate-context audit agent; primary judging, blind
verification, and the Grok leg were executed by further separate
contexts / an external vendor CLI (below). No eval or training
recommendations are made here; this is the fact base only.

## 1. Subject — the exact 100 held-out episodes, pinned

Dataset: `armnet/armnetbench_v01_lerobot_so101` at pinned revision
`1db69c82b589c7172ee31a6b8f7ef58d1fc2c663` (Apache-2.0), LeRobot v3.0,
20 fps; on-disk root on the DGX Spark:
`/home/openclaw/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101`.
The set is the frozen held-out split the Study 001 evals consumed,
exactly as recorded in the eval layer's own split record
(`run/eval-traj-20261122/split/split.json`, seed `20261122`,
`disjoint_from_training = true` against the 250 P0 training indices,
asserted programmatically by `build_split.py`).

Construction (eval-plan.md, read at the source): the P0 subsets
consumed ALL 250 teleoperated episodes of the 5 tasks, so no
held-out teleoperated episode exists. The held-out set is recorded
POLICY-ROLLOUT episodes of the same 5 tasks (policy_type ∈ {act,
diffusion, grootn1.7, molmoact2, pi0, pi0.5}; smolvla rollouts
excluded), 20 per task, seeded RNG seed=20261122, stratified across
source policy_type, identical for every checkpoint. For
block_stack / eye_drops_to_basket / ring_insert / tool_insert the
sample was drawn only from episodes with dataset-recorded
success == 1 (pools 34 / 113 / 65 / 52). For cable_clip no
non-teleop rollout of the task ever succeeded (0/270 recorded), so
its 20 were drawn from the recorded-failure pool (pool 240).

Per-task episode indices (dataset episode indices at the pinned
revision):

- block_stack: 411, 453, 476, 479, 489, 498, 499, 513, 514, 525, 526, 527, 530, 531, 539, 545, 546, 548, 549, 557
- eye_drops_to_basket: 1609, 1618, 1630, 1639, 1643, 1724, 1741, 1746, 1747, 1757, 1760, 1778, 1780, 1783, 1798, 1808, 1810, 1811, 1814, 1821
- ring_insert: 7, 11, 30, 31, 33, 48, 50, 160, 165, 167, 168, 177, 179, 188, 195, 198, 227, 240, 242, 253
- tool_insert: 622, 629, 630, 683, 684, 697, 779, 784, 824, 830, 852, 859, 871, 876, 883, 885, 888, 892, 902, 903
- cable_clip: 1896, 1903, 1905, 1939, 1959, 1960, 1968, 1969, 2020, 2032, 2042, 2046, 2059, 2068, 2085, 2092, 2101, 2125, 2130, 2146

## 2. Method

**(a) Recorded fields — read at the source.** Episode metadata
(`meta/episodes/chunk-000/file-000.parquet`) fields `success`,
`success_class`, `policy_type`, `policy_repo_id`, plus the final
frame's `next.reward` / `next.done` from the data parquet at each
episode's `dataset_to_index − 1` row, were read directly for all
100 episodes (`heldout-audit-20261123/recorded_fields.csv`). The
split record's per-episode `recorded_success` and
`source_policy_type` were cross-checked against the metadata for
every episode: 100/100 match. Recorded values: the 80 non-cable
episodes carry success=1 / "successful" / final reward 1.0; all 20
cable_clip episodes carry success=0 / "failure" / final reward 0.0
— the known fact about the held-out cable references is confirmed
at the source, per episode. Unlike the teleoperated demos (whose
success flag is assigned by construction), rollout labels in this
release were human-assigned per the release's label scheme — but
under the founder's rule they are treated as claims to verify,
not as evidence; legs (b)–(d) judge the episodes themselves.

**(b) Primary frontier-model judging.** For each episode, a
contact sheet was rendered from the recorded video (front + top
cameras; frames at 0/35/60/80/92/99% of the episode span, from the
episode's own video files/timestamps in the episode metadata) —
the same sheet method as the training-demo audit, for
comparability. Five judge contexts (one per task), blind to the
recorded fields and to each other, judged the END STATE of every
episode under the study's frozen per-task completion definitions
(`study001-success-criteria.md`, frozen 2026-11-20): PASS = the
final frames clearly show the completion end-state with the
gripper released; FAIL = the final frames clearly show the
end-state is not the completion state (a mid-episode success
undone by the end is FAIL); INDETERMINATE = the end state cannot
be determined from the final frames. The judging brief is
preserved at `heldout-audit-20261123/JUDGING_BRIEF.md`; the sheets
are the per-episode evidence pointers
(`heldout-audit-20261123/frames/`).

**(c) Blind independent verification (separate contexts).** A
seeded random 20% sample (seed `20261123`; single
`random.Random(20261123)` stream, tasks in the order block, eye,
ring, tool, cable; 4 episodes per task, 20 total: block 498, 545,
548, 549; eye 1630, 1780, 1811, 1821; ring 7, 30, 48, 179; tool
683, 824, 876, 903; cable 1903, 2032, 2042, 2085) was re-judged
blind by two verifier contexts with no access to the primary
verdicts or the recorded fields. (The draw contains no primary
FAIL or INDETERMINATE episode from eye_drops_to_basket — the
sample is a random draw and is reported as drawn.)

**(d) Grok Build CLI leg (external vendor, blind).** The Grok
Build CLI was located on the DGX at
`/home/openclaw/.local/bin/grok` (grok 1.0.88 (2fe3608f7398)
[stable]) — the earlier same-day report that it was not installed
was a too-narrow search; this audit's wide-net search (fleet
directories, all readable homes, /usr/local/bin, /opt, snap,
linuxbrew, npm globals, name variants grok/grok-build) found it
in the openclaw user's ~/.local/bin. It was run headless
(`--prompt-file`, `--always-approve`, `--disable-web-search`,
`--no-subagents`) over the same contact sheets and the same
frozen criteria, blind to all other verdicts, in 20 batches of 5
episodes (a first single 20-episode-per-task attempt was
abandoned after ~7 minutes without verdict output — it spent its
turns on crop/pixel tooling; the batched runs were instructed to
judge from the full sheets directly, and every batch returned
exactly 5 parseable verdict lines, rc=0). A one-episode smoke
test (cable ep2020) preceded the batches and returned the correct
FAIL in the required format. All 100 episodes were judged.

**(e) End-state evidence.** The final-frame pair (front + top at
99%) in each episode's contact sheet is the end-state evidence
for its verdicts; each leg's one-clause evidence statement per
episode is in `heldout-audit-20261123/verdicts_master.csv`.

## 3. Results — judged completion of the 100 held-out episodes (primary leg)

| Task | Held-out | PASS | FAIL | INDETERMINATE |
|---|---|---|---|---|
| block_stack | 20 | **20** | 0 | 0 |
| eye_drops_to_basket | 20 | **17** | 2 | 1 |
| ring_insert | 20 | **20** | 0 | 0 |
| tool_insert | 20 | **20** | 0 | 0 |
| cable_clip | 20 | **0** | 20 | 0 |
| **Total** | **100** | **77** | **22** | **1** |

Non-PASS episodes, primary leg:

- **eye_drops_to_basket ep1618 — FAIL**: tray tipped onto its side
  and visibly empty in the final top frames; platform empty, box
  nowhere in the basket. (Recorded label: success=1.)
- **eye_drops_to_basket ep1643 — FAIL**: tray tipped, then clearly
  empty and fully visible in the final top frames; platform empty,
  box gone. (Recorded label: success=1.)
- **eye_drops_to_basket ep1760 — INDETERMINATE**: the white box is
  never clearly seen in the tray; the tray tipped mid-episode and
  the final tray area is obscured by the arm in the final top
  frames.
- **cable_clip, all 20 episodes — FAIL**: in no episode is the
  cable seated/retained in the holder on the white block in the
  final frames; per-episode evidence in §7 / the master CSV. This
  matches the recorded labels (20/20 success=0) exactly.

The block_stack judge flagged four of its PASSes as borderline in
its own notes: ep453 and ep514 (offset stacks with partial
overhang, verified resting and stable between the 92% and 99%
frames) and ep489 and ep499 (top cube seated, gripper still in
contact with it in the 99% frame). Three of those four are
contested by the Grok leg (§4).

## 4. Verification legs and agreement

**Blind verifiers (leg c): agreement with the primary verdicts on
the seeded 20-episode sample: verifier 1 — 19/20 (95%);
verifier 2 — 20/20 (100%).** One disagreement: **cable_clip
ep2042** — primary FAIL vs verifier 1 INDETERMINATE ("final top
frame shows the cable running up to the side of the white box, but
whether the connector is actually seated/clipped in cannot be
clearly seen in either view"). Verifier 2 read the same sheet as
FAIL, agreeing with the primary. ep2042 is therefore contested on
two legs (this one and the Grok leg, §4 table) and is in the
founder's review pack. (Correction, filed the same day: the first
assembly of this report transcribed verifier 1's ep2042 verdict as
FAIL and stated 20/20 for both verifiers; verifier 1's own final
record says INDETERMINATE. The verdict files and master CSV have
been corrected to verifier 1's record.)

**Grok Build CLI (leg d): full-set verdicts — PASS 68, FAIL 30,
INDETERMINATE 2. Agreement with the primary verdicts: 80/100
(80%).** Per-task Grok PASS counts: block_stack 14/20,
eye_drops_to_basket 12/20, ring_insert 19/20, tool_insert 18/20,
cable_clip 5/20. The 20 disagreements are named here in full;
the audit does NOT resolve them — under the founder's standard
they go to the human-eyes leg (§6) with both reads attached:

| Episode | Task | Primary | Grok | Primary evidence | Grok evidence |
|---|---|---|---|---|---|
| 11 | ring_insert | PASS | FAIL | ring seated at the base of the central peg, gripper released, in final front and top frames | yellow ring rests on the peg tip, shaft not through it, gripper released |
| 489 | block_stack | PASS | FAIL | orange cube seated on purple; gripper finger still touching the top cube in the 99% frame (judge-flagged borderline) | gripper still closed on the orange block above the purple block |
| 499 | block_stack | PASS | FAIL | purple cube seated on orange at 99%, gripper just releasing (judge-flagged borderline) | gripper still closed on the purple block atop the orange block |
| 514 | block_stack | PASS | FAIL | green cube on purple, noticeably offset but resting on top, stable 92% to 99% (judge-flagged borderline) | green and purple blocks sit apart on the table |
| 531 | block_stack | PASS | FAIL | green cube on purple, clean stack, gripper released | green and purple blocks sit apart on the mat, gripper open |
| 539 | block_stack | PASS | FAIL | green cube on purple, clean stack, gripper released | green and purple blocks sit side by side, not stacked |
| 557 | block_stack | PASS | FAIL | orange cube on purple; released by 92%, stack stable at 99%, arm withdrawn | gripper still closed on the orange block at the end |
| 629 | tool_insert | PASS | FAIL | tool seated in tray slot, gripper empty at 99%, table clear | loose tool still lies outside the toolbox and the empty slot is unfilled |
| 784 | tool_insert | PASS | INDETERMINATE | tool seated in tray slot, gripper empty and withdrawn by 92%, table clear at 99% | gripper still covers the toolbox so seating and release are unclear |
| 1630 | eye_drops_to_basket | PASS | INDETERMINATE | blue box visible inside clear tray in final top frames | gripper covers the basket in the last top frame so release is unclear |
| 1639 | eye_drops_to_basket | PASS | FAIL | blue box visible inside clear tray at top 99% | bottle lies on the table beside the basket, not inside it |
| 1757 | eye_drops_to_basket | PASS | FAIL | blue box visible inside clear tray at top 99% | basket is empty and the bottle lies on the mat beside it |
| 1760 | eye_drops_to_basket | INDETERMINATE | FAIL | white box never clearly seen in the tray; tray tipped mid-episode and the final tray area is obscured by the arm in the final top frames | basket and gripper are empty, with the bottle nowhere inside |
| 1808 | eye_drops_to_basket | PASS | FAIL | silver/white box clearly inside clear tray at top 99% | basket, stand, and gripper are all empty at the end |
| 1811 | eye_drops_to_basket | PASS | FAIL | blue box clearly inside clear tray in final top frames | bottle lying on the mat beside the basket, gripper clear |
| 2020 | cable_clip | FAIL | PASS | cable stretches across the mat to a loose connector on the right; the holder on the white box is untouched by the cable | cable remains seated in the holder with the gripper open above it |
| 2042 | cable_clip | FAIL | PASS | cable connector lies loose on the mat; gripper points down left of the white box | cable stays seated in the white-block clip after the gripper pulls away |
| 2046 | cable_clip | FAIL | PASS | final cable remains coiled on the mat, not inserted into the holder | cable remains held in the clip with the gripper clear of it |
| 2092 | cable_clip | FAIL | PASS | cable stays coiled loose on the right side of the mat | cable stays raised in the clip on the white block after the gripper opens |
| 2101 | cable_clip | FAIL | PASS | final front frame shows the white box with no cable plugged into it; cable lies loose across the mat | cable remains seated in the holder with the gripper raised and open |

Patterns in the disagreements (stated factually): (i) Grok reads
five cable_clip episodes as PASS where the primary judge reads
FAIL — the only place any leg disputes the cable set; (ii) Grok
applies a stricter gripper-release reading on block_stack
(ep489, ep499 — both flagged borderline by the primary judge
itself — and ep557) and reads ep514/ep531/ep539 as not stacked;
(iii) Grok reads four eye_drops_to_basket episodes as FAIL where
the primary judge saw the bottle inside the basket (ep1639,
ep1757, ep1808, ep1811) and the primary INDETERMINATE (ep1760)
as FAIL.

**Coordinator reads** (audit coordinator's direct read of the
same sheets, recorded as reads — they change no verdict):
ep2020 — cable lies on the mat in front of the empty holder,
gripper lifted away: concurs FAIL. ep2046 — cable coiled on the
mat, holder on the white block visibly empty in the final front
and top frames: concurs FAIL. ep2101 — white block's holder
empty, cable lying on the mat to the left of the block: concurs
FAIL. ep531 — final front frames show the green cube resting on
the purple cube with the gripper withdrawn: concurs PASS. The
remaining 16 disagreement sheets were not coordinator-read; all
20 are in the founder's review pack (§6).

**Recorded labels vs judged verdicts:** the primary leg agrees
with the dataset's recorded success labels on 97/100 episodes
(the exceptions are eye ep1618 and ep1643, recorded success=1
but judged FAIL, and ep1760, judged INDETERMINATE). The Grok leg
agrees with the recorded labels on 78/100.

## 5. Gemini leg — NOT EXECUTED (reported, not skipped)

The founder's standard names the Gemini Antigravity CLI (`agy`).
Wide-net search on the DGX, 2026-11-23: no file named `agy`
exists in /home, /opt, or /snap (find, maxdepth 6); no
antigravity CLI binary in any readable home's ~/.local/bin,
/usr/local/bin, snap/bin, linuxbrew bin, or npm globals (the
only antigravity-named paths found are OpenClaw source files for
a google-antigravity auth extension, not a CLI). Two homes
(/home/username, /home/gh-runner) are not readable by the audit
account and /home/zk0bot is not accessible; the search evidence
covers every location the audit can read. A different
Gemini-family CLI IS present — `gemini` (Google Gemini CLI
0.26.0) in /home/linuxbrew/.linuxbrew/bin — but it is not
configured: it exits with "Please set an Auth method in your
/home/openclaw/.gemini/settings.json or specify one of the
following environment variables before running: GEMINI_API_KEY,
GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_GENAI_USE_GCA". No credentials
were improvised and no sign-in was attempted. **The Gemini leg
is therefore OWED**: it has not run on any of the 100 episodes,
and this audit's multi-frontier-model proof currently rests on
the Muse legs (b, c) and the Grok leg (d). The set it must
cover when a Gemini CLI is available is the full verdict list in
§7, with the 20 contested episodes in §4 first.

## 6. Human-eyes leg — review pack prepared; approval pending

Pack: `~/workspace/heldout-certification-review-20261123/` —
`fail/` (all 22 primary-FAIL sheets), `indeterminate/` (ep1760),
`disagreements/` (all 20 Grok-contested sheets), `pass_sample/`
(a seeded random sample of primary PASSes — seed `20261124`, 3
per task with PASSes: block 411, 453, 498; eye 1741, 1757, 1811;
ring 48, 188, 195; tool 630, 784, 902; cable has no PASSes to
sample), and `README.md` (per-task counts and how to review).
Certification is final only after the founder's approval of this
pack; until then every episode's status is PENDING HUMAN
APPROVAL (carried per-episode in the master CSV).

## 7. Full per-episode verdict list

Columns: episode · task · recorded success (at the source) ·
primary verdict · blind verifiers v1/v2 (sample only) · Grok
verdict. Per-episode evidence clauses for every leg are in
`heldout-audit-20261123/verdicts_master.csv`. Every episode's
certification status is PENDING HUMAN APPROVAL.

| Episode | Task | Recorded | Primary | V1 | V2 | Grok |
|---|---|---|---|---|---|---|
| 411 | block_stack | 1 | PASS | — | — | PASS |
| 453 | block_stack | 1 | PASS | — | — | PASS |
| 476 | block_stack | 1 | PASS | — | — | PASS |
| 479 | block_stack | 1 | PASS | — | — | PASS |
| 489 | block_stack | 1 | PASS | — | — | FAIL |
| 498 | block_stack | 1 | PASS | PASS | PASS | PASS |
| 499 | block_stack | 1 | PASS | — | — | FAIL |
| 513 | block_stack | 1 | PASS | — | — | PASS |
| 514 | block_stack | 1 | PASS | — | — | FAIL |
| 525 | block_stack | 1 | PASS | — | — | PASS |
| 526 | block_stack | 1 | PASS | — | — | PASS |
| 527 | block_stack | 1 | PASS | — | — | PASS |
| 530 | block_stack | 1 | PASS | — | — | PASS |
| 531 | block_stack | 1 | PASS | — | — | FAIL |
| 539 | block_stack | 1 | PASS | — | — | FAIL |
| 545 | block_stack | 1 | PASS | PASS | PASS | PASS |
| 546 | block_stack | 1 | PASS | — | — | PASS |
| 548 | block_stack | 1 | PASS | PASS | PASS | PASS |
| 549 | block_stack | 1 | PASS | PASS | PASS | PASS |
| 557 | block_stack | 1 | PASS | — | — | FAIL |
| 1609 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1618 | eye_drops_to_basket | 1 | FAIL | — | — | FAIL |
| 1630 | eye_drops_to_basket | 1 | PASS | PASS | PASS | INDETERMINATE |
| 1639 | eye_drops_to_basket | 1 | PASS | — | — | FAIL |
| 1643 | eye_drops_to_basket | 1 | FAIL | — | — | FAIL |
| 1724 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1741 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1746 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1747 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1757 | eye_drops_to_basket | 1 | PASS | — | — | FAIL |
| 1760 | eye_drops_to_basket | 1 | INDETERMINATE | — | — | FAIL |
| 1778 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1780 | eye_drops_to_basket | 1 | PASS | PASS | PASS | PASS |
| 1783 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1798 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1808 | eye_drops_to_basket | 1 | PASS | — | — | FAIL |
| 1810 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1811 | eye_drops_to_basket | 1 | PASS | PASS | PASS | FAIL |
| 1814 | eye_drops_to_basket | 1 | PASS | — | — | PASS |
| 1821 | eye_drops_to_basket | 1 | PASS | PASS | PASS | PASS |
| 7 | ring_insert | 1 | PASS | PASS | PASS | PASS |
| 11 | ring_insert | 1 | PASS | — | — | FAIL |
| 30 | ring_insert | 1 | PASS | PASS | PASS | PASS |
| 31 | ring_insert | 1 | PASS | — | — | PASS |
| 33 | ring_insert | 1 | PASS | — | — | PASS |
| 48 | ring_insert | 1 | PASS | PASS | PASS | PASS |
| 50 | ring_insert | 1 | PASS | — | — | PASS |
| 160 | ring_insert | 1 | PASS | — | — | PASS |
| 165 | ring_insert | 1 | PASS | — | — | PASS |
| 167 | ring_insert | 1 | PASS | — | — | PASS |
| 168 | ring_insert | 1 | PASS | — | — | PASS |
| 177 | ring_insert | 1 | PASS | — | — | PASS |
| 179 | ring_insert | 1 | PASS | PASS | PASS | PASS |
| 188 | ring_insert | 1 | PASS | — | — | PASS |
| 195 | ring_insert | 1 | PASS | — | — | PASS |
| 198 | ring_insert | 1 | PASS | — | — | PASS |
| 227 | ring_insert | 1 | PASS | — | — | PASS |
| 240 | ring_insert | 1 | PASS | — | — | PASS |
| 242 | ring_insert | 1 | PASS | — | — | PASS |
| 253 | ring_insert | 1 | PASS | — | — | PASS |
| 622 | tool_insert | 1 | PASS | — | — | PASS |
| 629 | tool_insert | 1 | PASS | — | — | FAIL |
| 630 | tool_insert | 1 | PASS | — | — | PASS |
| 683 | tool_insert | 1 | PASS | PASS | PASS | PASS |
| 684 | tool_insert | 1 | PASS | — | — | PASS |
| 697 | tool_insert | 1 | PASS | — | — | PASS |
| 779 | tool_insert | 1 | PASS | — | — | PASS |
| 784 | tool_insert | 1 | PASS | — | — | INDETERMINATE |
| 824 | tool_insert | 1 | PASS | PASS | PASS | PASS |
| 830 | tool_insert | 1 | PASS | — | — | PASS |
| 852 | tool_insert | 1 | PASS | — | — | PASS |
| 859 | tool_insert | 1 | PASS | — | — | PASS |
| 871 | tool_insert | 1 | PASS | — | — | PASS |
| 876 | tool_insert | 1 | PASS | PASS | PASS | PASS |
| 883 | tool_insert | 1 | PASS | — | — | PASS |
| 885 | tool_insert | 1 | PASS | — | — | PASS |
| 888 | tool_insert | 1 | PASS | — | — | PASS |
| 892 | tool_insert | 1 | PASS | — | — | PASS |
| 902 | tool_insert | 1 | PASS | — | — | PASS |
| 903 | tool_insert | 1 | PASS | PASS | PASS | PASS |
| 1896 | cable_clip | 0 | FAIL | — | — | FAIL |
| 1903 | cable_clip | 0 | FAIL | FAIL | FAIL | FAIL |
| 1905 | cable_clip | 0 | FAIL | — | — | FAIL |
| 1939 | cable_clip | 0 | FAIL | — | — | FAIL |
| 1959 | cable_clip | 0 | FAIL | — | — | FAIL |
| 1960 | cable_clip | 0 | FAIL | — | — | FAIL |
| 1968 | cable_clip | 0 | FAIL | — | — | FAIL |
| 1969 | cable_clip | 0 | FAIL | — | — | FAIL |
| 2020 | cable_clip | 0 | FAIL | — | — | PASS |
| 2032 | cable_clip | 0 | FAIL | FAIL | FAIL | FAIL |
| 2042 | cable_clip | 0 | FAIL | INDETERMINATE | FAIL | PASS |
| 2046 | cable_clip | 0 | FAIL | — | — | PASS |
| 2059 | cable_clip | 0 | FAIL | — | — | FAIL |
| 2068 | cable_clip | 0 | FAIL | — | — | FAIL |
| 2085 | cable_clip | 0 | FAIL | FAIL | FAIL | FAIL |
| 2092 | cable_clip | 0 | FAIL | — | — | PASS |
| 2101 | cable_clip | 0 | FAIL | — | — | PASS |
| 2125 | cable_clip | 0 | FAIL | — | — | FAIL |
| 2130 | cable_clip | 0 | FAIL | — | — | FAIL |
| 2146 | cable_clip | 0 | FAIL | — | — | FAIL |

## 8. Plain statement — what this audit establishes

Under the legs completed (primary judging of all 100, blind
verification at 95% (verifier 1) and 100% (verifier 2) agreement
on the seeded 20% sample — one named disagreement, ep2042 — Grok
full-set pass at 80% agreement with 20 named contested
episodes), and subject to the two outstanding legs (Gemini,
human eyes):

- block_stack: **20 of 20** held-out episodes judged task
  solutions by the primary leg (4 flagged borderline by the
  judge; 6 contested by Grok).
- eye_drops_to_basket: **17 of 20** (ep1618, ep1643 judged FAIL
  against their recorded success labels; ep1760 INDETERMINATE;
  6 episodes contested by Grok).
- ring_insert: **20 of 20** (1 contested by Grok: ep11).
- tool_insert: **20 of 20** (2 contested by Grok: ep629 FAIL,
  ep784 INDETERMINATE).
- cable_clip: **0 of 20** — every held-out cable reference is a
  failed execution, confirmed independently by the primary leg
  and by the recorded labels; Grok dissents on 5 (ep2020,
  ep2042, ep2046, ep2092, ep2101), which are contested, not
  certified.
- **Total: 77 of 100** held-out episodes judged task solutions
  by the primary leg; 22 judged FAIL; 1 INDETERMINATE; 20
  episodes carry a cross-vendor disagreement that only the
  human-eyes leg can close.

Before this audit, the completion axis of the held-out set had
been taken from the dataset's recorded labels alone (with the
cable failure known from the v1.1 twin record). This audit is
the first per-episode independent judging of the set.

## 9. Caveats

- Judging rests on 6 sampled timepoints × 2 cameras (front, top)
  at contact-sheet resolution; the wrist camera was not used —
  the same limits as the training-demo audit. Verdicts judge the
  end state only, per the frozen criteria.
- The frozen per-task criteria are the study's own draft
  operationalizations of the release instructions (the release
  publishes no per-task rubric); borderline calls inherit their
  stated ambiguities (e.g., cable seating depth, gripper-contact
  at the final frame on block_stack).
- Judge and verifier contexts are frontier models of the same
  family as the audit coordinator; the Grok leg adds the
  cross-vendor check, and its 80% agreement — with disagreements
  concentrated in genuinely borderline geometry (offset stacks,
  final-frame gripper contact, basket occlusion, cable seating)
  — is reported as measured, not reconciled away.
- The Gemini leg and the human-eyes leg are outstanding (§5,
  §6); nothing in this report is a final certification.

## 10. Artifacts

- This report: `heldout-episode-certification-audit-20261123.md`
  (study notes; committed to `study001/fl-substrate-20261121` at
  `twin-replay/heldout-episode-certification-audit-20261123.md`).
- Audit folder `heldout-audit-20261123/`: `JUDGING_BRIEF.md`,
  `recorded_fields.csv` (per-episode recorded fields read at the
  source), `verdicts_master.csv` (per-episode verdicts + evidence
  for every leg + certification status), `verdicts_j_block.txt` /
  `verdicts_j_eye.txt` / `verdicts_j_ring.txt` /
  `verdicts_j_tool.txt` / `verdicts_j_cable.txt` (primary judges),
  `verdicts_v1.txt` / `verdicts_v2.txt` (blind verifiers),
  `verdicts_grok.txt` (Grok Build CLI, 100 lines),
  `verifier_sample.json`, `frames/` (100 contact sheets, JPEG
  derivatives of the PNG masters rendered on the DGX from the
  pinned dataset; masters + generation script + logs in the DGX
  study workdir `heldout-audit-20261123/`).
- Founder review pack: `~/workspace/heldout-certification-review-20261123/`
  (see §6).

## Gemini leg addendum (2026-11-23, tier-certification coordinator #3)

Method: Google Antigravity CLI (agy v1.5.15) on the DGX, one-shot print
mode, model PINNED per invocation to **gemini-3.8-flash-high** (the
strongest Gemini selectable on the account; the leg is an independent
non-Claude read — the Claude/GPT model group was never used). Model
verified from the agy session records (print-mode log lines show
model="gemini-3.8-flash-high" as the only model string). Blind: same
contact sheets, same frozen criteria brief, no prior verdicts in the
prompts. JPG sheets are rejected by agy's media path (HTTP 400), so all
judging used PNG conversions of the same sheets. Per-episode Gemini
verdicts + evidence + model are in verdicts_master.csv
(gemini_verdict / gemini_evidence / gemini_model).

Stage 1 — the 20 Grok-contested episodes (tie-break reads): cable
ep2020 PASS / ep2042 FAIL / ep2046 FAIL / ep2092 FAIL / ep2101 FAIL;
ring ep11 PASS; block ep489 FAIL / ep499 FAIL / ep514 PASS / ep531
PASS / ep539 PASS / ep557 FAIL; tool ep629 PASS / ep784 FAIL; eye
ep1630 PASS / ep1639 PASS / ep1757 PASS / ep1760 FAIL / ep1808 PASS /
ep1811 PASS. All 20 remain contested for the founder — Gemini is a
third read, not a resolver.

ep2020 carries THREE Gemini reads, recorded in full: a single-image
micro-test read the sheet as cable-not-seated (FAIL-equivalent prose);
the contested batch read PASS ("cable is seated in the holder and
gripper released"); the full-set batch read FAIL. Gemini is not
self-consistent on this borderline episode; the founder's adjudication
is the deciding read.

Stage 2 — the full held-out 100 (5 runs x 20, all rc=0): Gemini
73 PASS / 27 FAIL. Agreement: Gemini vs primary 93/100; Gemini vs
Grok 82/100. Gemini disagrees with primary on 7 episodes: block
489/499/557 (FAIL, with Grok), tool 784 (FAIL; Grok INDETERMINATE),
tool 824 (FAIL vs primary+Grok PASS — NEW contested), eye 1643 (PASS
vs primary+Grok FAIL — NEW contested), eye 1760 (FAIL; primary
INDETERMINATE, Grok FAIL). Held-out contested after all three model
families: 22 episodes (the original 20 + ep824 + ep1643), all for the
founder with every read attached.

Quota stop: the Gemini weekly group (starter tier, SHARED with Cos's
machine) stood at 32% remaining after stage 2 (reset
2026-11-30T21:41:45Z). Per the program's quota discipline the leg
stops here: stage 3 (1h training set, 250), stage 4 (Spa-Bench +
RoboColosseum registers) and stage 5 (new sources) resume after the
reset. Note for a #bot-hub post: the week's Gemini usage was consumed
by this leg's contested + held-out-100 stages.
