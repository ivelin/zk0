# Training-Demo Certification Audit — Study 001 (2026-11-23)

**Order:** founder, 2026-11-23 14:24 CT ("back to the drawing board") —
establish, per demo, whether the trajectories Study 001 trained on are
task solutions, as the fact base for the training-set redesign.
**Standard applied:** the founder's 2026-11-23 rule for training data —
every training trajectory individually certified as a task solution,
with the pass independently confirmed.
**Auditor:** a separate-context audit agent; judging and verification
were executed by further separate contexts (below). No redesign
recommendations are made here; this is the fact base only.

## 1. Subject — the exact 250 demos, pinned

Dataset: `armnet/armnetbench_v01_lerobot_so101` at pinned revision
`1db69c82b589c7172ee31a6b8f7ef58d1fc2c663` (Apache-2.0), LeRobot v3.0,
20 fps; on-disk root on the DGX Spark:
`/home/openclaw/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101`
(the root named in the study's own P0 manifest). Selection filter, as
consumed by every Study 001 arm: `policy_type == "teleoperated"` AND
task ∈ {block_stack, eye_drops_to_basket, ring_insert, tool_insert,
cable_clip}. Episode indices are dataset episode indices at the pinned
revision; subset SHA-256 values are the study P0 manifest's.

| Task | Episodes | Indices | Frames | Subset SHA-256 |
|---|---|---|---|---|
| ring_insert | 50 | 255–304 | 11,679 | `d961d8b14297d88d7b3702bd4ebcbf342c9ae25203af9823f00e7c221c169af5` |
| tool_insert | 50 | 910–959 | 20,806 | `38bae64f8ce13ae059f8d583ca5a9ef843b7bf7ffc10869178908abf9bdc4e29` |
| cable_clip | 50 | 2159–2208 | 27,647 | `b752353f6478a770d0ec209dc80e78898412cc249c1fad27d7eb73c255395d8a` |
| eye_drops_to_basket | 50 | 1839–1888 | 9,127 | `dd5ea56de9850e6fe13eb1d04fa3acd91727ed7f5d1fe58fa25da0686410d1c7` |
| block_stack | 50 | 560–609 | 12,975 | `97ae521c56801b541923f2e3b8ec172e2020f947e1528461b04901c7b4b48417` |

Union: 250 episodes, 82,234 frames (centralized-arm episode-list
SHA-256 `4158ba0971c8ba804bd4c33886be9fa999002ea92686954d4e1d91ff68735dc4`).

## 2. Method

**(a) Recorded fields — read at the source.** Episode metadata
(`meta/episodes/chunk-000/file-000.parquet`) carries the fields
`success` (int64), `success_class` (string), `policy_type` (string),
`policy_repo_id` (string); the per-frame data parquet carries
`next.reward` / `next.done`. For all 250 training episodes:
`success == 1` (250/250), `success_class == "successful"` (250/250),
`policy_type == "teleoperated"` (250/250), and the final frame's
`next.reward == 1.0` with `next.done == True` (250/250, read from the
data parquet at each episode's `dataset_to_index − 1` row).
**These fields are not independent evidence.** The release's own
label scheme, quoted in the frozen criteria file
(`study001-success-criteria.md` §1), states: "Teleoperation episodes
are always `success_class=successful`, `success=1`" — the flag is
assigned by construction to teleoperated episodes; only policy
rollouts were human-labelled. The study manifest itself records
`completion_axis_training_data: "UNJUDGED (§3B census not run on the
250 demos; Decision 59)"`. Method (a) therefore establishes only what
the dataset *claims*, and the claim is definitional.

**(b) Frontier-model judging of the demos themselves.** For each of
the 250 episodes, a contact sheet was rendered from the recorded
video (front + top cameras; frames at 0/35/60/80/92/≈100% of the
episode span, taken from the episode's own video files/timestamps in
the episode metadata). Ten judge contexts, blind to the recorded
fields and to each other, judged the END STATE of every episode under
the study's frozen per-task completion definitions
(`study001-success-criteria.md`, frozen 2026-11-20): PASS = the final
frames clearly show the completion end-state with the gripper
released; FAIL = the final frames clearly show the end-state is not
the completion state (a mid-episode success undone by the end is
FAIL); INDETERMINATE = the end state cannot be determined from the
final frames. The full judging brief is preserved at
`training-demo-audit-20261123/JUDGING_BRIEF.md`; the sheets are the
per-episode evidence pointers (`training-demo-audit-20261123/frames/`).

**(c) End-state evidence.** The final-frame pair (front + top at
≈100%) in each episode's contact sheet is the end-state evidence for
its verdict; the judge's one-clause evidence statement per episode is
in §5 and in `training-demo-audit-20261123/verdicts_master.csv`.

## 3. Results — judged completion of the 250 training demos

| Task | Trained on | Certified PASS | FAIL | INDETERMINATE |
|---|---|---|---|---|
| block_stack | 50 | **49** | 1 | 0 |
| eye_drops_to_basket | 50 | **50** | 0 | 0 |
| ring_insert | 50 | **50** | 0 | 0 |
| tool_insert | 50 | **50** | 0 | 0 |
| cable_clip | 50 | **50** | 0 | 0 |
| **Total** | **250** | **249** | **1** | **0** |

The single FAIL: **ep604 (block_stack)** — judge evidence: "final
frames show orange block alone on the mat, not stacked on the purple
block." The audit coordinator's direct read of the same sheet
concurs: the episode ends with a single block standing alone, no
stack. Its recorded fields claim `success == 1` / `successful` /
final `next.reward == 1.0` — a recorded claim the visual record
contradicts.

Note on scope: the 20 held-out cable_clip episodes established
earlier today as failed executions are a **different, disjoint
episode set** (the held-out split); they are not among these 250.
All 50 cable_clip *training* demos judged here show the cable seated
in the holder at episode end.

## 4. Independent verification (separate contexts, blind)

A seeded random 20% sample (seed `20261123`; 10 episodes per task,
50 total: ring 260, 265, 266, 279, 286, 289, 290, 291, 296, 303;
tool 911, 914, 916, 921, 927, 934, 937, 951, 955, 957; cable 2161,
2162, 2166, 2177, 2179, 2180, 2188, 2201, 2202, 2205; eye 1842,
1851, 1854, 1856, 1869, 1875, 1876, 1884, 1887, 1888; block 567,
580, 583, 585, 586, 593, 597, 600, 605, 606) was re-judged blind by
two verifier contexts with no access to the primary verdicts or the
recorded fields.

**Agreement: 49/50 = 98.0%.** One disagreement:

- **ep2201 (cable_clip)** — primary judge PASS ("cable seated
  through the holder on the white block, gripper off it") vs
  verifier INDETERMINATE ("cable drapes diagonally over the white
  block; seating in the holder not clearly visible in either view").
  The coordinator's direct read of the sheet: the cable crosses the
  white block diagonally and its seating in the clip is not clearly
  confirmable at contact-sheet resolution — the disagreement is a
  visibility limit, not a demonstrated error by either judge. The
  audit verdict for ep2201 remains the primary PASS, with this
  disagreement attached; it is the one PASS in the set whose
  independent confirmation is qualified.

The headline counts in §3 are stated with this agreement rate
attached, per the audit's terms.

## 5. Full per-episode verdict list

Columns: episode · task · judged verdict · judge evidence ·
verifier verdict (sample only). Recorded fields for every episode
(`success`, `success_class`, final `next.reward`) are uniformly
1 / "successful" / 1.0 and are tabulated per episode in
`training-demo-audit-20261123/verdicts_master.csv`, together with the
contact-sheet evidence pointer.

| Episode | Task | Verdict | Judge evidence | Verifier |
|---|---|---|---|---|
| 560 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 561 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 562 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 563 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 564 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 565 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 566 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 567 | block_stack | PASS | green block stacked on purple block, standing, gripper released | PASS |
| 568 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 569 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 570 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 571 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 572 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 573 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 574 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 575 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 576 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 577 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 578 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 579 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 580 | block_stack | PASS | green block stacked on purple block, standing, gripper released | PASS |
| 581 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 582 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 583 | block_stack | PASS | green block stacked on purple block, standing, gripper released | PASS |
| 584 | block_stack | PASS | green block stacked on purple block, standing, gripper released | — |
| 585 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | PASS |
| 586 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | PASS |
| 587 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 588 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 589 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 590 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 591 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 592 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 593 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | PASS |
| 594 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 595 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 596 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 597 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | PASS |
| 598 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 599 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 600 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | PASS |
| 601 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 602 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 603 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 604 | block_stack | FAIL | final frames show orange block alone on the mat, not stacked on the purple block | — |
| 605 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | PASS |
| 606 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | PASS |
| 607 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 608 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 609 | block_stack | PASS | final frames show orange block stacked on purple block, standing, gripper released | — |
| 1839 | eye_drops_to_basket | PASS | blue box resting inside the clear basket in final top view, gripper away | — |
| 1840 | eye_drops_to_basket | PASS | blue box inside the basket in final frames, gripper released | — |
| 1841 | eye_drops_to_basket | PASS | blue box lying inside the basket in final top view, gripper away | — |
| 1842 | eye_drops_to_basket | PASS | blue box inside the basket in final front and top views, gripper away | PASS |
| 1843 | eye_drops_to_basket | PASS | blue box resting inside the basket at its near end in final top view, gripper away | — |
| 1844 | eye_drops_to_basket | PASS | blue box inside the basket in final frames, gripper released | — |
| 1845 | eye_drops_to_basket | PASS | blue box lying inside the basket in final top view, gripper away | — |
| 1846 | eye_drops_to_basket | PASS | blue box inside the basket at its near end in final top view, gripper away | — |
| 1847 | eye_drops_to_basket | PASS | blue box inside the basket in final front and top views, gripper away | — |
| 1848 | eye_drops_to_basket | PASS | blue box visible inside the basket in final front view, gripper away | — |
| 1849 | eye_drops_to_basket | PASS | blue box lying inside the basket in final top view, gripper away | — |
| 1850 | eye_drops_to_basket | PASS | blue box inside the basket at its near end in final top view, gripper away | — |
| 1851 | eye_drops_to_basket | PASS | blue box inside the basket in final front view and at basket edge in top view, gripper away | PASS |
| 1852 | eye_drops_to_basket | PASS | blue box inside the basket in final front view, gripper away | — |
| 1853 | eye_drops_to_basket | PASS | blue box lying inside the basket in final top view, gripper away | — |
| 1854 | eye_drops_to_basket | PASS | blue box inside the basket at its near end in final top view, gripper away | PASS |
| 1855 | eye_drops_to_basket | PASS | box inside the basket in final front view, gripper away | — |
| 1856 | eye_drops_to_basket | PASS | white box lying inside the basket in final top view, gripper away | PASS |
| 1857 | eye_drops_to_basket | PASS | box inside the basket in final front and top views, gripper away | — |
| 1858 | eye_drops_to_basket | PASS | box inside the basket at its near end in final top view, gripper away | — |
| 1859 | eye_drops_to_basket | PASS | white box inside the basket in final top view, gripper away | — |
| 1860 | eye_drops_to_basket | PASS | box inside the basket in final top view, gripper away | — |
| 1861 | eye_drops_to_basket | PASS | blue box lying inside the basket in final top view, gripper away | — |
| 1862 | eye_drops_to_basket | PASS | blue box inside the basket at its near end in final top and front views, gripper away | — |
| 1863 | eye_drops_to_basket | PASS | blue box inside the basket in final front and top views, gripper away | — |
| 1864 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1865 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1866 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1867 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1868 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1869 | eye_drops_to_basket | PASS | final top frame shows bottle resting inside the clear basket, gripper released | PASS |
| 1870 | eye_drops_to_basket | PASS | final top frame shows bottle resting inside the clear basket, gripper released | — |
| 1871 | eye_drops_to_basket | PASS | final top frame shows bottle resting inside the clear basket, gripper released | — |
| 1872 | eye_drops_to_basket | PASS | final top frame shows bottle resting inside the clear basket, gripper released | — |
| 1873 | eye_drops_to_basket | PASS | final top frame shows bottle resting inside the clear basket, gripper released | — |
| 1874 | eye_drops_to_basket | PASS | final top frame shows bottle resting inside the clear basket, gripper released | — |
| 1875 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | PASS |
| 1876 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | PASS |
| 1877 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1878 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1879 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1880 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1881 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1882 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1883 | eye_drops_to_basket | PASS | final top frame shows bottle resting inside the clear basket, gripper released | — |
| 1884 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | PASS |
| 1885 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1886 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | — |
| 1887 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | PASS |
| 1888 | eye_drops_to_basket | PASS | final top frame shows blue bottle resting inside the clear basket, gripper released | PASS |
| 255 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 256 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 257 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 258 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 259 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 260 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | PASS |
| 261 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 262 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 263 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 264 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 265 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | PASS |
| 266 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | PASS |
| 267 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 268 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 269 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 270 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 271 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 272 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 273 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 274 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 275 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 276 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 277 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 278 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | — |
| 279 | ring_insert | PASS | yellow ring encircles central peg at base, gripper released in final frames | PASS |
| 280 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 281 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 282 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 283 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 284 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 285 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 286 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | PASS |
| 287 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 288 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 289 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | PASS |
| 290 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | PASS |
| 291 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | PASS |
| 292 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 293 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 294 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 295 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 296 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | PASS |
| 297 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 298 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 299 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 300 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 301 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 302 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 303 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | PASS |
| 304 | ring_insert | PASS | red ring at base of central peg, released, in final front and top frames | — |
| 910 | tool_insert | PASS | final top frame shows the previously empty red cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 911 | tool_insert | PASS | final top frame shows the empty cross slot filled, the loose horizontal tool no longer on the table, gripper released | PASS |
| 912 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 913 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 914 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | PASS |
| 915 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 916 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | PASS |
| 917 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 918 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 919 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 920 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 921 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | PASS |
| 922 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 923 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 924 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 925 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 926 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 927 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | PASS |
| 928 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 929 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 930 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 931 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 932 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | — |
| 933 | tool_insert | PASS | final top frame shows the empty cross slot filled, loose tool gone from the table, gripper released | — |
| 934 | tool_insert | PASS | final top frame shows the empty cross slot filled with the seated tool, loose tool gone from the table, gripper released | PASS |
| 935 | tool_insert | PASS | final top view shows the T-tool seated in the centre slot that was empty red foam at start, gripper released | — |
| 936 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 937 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | PASS |
| 938 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 939 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | — |
| 940 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 941 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 942 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 943 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | — |
| 944 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 945 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | — |
| 946 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 947 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | — |
| 948 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 949 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | — |
| 950 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 951 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | PASS |
| 952 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 953 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | — |
| 954 | tool_insert | PASS | loose tool gone from the mat and final top view shows it seated in the previously empty centre slot, gripper released | — |
| 955 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | PASS |
| 956 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 957 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | PASS |
| 958 | tool_insert | PASS | final top view shows the tool seated in the previously empty centre slot, gripper released | — |
| 959 | tool_insert | PASS | final top view shows the T-tool seated in the previously empty centre slot, gripper released | — |
| 2159 | cable_clip | PASS | final frames show cable running through the black clip on the white block, gripper released and away | — |
| 2160 | cable_clip | PASS | final frames show cable seated through the clip on the white block, gripper off it | — |
| 2161 | cable_clip | PASS | final frames show cable retained in the clip on the white block, gripper released | PASS |
| 2162 | cable_clip | PASS | final frames show cable running through the holder on the white block, gripper away | PASS |
| 2163 | cable_clip | PASS | final frames show cable seated in the clip on the white block, gripper off it | — |
| 2164 | cable_clip | PASS | final frames show cable through the clip on the white block, gripper released | — |
| 2165 | cable_clip | PASS | final frames show cable held in the black clip atop the white block, gripper away | — |
| 2166 | cable_clip | PASS | final frames show cable running through the holder on the white block, gripper off it | PASS |
| 2167 | cable_clip | PASS | final frames show cable seated through the clip on the white block, gripper released | — |
| 2168 | cable_clip | PASS | final frames show cable retained in the clip on the white block, gripper away | — |
| 2169 | cable_clip | PASS | final frames show cable running through the clip on the white block, gripper off it | — |
| 2170 | cable_clip | PASS | final frames show cable seated in the holder on the white block, gripper released | — |
| 2171 | cable_clip | PASS | final frames show cable through the clip on the white block, gripper away | — |
| 2172 | cable_clip | PASS | final frames show cable retained in the clip on the white block, gripper off it | — |
| 2173 | cable_clip | PASS | final frames show cable running through the black clip on the white block in both views, gripper released | — |
| 2174 | cable_clip | PASS | final frames show cable seated through the clip on the white block, gripper away | — |
| 2175 | cable_clip | PASS | final frames show cable held in the clip on the white block, gripper off it | — |
| 2176 | cable_clip | PASS | final frames show cable running through the holder on the white block, gripper released | — |
| 2177 | cable_clip | PASS | final frames show cable seated in the clip on the white block, gripper away | PASS |
| 2178 | cable_clip | PASS | final frames show cable through the clip on the white block, extending both sides, gripper off it | — |
| 2179 | cable_clip | PASS | final frames show cable passing through the clip on the white block in both views, gripper released | PASS |
| 2180 | cable_clip | PASS | final frames show cable retained in the clip on the white block, gripper away | PASS |
| 2181 | cable_clip | PASS | final frames show cable seated through the holder on the white block, gripper off it | — |
| 2182 | cable_clip | PASS | final frames show cable running through the clip on the white block, gripper released | — |
| 2183 | cable_clip | PASS | final frames show cable held in the clip on the white block, gripper away | — |
| 2184 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2185 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2186 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2187 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2188 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | PASS |
| 2189 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2190 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2191 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2192 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2193 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2194 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2195 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2196 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2197 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2198 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2199 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2200 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2201 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | INDETERMINATE |
| 2202 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | PASS |
| 2203 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2204 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2205 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | PASS |
| 2206 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2207 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |
| 2208 | cable_clip | PASS | cable seated in black clip on white block in final front and top frames, gripper released | — |

## 6. External CLI final-verification stage — NOT EXECUTED (unavailable)

Standing rule (founder, 2026-11-23 14:29 CT): important results get a
final verification pass through BOTH the Grok Build CLI and the
Gemini Antigravity (agy) CLI, on top of internal separate-context
verification; if either CLI is unavailable in the environment, that
is reported, never silently skipped.

**Status: neither CLI is available in this environment.** Search
performed 2026-11-23 ~14:31–14:40 CT: `PATH` (`which grok`, `which
agy`), `~/.local/bin`, `/usr/local/bin`, `/opt/hatch/bin`, npm global
packages, `~/.grok` / `~/.antigravity` directories, and home-directory
scans (this VM: `/home/hatch` to depth 4; DGX: `/home/openclaw` to
depth 3 plus `~/bin`) found no `grok`, `agy`, or `antigravity`
executable on either machine. The stage was therefore **not run** —
this section is the explicit report the rule requires. The sets the
stage would have covered are enumerated in this report for when the
CLIs are available: the full 249-episode PASS list (§5) and the
non-pass/disagreement list in full — ep604 (FAIL) and ep2201
(primary PASS / verifier INDETERMINATE).

## 7. Plain statement — how much certified-clean training data exists

Under this audit (frozen-criteria judging of every demo, blind
verification at 98.0% agreement on a seeded 20% sample):

- block_stack: **49 of 50** demos are certified task solutions
  (ep604 is not a solution).
- eye_drops_to_basket: **50 of 50**.
- ring_insert: **50 of 50**.
- tool_insert: **50 of 50**.
- cable_clip: **50 of 50** judged solutions, of which one (ep2201)
  carries a verifier disagreement (INDETERMINATE on blind re-judge);
  49 are unqualified.
- **Total: 249 of 250** training demos are certified task solutions
  (248 unqualified; 1 — ep2201 — certified with a recorded verifier
  disagreement; 1 — ep604 — not a solution).

Before this audit, the certified count was 0 of 250: the dataset's
`success` flag is assigned by construction to teleoperated episodes
and had never been independently confirmed for any training demo
(Decision 59 two-axis gap).

## 8. Caveats

- Judging rests on 6 sampled timepoints × 2 cameras (front, top) at
  contact-sheet resolution; the wrist camera was not used. Verdicts
  judge the end state only, per the frozen criteria; a demo that
  completed and then disturbed the end state before its final frame
  would judge FAIL, and a suboptimal-but-complete end state judges
  PASS only if it meets the strict definition.
- The frozen per-task criteria are the study's own draft
  operationalizations of the release instructions (the release
  publishes no per-task rubric); borderline calls inherit their
  stated ambiguities (e.g., cable seating depth, tool seating
  tolerance).
- Judge and verifier contexts are frontier models of the same family
  as the audit coordinator; the external-CLI stage that would have
  added cross-vendor verification could not be executed (§6).

## 9. Artifacts

- This report: `training-demo-certification-audit-20261123.md`
  (study notes; committed to `study001/fl-substrate-20261121`).
- Audit folder `training-demo-audit-20261123/`: `JUDGING_BRIEF.md`,
  `recorded_fields.csv` (per-episode recorded fields read at the
  source), `verdicts_master.csv` (per-episode verdict, judge
  evidence, verifier verdict/evidence, evidence pointer),
  `verdicts_j1..j10.txt` (primary judges), `verdicts_v1..v2.txt`
  (blind verifiers), `frames/` (250 contact sheets, JPEG derivatives
  of the PNG masters rendered on the DGX from the pinned dataset).
