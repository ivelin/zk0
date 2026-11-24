# 10-hour tier certification program — 2026-11-23

**Scope.** Completion-axis certification of the zk0 SO-101 tier-10h manifest
(`../tier-10h-manifest.yaml`, assembled 2026-11-19/20): 2,282 episodes,
10.80 h, five sources, contamination axis CONFIRMED for every entry,
completion axis never judged before this program.

**Scope extension (2026-11-23, founder clearing directive; re-measurement
§5 Wave 0).** The program covers the FULL contamination-CONFIRMED pool —
Wave 0 = **2,792 episodes / 16.11 h**: the tier's 2,282 plus DecisionFacts
reserve (301: pipe/joint sorting 141 @ e0546023 + test-tube nesting 160 @
8c161ed27), BusyBox singles (59: push_green_button 39 @ aceecaaf +
flip_left_switch_off 20 @ 3ca568fe), and the ArmnetBench held-out tasks
for eval use (150 teleoperated episodes: eye_drops_to_shelf eps
1549–1598, tool_removal eps 1200–1249, cable_unclip eps 2449–2498 — index
ranges read from the dataset's episodes parquet at the pinned revision).
Waves 1–3 are NOT authorized (founder decisions pending in the
re-measurement §6). Fresh judging total: 2,542 episodes (2,792 − 250
carried in). Governed by the founder's
certified-data gate (2026-11-23 15:34 CT): every episode individually
certified for actually passing its task — multi-frontier-model proof +
human-eyes approval; no exceptions, no compromises, no shortcuts.

**Status language.** Nothing in this program is called "certified". Every
episode's status is **PENDING HUMAN APPROVAL** until the founder approves
the review pack and the Gemini leg closes (Gemini leg is OWED: agy is
absent on all machines checked; the DGX `gemini` CLI exists but is
unauthenticated — no credentials improvised).

## Sources (manifest order)

| # | source | revision | eps | fps | cameras used for sheets |
|---|---|---|---|---|---|
| 1 | armnet/armnetbench_v01_lerobot_so101 | 1db69c82b589c7172ee31a6b8f7ef58d1fc2c663 | 250 | 20 | front + top |
| 2 | Spa-Bench/spa-bench-training-teleoperation-1200 | 375b9e6cc137266af5dbbcf83535c3b1adbf0207 | 1,200 | 30 | middle (front-equiv) + above (top-equiv) |
| 3a | RoboColosseum/so101-cube-drawer | 5d07c27a375fadc57132b76d2c13b3864a438e25 | 100 | 30 | front + wrist (only 2 cams) |
| 3b | RoboColosseum/so101-remove-screwdriver | 714ef036ee9afded3d23301443b55c78da846f07 | 100 | 30 | front + wrist |
| 3c | RoboColosseum/so101-stack-bowls | 76088f0fa851a5c3496636772ad68625cf801fcf | 100 | 30 | front + wrist |
| 3d | RoboColosseum/so101-stack-cubes | 14dfe1059c1c3be29244612d0f1eb0a03aa78a98 | 100 | 30 | front + wrist |
| 3e | RoboColosseum/so101-upright-bottle | de1f89d27d556f6dd3c943f568c7158c3c5822ad | 100 | 30 | front + wrist |
| 4 | armnet/busybox_multitask | eb656116832d18d112940b33db254f54eece4d9a | 132 | 20* | front + top |
| 5 | DecisionFacts/Physical_AI_SO101_Cup_Nesting_Task | bd28700173c66f785352fcaf76c413d50fbe1eab | 200 | 30 | cam_front + cam_top |

*BusyBox manifest says 30 fps; the dataset's own info.json at the pinned
revision says 20 fps and 25,734 frames — dataset metadata governs;
discrepancy recorded here. RoboColosseum frame totals in the tier manifest
also differ from the datasets' info.json (e.g. stack-cubes: manifest 14,988
vs info.json 41,359); per-episode lengths in this program are read from
each dataset's own meta/episodes parquet at the pinned revision, and the
discrepancy is flagged in the final report.

## Method (the established certification pipeline, scaled)

1. **Carry-in dedupe.** Source 1's 250 episodes already carry verdicts from
   the 2026-11-23 training-demo certification audit + Grok leg + founder
   adjudication of ep573 (249 PASS / ep604 FAIL; commits 6ad16ac, 12f4871,
   38d2e34 on study001/fl-substrate-20261121). Carried in verbatim — NOT
   re-judged.
2. **Contact sheets.** For every remaining episode (2,032): one sheet,
   two camera rows × six frames at 0/35/60/80/92/~100% of the episode,
   rendered from the real recordings (PyAV decode on the DGX).
3. **Primary judging.** Blind batches under the frozen per-task criteria
   in `criteria-frozen.md` (each source's criteria written and frozen
   BEFORE its judging starts). Verdicts PASS / FAIL / INDETERMINATE,
   end-state only, one evidence clause per episode.
3b. **Cleanliness pass** (founder amendment, 2026-11-23 17:00 CT,
   taxonomy refined 17:02 CT; criteria-frozen.md Amendment 2). Every
   judged episode — the carried-in 250 included — is graded in a
   separate pass in separate contexts, one of CLEAN / RECOVERED /
   INCIDENT, each incident described in one line (drop, slip,
   misplacement, knock, brush of a non-target, …). For passes the
   three grades label three subsets: clean = imitation core,
   recovered = recovery-training subset, incident = pass evidence
   only. The pass/fail criteria do not change; the review pack
   separates the three pass subsets.
4. **Blind re-judging.** Seeded 20% sample (seed **20261123**), judged in
   separate contexts without the primary verdicts; agreement rate and
   named disagreements reported.
5. **Grok leg.** Grok Build CLI (`/home/openclaw/.local/bin/grok` on the
   DGX, v1.0.88) blind pass over the full fresh set; per-episode
   comparison; contested episodes listed, never silently resolved.
6. **Register + review pack.** `register.csv` carries every episode:
   source, task, duration, verdict per leg, evidence pointer, status
   PENDING HUMAN APPROVAL. Review pack: all FAIL sheets, all contested
   sheets, seeded PASS samples per source/task (tier standing rule:
   seed **20261118**, fixed, never cherry-picked).
7. **Gemini leg: OWED** (see status language above).

## Files

- `register.csv` — the per-episode register (checkpointed + committed
  incrementally, source by source).
- `criteria-frozen.md` — frozen per-task completion criteria.
- `sheets/` — contact sheets for the 2,032 freshly judged episodes
  (kept on the DGX workdir; per-episode pointers in the register are
  DGX-workdir-relative; FAIL/contested/sample sheets are copied into
  `review-pack/` on this VM).
- `review-pack/` — founder review pack.
- `report.md` — the program report (written at completion).
