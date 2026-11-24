# BENCH-003 Study 001 — Preregistration DRAFT (contribution test)

**Question in one sentence:** Starting from the same `lerobot/smolvla_base`,
on the same five tasks with the same per-task data and a matched per-task
training budget, does one **federated** shared model match five per-task
**specialist** fine-tunes — and where it doesn't, is the loss federation's
or the tasks'?

> **Study:** BENCH-003 Study 001 — the contribution test (BENCH-003 design
> §10), with the §11 forgetting/saturation watch hooks registered in §7.
> **Registered by:** zk0 lane · **Date:** 2026-11-19 · **Status:** DRAFT —
> founder signature required before any training run. The BENCH-003 design
> itself is a draft for founder review; this preregistration inherits that
> status and authorizes nothing on its own.
> **Rule:** any change to §§1–8 after the first training run starts is a
> protocol amendment: appended to the deviation log (§10) with date and
> reason *before* the affected run, never retroactively.

**Verdicts, defined up front (binding text in §6):** the study ends in one
contribution verdict — **MATCH** / **BEAT** / **TRAIL** — read per task and
pooled, plus a numeric **federation cost** (FL vs the centralized ceiling
control) and an **interference read** (task-pair matrix). The M1 milestone
read is reported separately, on the same data, and never blended into the
contribution verdict (design §10.4).

**Marking convention:** every value taken from a source carries its
source; every value we chose carries **[OURS]**. Nothing is silently
borrowed.

---

## 1. Hypotheses

- **H1 — contribution (primary).** The FL shared model's judged completion
  matches the specialists': pooled difference within **5 pp**, with **no
  per-task regression** beyond 5 pp (the founder-signed contribution bar:
  parity within 5 pp of the local fine-tune, no task regressing [SOURCE:
  BENCH-003 design §10.4; node-package benchmark preregistration]).
- **H2 — attribution.** Any FL shortfall against the specialists is
  explained by the centralized arm's own shortfall (multi-task
  interference) plus a federation cost of at most 5 pp pooled. If FL
  trails where centralized holds, H2 fails and the loss is federation's
  (aggregation, client drift, adapter merging) — the aggregator A/B
  (FedProx vs dynamic weighting) is then where the fix lives [SOURCE:
  design §10.1].
- **H3 — retention (§11 watch).** Under the sequential-addition protocol
  (§7), adding tasks one at a time produces **no forgetting event** (a
  judged-rate drop > 5 pp on any previously learned task) within the five
  tasks, and the shared models' final retention matrices show no task
  silently evicted [SOURCE: design §11.3].

## 2. Tasks and data

**Suite:** ArmnetBench v0.1 single-arm tasks, registered BENCH-002 split
[SOURCE: BENCH-002 preregistration §2; BENCH-003 design §10.2].

**Training tasks (5)** — the only tasks any arm trains on:

| Task | Family |
|---|---|
| block_stack | pick-place / stacking |
| eye_drops_to_basket | pick-place (small object) |
| ring_insert | precision insertion |
| tool_insert | precision insertion (different geometry) |
| cable_clip | fastener — engage |

**Sealed tasks (3) — BENCH-002's confirmation set, untouched by this
study:** eye_drops_to_shelf, tool_removal, cable_unclip. No Study 001
process trains, tunes, selects, or evaluates on them. They are not
unsealed by any result of this study [SOURCE: design §§6, 10.2].

**Dataset (one, pinned):** `armnet/armnetbench_v01_lerobot_so101` at
revision `1db69c82b589c7172ee31a6b8f7ef58d1fc2c663` [SOURCE: baseline
collection manifest, 2026-11-18], license **apache-2.0**, LeRobot v3.0,
SO-101, 20 fps, cameras front/top/wrist. Already on the Spark
(`~/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101/`,
57 GiB, presence re-verified 2026-11-19).

**Training subsets:** the combined dataset holds 2,499 episodes = 400
teleoperated demonstrations + 2,099 labelled policy rollouts. Training
data is exactly the filter `policy_type = teleoperated` ∧ task ∈ the 5
training tasks → **exactly 50 demonstrations per task** (verified from
the files 2026-11-17 [SOURCE: on-arm data-prep §2a]). Policy rollouts —
including the suite's own SmolVLA rollouts — are **never** training data
in any arm. Per-task subsets are materialized as separate datasets with
SHA-256 recorded in the run manifest at pre-flight P0 (§5).

**Contamination status:** CONFIRMED clean against both `smolvla_base`
and `ivelin/zk0-smolvla-fl` [SOURCE: collection manifest contamination
verdicts clean/clean; tiers verification 2026-11-18]. The grade travels
with every result this study produces (design §2 rule 1).
**Completion axis, stated honestly:** the §3B completion census has not
been run on these 250 demonstrations. They are the benchmark's canonical
teleoperated demos and the study trains on them as released; the
completion axis for the *training* data is recorded as UNJUDGED, not
assumed (SSOT Decision 59 two-axis rule).

**Eval episodes:** held-out **unseen** episodes — new rollouts generated
at eval time on the real arm (§4), never training data, placements
recorded per trial and shared across arms (paired by placement index
[SOURCE: BENCH-002 §4]).

## 3. Arms and training budgets

All arms start from **`lerobot/smolvla_base`** (revision pinned at run
start, recorded in the manifest; `5e8d12a6` at data-prep time [SOURCE:
on-arm data-prep §1]). Same per-task data (§2), **matched per-task
gradient-step budget**: the unit is **K = 3,000 steps @ batch 32 per
task** [SOURCE: BENCH-002 §3.1 — the registered unit for this suite;
ArmnetBench publishes no step counts, BENCH-002 thin-source #2]. At
~705 steps/epoch on a task's 50 demos, K ≈ 4.3 epochs per task in every
arm — the matching is exact in step-equivalents per task, by
construction below.

| Arm | Training | Steps |
|---|---|---|
| **Specialists (5)** | one full fine-tune per task, on that task's 50 demos only | 5 × K = 15,000 full-FT steps |
| **Centralized pooled (1)** | one full fine-tune on the union (250 demos), sampling proportional to frame counts so each task's data is seen exactly as many times as in its specialist [method SOURCE: BENCH-001 Module M, via BENCH-002 §3.2] | 5K = 15,000 full-FT steps |
| **FL shared (1)** | Flower/FedProx over 5 nodes; node *i* holds **only** task *i*'s 50 demos (deliberately maximally non-IID, by-task partition); **30 rounds × 100 local steps/node** = 15,000 step-equivalents [SOURCE: BENCH-002 §3.2] | 15,000 adapter-subset step-equivalents |
| **Base floor (0)** | `smolvla_base` as-is, zero-shot, evaluated on all 5 tasks | 0 — context arm; required for the milestone read and the floor rule (§6) |

**Recipe (identical wherever training happens)** [SOURCE: FT re-run
recipe, 2026-11-18/19; BENCH-002 §3.1]:

| Field | Value | Provenance |
|---|---|---|
| Optimizer / schedule | AdamW lr 1e-4, β (0.9, 0.95), wd 1e-10, grad-clip 10; cosine decay, warmup 100 | paper §4.3 + released configs via BENCH-001; warmup **[OURS]** |
| Batch size | 32 | released-config family [BENCH-001 D-1] |
| Specialists / centralized scope | **full fine-tune** — LeRobot 0.7.0's full-FT mode: `--policy.train_expert_only=false --policy.freeze_vision_encoder=false` set **explicitly** (both checkpoints ship these true; a naive run is silently expert-only [SOURCE: on-arm data-prep §4]). The library's full-FT mode freezes the LM-head-only set (57,142,080 params) by design, identically for every full-FT run [SOURCE: 2×2 preregistration A-7] | |
| FL scope | frozen VLM backbone; the certified stack's **action-expert subset** aggregated — 155 tensors / ~99.9M params [SOURCE: BENCH-001 plan §6.3 Module M certified configuration, via BENCH-002 §3]. *Registered mapping:* design §10's "LoRA/action-expert adapters only" is realized as this certified subset (§10, D-2). **Registered asymmetry, stated plainly:** the FL arm trains fewer parameters by design of the certified stack; the federation cost (§6) is read with this asymmetry attached to every statement of it [SOURCE: BENCH-002 §3.1] | |
| FL aggregation | FedProx **μ = 0.1**, dynamic weighting clip(median/loss, 0.5, 2.0) [SOURCE: BENCH-001 Module M certified configuration] | |
| Chunk / flow steps / image size | 50 / 10 / 512×512 | paper §4.3 + released configs |
| Seed | 1000 | **[OURS]** (BENCH-001 D-7 precedent) |
| `push_to_hub` | false, everywhere | [BENCH-001 D-10] |
| Evaluated checkpoint | final step / final round, no selection, SHA-256 at production | [BENCH-001 discipline] |
| Checkpoints during training | full-state, ≤ ~30 min of compute apart, atomic + hash-verified, per the standing checkpoint standard; a run whose resume path is unproven does not launch at full length | standing rule 2026-11-18 |

**Camera adaptation (training):** base declares camera1/2/3; dataset
views map front→camera1, top→camera2, wrist→camera3, `empty_cameras=0`
[SOURCE: on-arm data-prep §2a]. Delta-timestamps derive from the
dataset's own 20 fps.

## 4. Evaluation protocol

**Venue — registered plainly:** there is **no faithful sim route for
ArmnetBench's tasks** (the only published SO-101 sim twin implements
different tasks [SOURCE: BENCH-002 §5]). Study 001 therefore evaluates
on the **founder's real SO-101** through the node-package bridge and its
hard safety layer, promotion path per cell (sim record where one exists
→ shadow → canary → counted trials) as in the 2×2 registration §4.3.
This is design deviation D-1 (§10): BENCH-003 §10.3's "through the
funnel" is realized as real-arm rollouts whose **recordings** pass
through the funnel's Stage 2 judging, unchanged. Twin-rebuilding these
tasks is an open program question (design §9 Q4) and is **not** assumed
by this study.

**Cells and trials:** 10 trials per cell [SOURCE: BENCH-002 §4].
Primary cells: base ×5 tasks, each specialist on its own task ×5,
centralized ×5, FL ×5 = **20 cells = 200 trials**. Sequential-protocol
evals (§7) add 15 task-cells = **150 trials**. Total ≈ **350 real-arm
trials**, phased: primary cells first, sequential second.

**Judging (§3B, dual-family):** every counted trial's recording is
judged under the §3B census protocol — COMPLETE / INCOMPLETE /
AMBIGUOUS + rationale + decisive timestamp — by **two frontier judges
from different families, blind to each other**: judge #1 = Muse (Muse Wonder)
family; judge #2 = Grok family via the Grok Bot fleet (asked
2026-11-19; **reply awaited** — design §9 Q1). Disagreements and
AMBIGUOUS verdicts escalate per §3B. Alongside, the operator records
the suite's own three-way label (successful / suboptimal / failure
[SOURCE: BENCH-002 §4]) per trial as a cross-check; the judged rate is
primary, the operator label is never blended into it.

**Calibration status — stated, not footnoted:** both judges are
calibrated on the human-labeled 230-packet anchor before any verdict
is certified; the anchor's human labeling is **not yet staffed** and is
the critical path (design §4.3). Until calibration lands, **every
verdict of this study is PROVISIONAL**, and after it lands the
false-complete rate travels with every success claim. A judge whose
false-complete upper bound exceeds 2% cannot certify completions
[SOURCE: §3B]. Certified (non-provisional) verdicts additionally
require both judge families seated; a single-family read is
provisional-internal only (design §4.2 — the two-family rule is not
negotiable).

**Failure taxonomy:** every INCOMPLETE verdict is classified (grasp
miss, place miss, out-of-range action, contact/force violation,
timeout, wrong-object, judge-ambiguous) [SOURCE: design §4.4]. The
taxonomy feeds the interference read (§6) — it is a routing instrument,
not decoration.

**Scoring within a trial:** judged completion (binary, from §3B) is
the headline unit; the suite's strict-success operator label and the
grasp/completion partial credit (0.5/0.5 [SOURCE: BENCH-002 §4]) are
reported alongside. A trial with a safety-layer abort or a human touch
mid-trial is scored per the 2×2 registration §4.3 (abort/contact
≥ 25 N PROVISIONAL → scored 0 and flagged; touch → voided, recorded).

**What n buys (stated before anyone reads a result):** at 10
trials/cell a single cell's rate resolves to ≈ ±26 pp at p = 0.5 —
per-cell numbers are descriptive. Pooled per arm, n = 50 → Wilson
half-width ≈ ±14 pp; a two-arm difference carries a Newcombe
half-width of ≈ ±19–20 pp. **The 5 pp contribution band (§6) is
therefore read on point estimates with intervals attached, exactly as
design §10.4 registers it — and any MATCH whose interval half-width
exceeds the band is reported as "MATCH (point estimate) —
resolution-limited," with the §6 remedy pre-armed.** This study will
not resolve a true 5 pp effect at these counts, and no verdict below
claims otherwise. (§10, D-3.)

## 5. Pre-flights (all before the first training run; independently verified, §9)

- **P0 — data.** Materialize the five per-task teleoperated subsets;
  re-verify from the files: exactly 50 demos/task, LeRobot 0.7.0
  loadability, per-task SHA-256 into the manifest; extract each task's
  success criteria verbatim from the ArmnetBench release into the
  manifest (a task without criteria on file does not train or evaluate
  [SOURCE: BENCH-002 §1, thin-source #1]). P0 also resolves the
  on-disk file-count discrepancy noted 2026-11-19 (1,902 files present
  vs 633 in the 2026-11-17 prep record, at the same 57 GiB) by
  re-deriving episode counts from `meta/` — counts from files govern,
  not the prep record.
- **P1 — initialization integrity.** 20 training steps of the base
  checkpoint on a training task's data: full tensor initialization,
  finite sane step-0 loss [SOURCE: BENCH-002 §3.3]. For the FL arm: one
  pilot round verifying the aggregated subset is exactly the certified
  155-tensor set and the aggregated weights hash identically at server
  and clients; the pilot round's wall-clock is recorded (it prices the
  FL arm's round overhead, §8).
- **P2 — eval camera mapping.** The dataset carries front/top/wrist;
  the founder's rig presents front + wrist [SOURCE: 2×2 runbook §§3, 5].
  Before any training, P2 records the eval-time mapping (e.g. trained
  views with `empty_cameras` covering the absent view, or training
  features restricted to the rig's views) — **identical across all
  arms**, frozen in the manifest. A mismatch discovered after training
  is a protocol failure, not a mid-study fix (2×2 §4.1 discipline).
- **P3 — timing probe on ArmnetBench data.** 20 real optimizer steps
  per training mode (full-FT; expert-subset) on ArmnetBench data
  through the training pipeline, weights discarded [method SOURCE:
  on-arm data-prep §4]. The bench venv decodes video via the PyAV
  fallback and ArmnetBench decode volume is ~100× the svla set's —
  P3's measured s/step **replaces** §8's planning rates before budgets
  harden (data-prep flag #8).

## 6. Decision rules (binding)

Statistics: rates carry Wilson 95% CIs; differences carry Newcombe 95%
CIs, with placement-paired sensitivity (McNemar on judged completion;
paired bootstrap on partial credit) reported alongside, never
substituted [SOURCE: BENCH-002 §6 discipline]. Verdicts are read
**per task first, pooled second** — a pooled average that hides a dead
task is a failed report, not a passing model [SOURCE: design §10.3].

**Contribution verdict (primary) — FL vs specialists** [SOURCE: design
§10.4]:
- **MATCH:** pooled point difference within ±5 pp **and** no per-task
  difference below −5 pp.
- **BEAT:** pooled point difference above +5 pp (per-task regressions,
  if any, reported in the same sentence).
- **TRAIL:** pooled point difference below −5 pp, or any single task
  below −5 pp with its interval excluding 0.
- **INCONCLUSIVE:** the intervals straddle the band boundaries such
  that none of the above is forced. Remedy, the only one permitted:
  **+10 trials per primary cell** for the arms involved (band
  unchanged, rules re-applied once) [SOURCE: BENCH-002 §6 remedy,
  adapted]. If still inseparable, INCONCLUSIVE is the reported result.
- Every MATCH carries its resolution label from §4 ("resolution-
  limited" when the pooled interval half-width exceeds 5 pp).

**Attribution read — FL vs centralized, same bands:** the pooled gap
is the **federation cost**, stated as a number with its interval and
the §3 trainable-scope asymmetry attached. Reading rule: FL trails a
task where centralized also trails its specialist → interference is
the suspect; FL trails where centralized holds → federation is the
suspect [SOURCE: design §10.1].

**Interference localization:** for every task where FL trails its
specialist, the failure taxonomy + the centralized arm's result on
that task jointly name the suspect. Negative transfer is reported as
a **task-pair matrix** (which pair, which direction), never as a
pooled average alone [SOURCE: design §10.4].

**Floor rule:** if the base floor and every trained arm are below 10%
judged completion on every task, the study reports **AT FLOOR** and
stops before §7's sequential protocol — no verdict words are spent on
differences between models that cannot do the tasks (2×2 floor-rule
precedent).

**Milestone read (reported separately, never blended):** M1's bars on
this study's data — federated ≥ +10 pp over the best local baseline
(the best specialist per task, pooled) and the kill line at > 10 pp
below centralized after normalization [SOURCE: design §10.4].
*Registered scope limit:* Study 001 contains **no held-out-task
generalization read** — the held-out tasks are sealed (§2) — so this
milestone read is **partial**: pooled seen-task evidence only. The
full M1 verdict waits for BENCH-002's sealed confirmation stage. A
contribution MATCH can still miss the M1 claim, and the report must
say which [SOURCE: design §10.4].

**Scope fences:** every verdict is about this suite, this base model,
this data scale (50 demos/task), and the certified FL configuration —
not about federated learning in general. The training data is public,
so the FL arm tests the pooling algorithm under a by-task partition,
not privacy and not the private-data thesis [SOURCE: BENCH-002 §6,
adopted]. The by-task partition is deliberately more non-IID than real
deployments; a large federation cost here bounds, but does not
predict, the deployed cost.

## 7. §11 hooks — forgetting / saturation watch (registered part of this study)

- **Retention matrix (standing artifact).** Both shared models
  (centralized, FL) are re-tested on **all 5 tasks** after full
  training — those rows *are* the primary eval cells (§4), issued as
  the retention matrix alongside any headline rate. A shared model's
  rate is never quoted without its retention matrix beside it
  [SOURCE: design §11.1].
- **Sequential-addition protocol** [SOURCE: design §11.2]: on the
  **centralized recipe** (full FT), tasks are added one at a time in
  the §2 table order (block_stack → eye_drops_to_basket → ring_insert
  → tool_insert → cable_clip). Each addition trains **K = 3,000 steps
  on the new task's data only**, starting from the previous addition's
  final checkpoint. After each addition, the judged eval (§4) re-runs
  on **all tasks learned so far** (15 task-cells, 150 trials). Per
  addition: the new task's learning point and **backward transfer** —
  the signed change on every prior task.
- **Forgetting line:** a judged-rate drop of **> 5 pp** on any
  previously learned task after an addition = a **forgetting event**
  (a drop whose interval straddles the line is reported as borderline,
  not silently passed) [SOURCE: design §11.3].
- **Flattening line [OURS — operationalization of design §11.3]:**
  let g_k = the sequential model's judged rate on task k just after
  addition k, minus the base floor's judged rate on task k. A
  **flattening event** = g_k < g_{k−1} − 5 pp (the same 5 pp currency
  as the forgetting line).
- **Reportable number:** the task count at which either line first
  breaks — or "no break within 5 tasks" [SOURCE: design §11.3].
- **FL retention scope [OURS]:** the FL arm trains simultaneously by
  design; its retention read is the final-round matrix above. A
  sequential-federated variant runs only under a future amendment
  (design §11.2's "where node scheduling allows" is not assumed here)
  (§10, D-4).
- H3 (§1) is read off this section's outputs.

## 8. Compute estimate and dependencies

**Training (planning rates — replaced by P3's measurements before
budgets harden):** full-FT step on this stack measured at 2.455 s/step
(20-step probe, base, svla data [SOURCE: on-arm data-prep §4]) and
3.77 s/step wall across the FT re-run (10,000 steps incl. checkpoint
saves under the guard, 2026-11-19). Expert-subset step: 1.27 s/step
probe [SOURCE: on-arm data-prep §4].

| Block | Steps | @ probe rate | @ re-run wall rate |
|---|---|---|---|
| Specialists | 15,000 full-FT | ~10.2 h | ~15.7 h |
| Centralized | 15,000 full-FT | ~10.2 h | ~15.7 h |
| Sequential chain (§7) | 15,000 full-FT | ~10.2 h | ~15.7 h |
| FL (serialized on one host) | 15,000 subset step-equiv | ~5.3 h | ~6.9 h (@1.65 s/step Stage-0 calibration) |
| **Total training** | | **≈ 36 h** | **≈ 54 h** |

Spark-hours, schedulable in background under the job guard.
**Judging:** ~350 recordings ≈ under an hour of judge throughput per
family at the §3B pilot's ~800 episodes/hour [SOURCE: §3B pilot,
2026-11-18]; packet building, not judging, is the throughput limit.

**Dependencies (hard, in order):**
1. **MPS drill verdict** (running tonight, 2026-11-19): heavy or
   parallel training on the Spark is gated on it. A FAIL does not kill
   this study — training proceeds serialized under the calm duty-cycle
   guard (design §9 Q5) — but no parallel scheduling is planned until
   a PASS.
2. **Sunday's per-trial wall-clock** (Nov 22, 2×2 session): eval
   phasing and the arm budget (~350 trials at an unmeasured
   per-trial cost; BENCH-002's illustrative 2–5 min/trial ⇒ ~12–29
   arm-hours [ESTIMATE, not a measurement]) are finalized only when
   that number is recorded. If Sunday slips, eval slips with it; no
   estimate is substituted.
3. **Anchor labeling** (unstaffed): gates certification of verdicts,
   not execution — the study may run and report PROVISIONAL verdicts;
   it may not call them certified (§4).
4. **Judge #2 seating** (Cos/fleet reply awaited): gates certified
   verdicts the same way (§4).
5. **Founder signature** on this preregistration (and the BENCH-003
   design review it hangs from): gates everything.

## 9. Independent verification — staged (builder ≠ verifier)

A stage's outputs are not quotable — internally or externally — until
its verification is filed [SOURCE: BENCH-002 §8, adapted]:
1. **Data stage:** independent re-derivation of the per-task subsets
   from the pinned dataset revision — counts (50/task), the
   teleoperated-only filter, per-task SHA-256, extracted success
   criteria — and proof that no sealed-task episode appears in any
   subset.
2. **Training stage:** every training manifest audited against §3
   (recipe fields, data hash per arm — including proof that each FL
   node's dataset hash is exactly its own task's subset and no other,
   and that sealed-task data appears in **no** training manifest);
   final-step/final-round artifacts and SHA-256s confirmed; FL
   aggregation integrity re-derived for a sampled round.
3. **Evaluation stage:** every reported rate re-derived **from the
   per-trial logs and recordings** (recount, including re-judging a
   sample under §3B), §6's rules re-applied to the recounted numbers,
   verdicts must reproduce.
4. **Filing:** each stage's certification (or corrections) filed to
   the Verification Desk as it completes; the final claim set sealed
   to its exact input set (dataset revision + subset hashes,
   checkpoint hashes, recipe, trial logs, judge calibration state).

## 10. Deviation log

Pre-registered deviations from the design text or a naive reading
(accepted at registration):

| # | Deviation | Reason |
|---|---|---|
| D-1 | Eval venue is the **real arm**, with §3B judging applied to the recordings — not sim rollouts | No faithful sim twin of ArmnetBench's tasks exists [BENCH-002 §5]; design §10.3's "through the funnel" does not name a venue, and the funnel's judging stage is venue-agnostic. Twin-rebuild (design §9 Q4) is not assumed |
| D-2 | FL trainable scope = the certified **155-tensor action-expert subset** | Design §10's "LoRA/action-expert adapters" names a family, not a tensor set; the certified stack's subset is the only auditable realization [BENCH-001 Module M] |
| D-3 | The 5 pp contribution band is read on **point estimates with intervals attached**, with a registered resolution label and the +10-trials remedy pre-armed | At BENCH-002's trial counts the pooled interval half-width (≈ ±14 pp) exceeds the band; pretending otherwise would dress noise as a verdict (§4). The band itself is the founder-signed bar and is **not** changed here |
| D-4 | Sequential-addition protocol runs on the **centralized recipe only**; FL retention = final-round matrix | Design §11.2 makes the sequential-federated variant conditional ("where node scheduling allows"); Study 001 does not assume it |
| D-5 | The M1 milestone read is registered as **partial** (pooled seen-task only) | The held-out tasks are sealed (§2); a generalization read inside this study would require unsealing them, which no result here licenses |

New deviations (blank — append only, with date + reason, **before**
the affected run):

| # | Date | Deviation | Reason |
|---|---|---|---|
| | | | |
| | | | |

## 11. Thin-source register (values this registration could not pin from sources)

1. **ArmnetBench per-task success criteria** — documented nowhere in
   our sources beyond the label scheme. → P0 extracts them verbatim;
   a task without criteria on file does not run.
2. **ArmnetBench's own fine-tune step counts** — not published. → K =
   3,000 is BENCH-002's [OURS], adopted for comparability; our
   specialists are not a replication of their SmolVLA run, and their
   published 15.0% strict is context, not an anchor [SOURCE:
   BENCH-002 §§1, 10].
3. **Real-arm per-trial wall-clock** — unmeasured. → §8 dependency 2;
   Sunday's 2×2 session is the calibration point.
4. **FL round wall-clock overhead at 30 rounds on this stack** — the
   certified loop is proven at small scale (node-package slices), not
   at this round count. → P1's pilot round measures one round; §8's
   FL line is step-time only until then.
5. **Per-task demo frame counts** — the combined dataset's totals are
   known (1,127,881 frames / 2,499 episodes incl. rollouts); the
   teleoperated subsets' exact frame counts are derived at P0. The
   ≈705 steps/epoch figure in §3 is an average-based estimate and is
   used for nothing binding.
6. **Judge #2's model/version** — the fleet's to name when it accepts
   (asked 2026-11-19). → recorded in the eval manifest when seated;
   the two-family rule does not depend on which Grok it is.

---

*Drafted 2026-11-19. Sources: BENCH-003 design draft §§10–11 (same
folder); BENCH-002 preregistration draft (2026-11-17); on-arm 2×2
preregistration (frozen 2026-11-17) for house style and inherited
protocol; on-arm data prep + verification (2026-11-17); baseline
collection manifest + tiers verification (2026-11-18); FT re-run
manifests (2026-11-19). Nothing in this draft authorizes a run.*
