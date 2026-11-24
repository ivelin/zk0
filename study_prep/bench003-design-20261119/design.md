# BENCH-003 — Design Draft

**Status:** DRAFT for founder review, 2026-11-19. Design only — nothing here
authorizes a run. Each stage gets its own preregistration with frozen
acceptance criteria before it executes, per the standing discipline.

**One-sentence purpose:** BENCH-003 is the discovery funnel — cheap, wide,
honest simulation finds what's worth believing; the real arm, our scarcest
resource, only confirms. It inverts BENCH-002's original shape: the arm stops
being the discovery tool and becomes the confirmation gate.

---

## 1. Inputs — the three policies

The FT re-run completed 2026-11-19 (both cells 10,000/10,000 steps, both
post-run verify verdicts PASS — manifest integrity, 500/500 tensors load,
forward pass finite, action shape [1, 50, 6]; zero crashes, zero resumes;
one logged deviation A-8 in each manifest). Checkpoints and manifests:
DGX `/home/openclaw/muse-work/zk0-onarm-20261117/ft/outputs/` (`ft_base`,
`ft_zk0`; `logs/*_run_manifest.json`).

| Policy | What it is | Final train loss |
|---|---|---|
| `lerobot/smolvla_base` | the shared pretrained base, zero-shot | — |
| FT-base | smolvla_base fine-tuned 10k steps on `lerobot/svla_so101_pickplace` | 0.047 |
| FT-zk0 | `ivelin/zk0-smolvla-fl` fine-tuned identically | 0.052 |

Config-diff gate from the re-run: the zk0 cell shows **zero diffs** against
the preregistered reference; the base cell differed only in `save_freq`.
Training losses are reported for the record and mean nothing on their own —
the funnel exists because loss is not capability (the 0/20 lesson).

**Admission consequence:** both FT cells trained on
`lerobot/svla_so101_pickplace`. That dataset is therefore *seen* by two of
the three policies and is excluded from every BENCH-003 eval set, alongside
the standing exclusions below.

## 2. Stage 0 — Dataset admission (nothing enters unscreened)

Builds directly on the clean-baseline collection draft
(`research_notes/zk0-baseline-dataset-20261118/`) and SSOT Decision 59:
cleanliness is two-axis (contamination × completion) and neither axis
substitutes for the other.

Rules for any dataset or task set entering the funnel:

1. **Contamination screen** against (a) the SmolVLA paper's 481-dataset
   Appendix A.1 pretraining list, (b) zk0-fl's training corpora (model-card +
   pyproject rosters), and (c) the FT cells' fine-tune set
   (`svla_so101_pickplace`). A set is admissible only at the grade its
   evidence supports (CONFIRMED / named-list), and the grade travels with
   every result it produces.
2. **Episode-level dedupe** against every other admitted set and against
   the SO-101 Bench eval tier (the 5hadytru overlap warning is the standing
   example: no pooling alongside `bench_real` until dedupe is done).
3. **Pinned revisions** — every set is referenced by immutable revision;
   a set whose upstream revision moves is a new set and re-enters screening.
4. **Sealed eval sets** — held-out task sets are sealed once admitted:
   no training, tuning, or prompt/selection decisions may be made against
   them; unsealing is a preregistered event, not a convenience.
5. **License recorded per set** (DecisionFacts' CC-BY-NC-4.0 flag is the
   standing example: usable for research reads, ruled separately before
   any commercial training use).

## 3. Stage 1 — Broad sim screen (discovery)

**What runs:** the three policies, rolled out across the admitted sim-native
suites (SO-101 Bench sim tier; RoboColosseum where licensing/hosting allows)
plus twin-built tasks (Stage 3's twin, used generatively here) plus
**procedurally generated novel tasks** — new object/layout/goal combinations
absent from every training corpus, generated under domain randomization
around *measured* parameters (friction ranges, masses, servo limits), never
around invented ones.

**Why novel tasks are first-class:** the network's claim is task
generalization. A screen that only replays known task families measures
memorization headroom, not generalization. Novel-task performance is the
funnel's headline read; known-task performance is the control.

**Execution shape:**
- Thousands of episodes; per-episode results persisted incrementally with a
  manifest (a crashed screen loses minutes, not the battery).
- Split across the Spark and Ivelin's desktop under the job guard (70%
  whole-box ceiling, explicit caps, 15 s telemetry, off-box mirror) —
  placement is Muse's call per the standing delegation, minimal footprint
  while the desktop is in interactive use.
- Kill-and-resume proven by drill before the first full screen, per the
  checkpoint standard.
- Engine roles as settled: **Genesis** is the generation front door
  (probation — per-release re-test; off the grasp path until it earns its
  way back), **Drake** is the eval gate, **Isaac Sim** is the lab tier on
  the Spark. No engine's plausible physics is sold as eval truth.

**Output:** raw completion counts per policy × task family × novelty class,
with Wilson intervals. No verdicts yet — Stage 1 produces recordings, and
recordings are not completions until judged.

## 4. Stage 2 — Judged completion census (§3B)

Every Stage 1 rollout recording passes through the §3B census machinery —
the same protocol built for the baseline dataset program, applied to
rollouts:

1. **Frontier judging** — a frontier model judges every episode (video +
   action/state agreement vs the stated task): COMPLETE / INCOMPLETE /
   AMBIGUOUS, with rationale and decisive timestamp.
2. **Dual judges, different families** — judge #1 is the Muse (Muse Wonder) family
   (the pilot's judge); judge #2 is the Grok family via the Grok Bot fleet
   (Cos was asked 2026-11-19; reply awaited). Judges run blind to each
   other and to any human labels. Disagreements escalate automatically.
3. **Calibration before certification** — both judges are calibrated on
   the human-labeled anchor set (230 packets: 55 pilot core, 8 founder
   sample, 160 genuine teleop failures across 5 task families, 9
   constructed/boundary cases; 47 deformable-flagged). The **false-complete
   rate is published alongside any success claim**; a judge whose
   false-complete upper bound exceeds 2% cannot certify completions, per
   §3B. Human labeling of the anchor is the critical path and is not yet
   staffed — **Stage 2 verdicts are PROVISIONAL until calibration lands**,
   and the design does not pretend otherwise.
4. **Failure taxonomy** — every INCOMPLETE verdict is classified (grasp
   miss, place miss, out-of-range action, contact/force violation,
   timeout, wrong-object, judge-ambiguous). The taxonomy, not the headline
   rate, is what routes work: a policy that fails on force violations has
   a different problem from one that fails on grasp misses.
5. **Deformables rule (§3B item 7)** — deformable-object episodes are never
   census-approved on judge verdict alone: twin replay or human review,
   always.

**Output:** per-policy completion rates on judged evidence only, the
exclusion lists (published, never silent), the failure taxonomy, and each
judge's calibration numbers.

## 5. Stage 3 — Twin replay escalation

Episodes that are AMBIGUOUS, judge-disputed, or *surprising* (a completion
where the twin's physics says the recorded forces should have failed, or
vice versa) are replayed in the digital twin under **harsher physics**:
friction at the edges of the measured range, honest STS3215 limits
(torque/speed/load as LeRobot can actually command and read), real gripper
geometry.

This is the Campaign 001 lesson institutionalized: the false mug pass
flipped under honest force accounting. A result that only survives friendly
physics is not a result.

**Twin-fidelity metrics** (reported per task family, per policy):
- **Success delta** — |sim judged rate − real rate| on matched tasks, from
  Stage 4's calibration pairs; the twin is trusted exactly as far as this
  delta is small, and the delta is printed next to every sim claim.
- **Contact agreement** — distribution of peak contact forces, sim vs the
  real arm's STS3215 load records, on matched episodes.
- **Failure-mode overlap** — fraction of real-arm failures whose taxonomy
  class also appears in sim for the same task (and the reverse: sim-only
  failure classes are twin suspects, not policy suspects).

Every Stage 4 real result recalibrates the twin: parameters move toward
the real measurements, and the fidelity metrics are re-issued. The twin is
a living instrument with a published error bar, not a fixture.

## 6. Stage 4 — The real arm, last and smallest

**Budget rule:** arm time is spent only on —
1. **Models that cleared the sim bar** (≥60% judged completion on the
   screen's held-out families — the founder-signed sim bar from the node
   package preregistration), and
2. **Discriminating cases**: episodes where models disagree, where the
   judges disagreed, or where sim and twin-replay disagree. Cases where
   everything already agrees teach us the least per arm-minute.

**BENCH-002 becomes the confirmation gate.** Its preregistered structure
(ArmnetBench 5 train / 3 held-out tasks, pooled endpoints, δ = 20 pp,
760-trial full campaign) is no longer run as a discovery sweep. Under
BENCH-003, BENCH-002's held-out tasks are the sealed confirmation set:
a model that the funnel believes in is confirmed (or refuted) there, at
the preregistration's pooled endpoints, with trial counts spent on the
cells the funnel's evidence says are decisive — any reduction from the
full 760 is itself an amendment to the BENCH-002 preregistration, made
before the run, never after seeing results.

**Real-arm bars (founder-signed, node package):** ≥50% over ≥10 trials per
task for a real-arm pass; every real session also harvests calibration
pairs for Stage 3 (matched task, matched setup, sim + real recordings).

### 6.1 External confirmation option — RoboColosseum (roadmap addition, founder-directed 2026-11-20)

An external evaluation stage alongside BENCH-002, not a replacement for
it. RoboColosseum is a distributed real-robot evaluation network (NUS
MAGIC Lab, Hugging Face, BitRobot, Singapore EAIC, UW, Stanford): 100+
robots, 140+ scenes, locally operated sites; embodiments include the
SO-101 — our exact arm — alongside Franka/DROID, bimanual YAM, and
Unitree G1. Two tracks: an **Open Track** (blinded pairwise comparisons
on fresh instructions, no task-specific adaptation) and a **Fine-tune
Track** (every policy fine-tuned on the same 100 demonstrations per
task, 25 attempts, scored on full success AND partial progress). All
rollouts, failures included, are published on Hugging Face in LeRobot
format. 10,000 hours of free evaluation are pledged for academic labs
and partners; applications are open via robocolosseum.ai; a lab
connects via their policy API or hands over a checkpoint to be served
next to the robot.

**Fit.** Their SO-101 fleet gives cross-site, cross-operator
generalization evidence our single arm cannot; the Fine-tune Track's
fixed-100-demo protocol is directly comparable to Study 001's
specialist/FL setup. First-study findings worth weighing: frontier
agents rivaled specialized VLAs; policy rankings barely transfer across
embodiments (Kendall W = 0.24) and Open Track rank poorly predicts
post-fine-tune rank; after fine-tuning GR00T N1.7 led on SO-101; no
policy completed any G1 task (partial progress 13.3–23.2%).

**Gating.** The funnel's arm-time rule applies unchanged: only a
checkpoint that clears our own sim ≥60% bar — and, when it exists, the
BENCH-002 confirmation — earns a RoboColosseum submission.

**Open question — eligibility UNCONFIRMED.** The free hours are framed
for academic labs and partners; zk0 is a commercial open-source project
(CocoonHive LLC). Do not assume we qualify.

**Posture.** Roadmap entry only: no application and no checkpoint
submission without Ivelin's explicit go. The shared model is Apache-2.0
(Decision 60), so a submission is legally clean once the contributor
terms are final, but timing is Ivelin's call, and the zk0 promotion
gate governs how any external results are used publicly. The W = 0.24
finding cuts both ways: a RoboColosseum SO-101 result is evidence about
SO-101-in-the-wild, not a substitute for our sealed BENCH-002 protocol.

*Claims-evidence: RoboColosseum network composition, tracks, free-hours
pledge, application route, and first-study findings · sources:
arXiv:2611.02497 (abstract) and robocolosseum.ai · checked 2026-11-20.*

## 7. What Sunday's 2×2 becomes under this design

The Sunday session (Nov 22) does not change its frozen preregistration —
the funnel wraps around it, it does not rewrite it:

- The **cold-install acceptance test** runs as the runbook specifies
  (install, doctor, M0, safety stages) — it is the node's acceptance gate,
  independent of BENCH-003.
- The **2×2 cells** run as preregistered (20 ID + 10 OOD trials per cell,
  δ = 0.20, Q2 primary). But their *role* in the program changes: they are
  the funnel's **first calibration harvest**, not a standalone verdict
  machine. Specifically, Sunday produces:
  - **REAL-1 / REAL-2 baselines** for the two starting checkpoints
    (Decision 57 — the shared checkpoint's baseline is established on the
    real arm);
  - the **per-trial wall-clock measurement** that BENCH-002's Phase A and
    Stage 4 budgeting both wait on;
  - the **first sim-vs-real calibration pairs** — Sunday's tasks get twin
    counterparts, and Stage 3's fidelity metrics get their first real
    numbers from this session.
- The FT cells' *next* comparisons (any future fine-tune, any aggregator
  A/B) enter at Stage 1, not at the arm. Sunday is the last time a model
  comparison starts at the arm; after Sunday, the arm confirms what the
  funnel found.

## 8. Discipline (non-negotiable, inherited)

- Every stage preregistered with frozen acceptance criteria **before** it
  runs; amendments are dated, reasoned, and made before results are seen.
- **Builder ≠ verifier**: the agent that builds a stage never certifies
  it; certifications file to the Verification Desk; claims seal to their
  input sets.
- Capability claims ("cannot", "beyond envelope") require direct
  measurement of the limiting quantity. A gate failure is evidence about
  quality, never capability.
- All compute under the job guard; full-path smoke test before any run
  over ~30 minutes; per-episode persistence everywhere.
- Numbers are quoted with their judge-calibration state attached:
  PROVISIONAL until the anchor calibration lands, and the false-complete
  rate travels with the claim after it does.

## 9. Open questions (founder / program decisions, not drafting gaps)

1. **Judge #2 confirmation** — Cos's reply on the Grok fleet running the
   blind second judge is awaited. If the fleet can't, the fallback is a
   second frontier family via API; the two-family rule itself is not
   negotiable.
2. **Anchor labeling staffing** — two human labelers (~4–7.5 h each +
   adjudication) are the critical path for Stage 2 certification. Who
   labels is a founder call.
3. **Novel-task generation scope** — which task families get procedural
   generators first? Recommendation: pick-and-place variants (the arm's
   proven envelope) plus one stretch family (pouring has no clean training
   source — a generated-task family there is high-information precisely
   because nothing can have memorized it).
4. **Twin-rebuild of real tasks** — admitted real datasets define tasks
   but don't run in a twin directly; rebuilding a real task in the twin
   (scene, objects, success predicate) is engineering work per task.
   Which admitted tasks earn a twin rebuild is a selection decision for
   the Stage 1 preregistration.
5. **MPS drill ordering** — the broad screen is exactly the parallel
   workload the MPS design was built for; the MPS drill remains the gate
   for heavy parallel sim on the Spark. Whether BENCH-003's Stage 1 waits
   for the drill or runs its first screen under the calm watchdog is a
   scheduling call at preregistration time.
6. **BENCH-002 amendment** — reducing/re-aiming its 760 trials under the
   confirmation-gate role needs a written amendment to its
   preregistration before any Stage 4 run.

---

*Prepared 2026-11-19. Sources: FT re-run manifests (DGX, verified
2026-11-19); clean-baseline dataset draft + validation-plan §3B
(2026-11-18); calibration anchor build (230 packets, 2026-11-19);
BENCH-002 preregistration draft; on-arm 2×2 preregistration (frozen
2026-11-17); SSOT Decisions 56–60.*

---

## 10. Study 001 — the contribution test (the funnel's first study)

The funnel exists to answer one question before all others: **does pooling
actually help?** Study 001 is that question, run as the funnel's first
study, on the record BENCH-002's draft pointed at and the M1 milestone
needs.

### 10.1 Arms (three, one baseline)

All arms start from the **same baseline checkpoint** (`lerobot/smolvla_base`),
train on the **same per-task data**, and receive a **matched per-task
training budget** (identical gradient-step count per task in every arm).
Anything less rigs the comparison.

| Arm | What trains | What it isolates |
|---|---|---|
| **Specialists** | 5 separate fine-tunes, one per task | the per-task ceiling of single-task learning |
| **FL shared** | one model federated across all 5 tasks — 5 nodes, one task's data each, Flower/FedProx, frozen VLM backbone, LoRA/action-expert adapters only, no data moves | the zk0 claim itself |
| **Centralized pooled** | one fine-tune on all 5 tasks' data pooled centrally | the ceiling control: separates *federation cost* from *multi-task interference* |

The centralized arm is what makes a loss attributable. If FL loses on a
task where centralized also loses → the tasks interfere; federation is
exonerated. If FL loses where centralized holds → the loss is federation's
(aggregation, client drift, adapter merging), and the aggregator A/B
(FedProx vs dynamic weighting) is where the fix lives.

### 10.2 Tasks

5 diverse tasks from the clean-baseline pool — diverse across objects and
interaction types, each at the pool's 50-episode unit where available.
Default candidate: **ArmnetBench's 5 train tasks** (the BENCH-002 split —
exactly 5 × 50, contamination-CONFIRMED). Final selection is frozen in
the study preregistration; diversity is a stated criterion, not a hope.
BENCH-002's 3 held-out tasks stay sealed — they are the confirmation set
(§6), untouched by this study's training or tuning.

### 10.3 Evaluation

Through the funnel, unchanged: held-out **unseen** episodes per task,
rollouts judged under §3B (dual-family judges, calibrated, false-complete
rate attached), failure taxonomy on every incomplete. Verdicts are read
**per task first, pooled second** — a pooled average that hides a dead
task is a failed report, not a passing model.

### 10.4 Verdict rules (frozen here, restated in the preregistration)

- **Contribution verdict (primary):** FL vs specialists, per task and
  pooled. **Match = parity within 5 pp with no regression** — the
  founder-signed contribution bar (parity within 5 pp of the local
  fine-tune, no task regressing). Beat and lose are the symmetric reads
  outside the band, with intervals attached (Wilson/Newcombe discipline
  as BENCH-002 §6).
- **Attribution read:** FL vs centralized, same bands. The gap between
  the two reads *is* the federation cost, stated as a number.
- **Interference localization:** for every task where FL trails its
  specialist, the failure taxonomy and the centralized arm's result on
  the same task jointly name the suspect — task-pair interference
  (which pair, which direction) vs aggregation loss. Negative transfer
  is reported as a task-pair matrix, never as a pooled shrug.
- **Milestone read:** the same data answers M1's bars (federated ≥ +10 pp
  over the best local baseline on the pooled generalization read; kill
  line at >10 pp below centralized after normalization). Study verdict
  and milestone verdict are reported separately — a contribution *match*
  can still miss the M1 *claim*, and the report must say which.

## 11. The forgetting / saturation watch (standing harness dimension)

Not a study — a **permanent measurement dimension** of BENCH-003, applied
to every model the funnel tests, starting with Study 001's shared models.
The question it never stops asking: as tasks accumulate, does the model
keep them — or does each new task quietly evict an old one, and does
learning itself flatten out?

### 11.1 Retention matrix

After any shared model trains on all 5 tasks, **every task is re-tested**
through the funnel's judged eval. The retention matrix (tasks × checkpoints:
after-each-task and final) is a standing artifact of the harness, issued
with every shared-model result. A shared model's headline rate is never
quoted without its retention matrix beside it.

### 11.2 Sequential-addition protocol

Tasks are added **one at a time** in a preregistered order. After each
addition, the full judged eval re-runs on **all previously learned tasks**
plus the new one. This yields, per added task: the new task's learning
curve point, and **backward transfer** — the signed change on every prior
task. (The FL arm of Study 001 trains simultaneously by design; the
sequential protocol runs on the centralized recipe and, where node
scheduling allows, a sequential-federated variant — the protocol measures
the *model family's* memory, and the simultaneous-vs-sequential contrast
is itself a reported read.)

### 11.3 Saturation signals (lines preregistered, not vibes)

- **Forgetting event:** a retention drop of **> 5 pp** on any previously
  learned task after a new task is added (judged rates, intervals
  attached; a drop whose interval straddles the line is reported as
  borderline, not silently passed).
- **Learning flattening:** the new-task gain per added task falls below
  the preregistered slope line as task count grows — the model's
  *marginal* task is worth less than the last, on comparable task
  difficulty (novelty class held as constant as generation allows).
- **The reportable number:** the **task count at which either line
  breaks** — "the shared model holds N tasks cleanly; task N+1 is where
  it starts paying for new skills with old ones." That number, tracked
  across model generations, is the saturation curve of the whole program.

### 11.4 The built-in mitigation, measured not assumed

The federation recipe is itself a forgetting mitigation on paper: the
VLM backbone is frozen, only LoRA/action-expert adapters train, and
per-arm calibration absorbs hardware variation. Plausible — and
unproven. The watch exists to measure whether adapter-only learning
actually protects old tasks, or merely forgets more politely. If the
lines break early, the wall queue's compressed-history work and the
aggregator A/B stop being future options and become the program.

---

*Addendum 2026-11-19 (same drafting session): Study 001 and the
forgetting/saturation watch folded in at the founder's direction. Study
001's task selection, step budgets, FL recipe parameters (FedProx μ,
weighting), and the sequential-addition order are frozen in its own
preregistration before any training runs.*

---

## Erratum — judge-name label correction (2026-11-23)

Per founder order 2026-11-23 22:54 CT ("Audit all records. Remove the
Claude mislabeling and treat Muse models clearly and differently from
Gemini Google models and agy Claude models"), the judge label in §3B
item 2 of this design is corrected in place:

- **Original wording:** "judge #1 is the Claude family (the pilot's judge)"
- **Corrected wording:** "judge #1 is the Muse (Muse Wonder) family (the pilot's judge)"

The judge leg so labeled was at all times Muse Wonder (Meta), executed
in isolated subagent instances of the assistant. Anthropic Claude was
never the judge in this program. The mislabel originated in Muse's own
2026-11-19 note on its 2026-11-18 pilot run and propagated into this
design (2026-11-20) and from there into the sealed preregistration,
which carries its own matching erratum. This erratum changes ONLY the
judge-name label; no design provision is altered. Full audit record:
`claude-mislabel-audit-20261123.md` (same folder).
