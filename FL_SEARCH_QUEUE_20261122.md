# FL Setup Search Queue — canonical, living (2026-11-22)

**Status: ACTIVE SYSTEM, not a design note.** This document supersedes
`fl-grid-search-design-20261122.md` as the queue's operating document
(that note remains the design record; §3 below lists every place this
document overrules it and why). Founder directives behind it:
2026-11-22 12:06 CT (one unified dynamic queue), **15:42 CT (execute
diligently and autonomously; highest impact / lowest effort first;
re-sort as new data lands; he checks in and reprioritizes)**, and
**15:43 CT (document + commit + continuous-verification loop, §1)**.

**The yardstick.** Certified Study 001 result, independently verified:
FL **26/100** judged completion vs C-exp **70/100** on the same frozen
100 held-out episodes → **Φ = −44 pp** (strict-reading bound −45).
Base 0/100. Every FL-training item below is scored against closing
that gap; every claim is judged on the same 100 episodes, same rubric,
labeled "JUDGED completion (offline replay)". Noise floor: binomial
noise at n=100 is ±~4–5 pp and training variance is unmeasured, so
**effects <10 pp rank items but carry no claim**; claims require the
confirmation standard in §2.

---

## 1. Standing protocol (founder addendum 2026-11-22 15:43 CT — applies to EVERY queue item)

1. **Document + commit.** Every finding from queue work is carefully
   documented and committed to remote repos for backup and
   third-party reproducibility: non-sensitive experiment records to
   the public repo (`ivelin/zk0`) on the working branch
   `study001/fl-substrate-20261121` (public-by-default rule);
   significant checkpoints to Hugging Face with hashes recorded;
   secret-sauce only to Drive. **Each item's protocol includes its
   documentation + commit step; a result that exists only locally is
   not done.**
2. **Continuous independent verification.** Verifiers in separate
   contexts, briefed with raw artifacts and acceptance criteria —
   never the builder's conclusions — are asked continuously to look
   for gaps in queue results, configs, and the pipeline itself. No
   result is reported as settled before its verification lands.
3. **Regression closure.** Every gap a verifier finds becomes a
   permanent test in the regression suite — the config-equivalence
   gate pattern (fast tier, runs in minutes, before any expensive
   work). Open gaps and their test status are tracked in §8.
4. **Per-item execution protocol** (founder, 15:42 CT): one variable
   at a time against the certified Study 001 FL recipe · config-
   equivalence gate **PASS** before launch (checker diffs resolved
   configs from constructed objects against the item's registered
   manifest; the original incident config must FAIL it — drilled
   2026-11-22) · full-path miniature smoke (train → save → reload →
   hash → package → standard load, every production path on) ·
   serialized Flower substrate only, launched via
   `launch_dgx_job.sh` (guard, exactly one fleet-lock layer, mirror
   target set, breaker logonly, thermal monitor armed for the run's
   duration; 70% whole-box ceiling) · judged eval on the frozen
   held-out 100 with the standing rubric · independent verification ·
   HF backup of significant checkpoints with hashes · documentation +
   commit (rule 1). **Never** write to `ivelin/zk0-smolvla-fl`
   (pin discipline stands).

## 2. Scoring

Same ranking function as the design note, for continuity:

> **Score = (Impact × P ÷ Effort) × 100**, Effort = DGX-hours +
> engineering-hours for one honest test.

- **Impact (1–5)** judged against the yardstick above (5 = closes or
  decisively explains the Φ gap; 4 = large standing gain; 3 =
  material; 2 = useful/local; 1 = marginal).
- **P** = probability the item produces its intended positive result,
  from cited evidence.
- Scores rank; they do not measure. Dependencies and trigger gates
  override raw order (§6 gives the actual GPU execution sequence).
  The founder can re-order at will.

## 3. Critical assessment — where this document overrules the earlier design

Checked against the certified record on 2026-11-22; each supersession
is dated and reasoned, none is silent.

1. **The factorial's camera factor (C) is dead — dropped.**
   The design note's Workstream 1 treated "2 vs 3 cameras" as a
   training factor with Study 001's run as the 2-camera baseline
   cell. The certified record disproves the premise: FL *training*
   consumed all three views for all 30 rounds (proven from primary
   sources, `study001-fl-3cam-training-verification-20261122.md`);
   the two-camera condition existed only in the saved checkpoint's
   declared config and therefore in the eval feed. A training factor
   whose "−" level never existed cannot be run. **C is removed.**
2. **The factorial itself is superseded by one-variable-at-a-time.**
   With C gone, the founder's 15:42 directive ("each item: one
   variable at a time") post-dates and overrules the 2^(4−1)
   fractional factorial. Factors A (AdamW carry), B (round
   structure), D (server momentum) become single-variable items
   Q-A / Q-B / Q-D below. We lose interaction contrasts; we gain
   clean attribution per ~10.7 h run and compliance with the latest
   directive. If A/B/D readouts ever suggest an interaction, a
   combined cell can be registered as a new item at a re-sort.
3. **The "clean FL re-run" is removed from the queue.** The design
   note's addendum made it the GPU head to obtain a trustworthy Φ
   row. That row now exists: the 3-camera re-eval of the existing
   round-30 checkpoint (metadata-only repair, weights hash
   identical) gave FL 26/100, Φ = −44, independently verified
   PASS-WITH-FIXES. Re-running 10.7 h of identical training would
   reproduce a number we already hold, and the artifact defect is
   separately repaired (A5 fix, in place + HF backup updated at
   founder approval). What survives from the sweep is the
   **substrate hardening** (D1 server-template fix, D2 server-eval
   landmine) — now item S0, a prerequisite for every FL queue run,
   enforced by the gate rather than by a re-run.
4. **Design-note Q1 (defect A5) is CLOSED.** Repaired, standard-load
   verified, HF backup `zk0bot/study001-fl-round30` updated
   (2026-11-22, founder-approved push). Evidence:
   `study001-a5-repair-20261122.md`. Its packaging correct-by-
   construction half lives on in the gate (P2 standard-load test).
5. **Framework: Optuna v5 as the record layer only — accepted with
   a caveat.** The design note's framework research stands (Optuna
   v5.0 Sep 2026 is the community standard; Ray Tune rejected —
   cluster-shaped, and the serial rule reduces it to a for-loop
   carrying a distributed runtime; Hyperopt/scikit-optimize dead).
   Caveat: with single-variable items and ~10.7 h trials, samplers
   earn nothing yet. Optuna is adopted as **bookkeeping**: one study
   per item, SQLite storage, `n_jobs=1`, each objective call wrapping
   one guarded run — a permanent queryable record for the public
   experiment record. Execution stays plain config runs via the
   launch wrapper. If the continuous stage (Q-HPO) opens, the same
   store switches to `TPESampler` with no re-platforming. Item S1.
6. **Multi-fidelity / ASHA stays rejected** (design note §6 stands):
   factor B *is* round structure (truncation confounds it), and the
   cheap proxy doesn't predict the metric (trajectory-error AUC
   0.535 on Study 001's own data). Q-PROXY exists to revisit this
   with queue-generated evidence.

## 4. The queue — sorted

Track: **G** = GPU track (serial DGX), **P** = parallel track
(≤2 DGX-h or VM/CPU; runs in slack, does not wait for GPU position).
Lane: **Φ** = closes/explains the FL gap · **M** = measurement ·
**I** = inference efficiency · **A** = architecture/reference ·
**S** = standing infrastructure.

| # | Item | Trk/Lane | Imp | P | DGX-h | Eng-h | Score | One-line reasoning |
|---|---|---|---|---|---|---|---|---|
| S0 | **Substrate hardening for queue runs** (D1 server-template fix, D2 eval-off assertion, queue-run gate manifest, batch-log diagnostic fix) | P/S | 3 | .95 | 0 | 6 | **47.5** | Prerequisite for every FL run below; converts the postmortem's fixes from prose into the substrate + gate, at zero GPU cost |
| S1 | **Optuna v5 record layer** (one study per item, SQLite, wrapper around guarded runs) | P/S | 2 | .90 | 0 | 4 | **45.0** | Cheapest permanent win: every queue result becomes a queryable public record from item #1 onward |
| Q-NAT-A | **NAT VLA Slot A — as-released reference run** | G/A | 3 | .85 | 3 | 8 | **23.2** | ~3 DGX-h buys the only external reference point for the token-space lane; its number re-prices Q-NAT-B and Q-SPEC |
| Q-CSD | **Chunk-switch discontinuity metric in the eval harness** | P/M | 2 | .90 | 0 | 8 | **22.5** | Zero GPU; quantifies FL's known tool_insert plunge signature, which the current table cannot express |
| Q-STEPS | **FL +20% steps continuation** — 6 more rounds from the round-30 checkpoint (36 rounds, 18,000 steps) — *trigger-gated: activates only if the rounds 10/20 diagnostic shows judged completion still climbing r20→r30; score conditional on the trigger* | G/Φ | 3 | .50 | 3.7 | 4 | **19.5** | Cheapest test of "just train longer": median client loss was still falling at r30 (0.246→0.222), not plateaued; if judged scores are flat r20→r30 it drops to low priority — steps would not be the binding constraint (protocol §5) |
| Q-VLR | **Vision-encoder knobs: unfreeze check + differential vision LR** (specialist scale) | G/Φ | 3 | .60 | 4 | 6 | **18.0** | Published ablation says a frozen encoder cost PushT 57.8%→25%; one controlled check replaces an inherited default in every future fine-tune |
| Q-PROXY | **Proxy-metric validation for future multi-fidelity** | P/M | 3 | .60 | 0 | 10 | **18.0** | Converts accumulated judged cells (rounds 10/20/30 + queue items) into a validated — or killed — early-stopping proxy; makes all later search cheaper |
| Q-DW | **Dynamic-weighting offline audit** (did loss-based weighting help or hurt Study 001's FL?) | P/M | 2 | .50 | 0 | 8 | **12.5** | Free evidence from existing round logs; sets Q-WA's P and sharpens Q-D's interpretation |
| Q-INT8 | **INT8 inference, action-aware PTQ calibration** | G/I | 3 | .75 | 2 | 16 | **12.5** | ~2× memory/bandwidth on every node forever; precision tasks are built-in canaries behind the 5 pp gate |
| Q-KV | **Static-prefix KV cache + torch.compile in the inference stack** | P/I | 4 | .80 | 2 | 24 | **12.3** | ~4.2× reported on SmolVLA-class, training-free, multiplicative with Q-CD; pure engineering |
| Q-CD | **Few-step consistency distillation of the flow expert** | G/I | 4 | .70 | 4 | 20 | **11.7** | The denoise loop is ~79% of action latency; published 10× precedent at equal success, teacher is our own checkpoint, zero license surface |
| Q-B | **Fewer/longer rounds: 300 steps × 10 rounds** (total steps fixed at 3,000/client) | G/Φ | 4 | .40 | 12.2 | 3 | **10.5** | Tests whether 30 rounds of averaging wash out local progress — the aggregation-frequency half of the FL-vs-C-exp structural difference |
| Q-A | **AdamW carry across rounds** (per-client optimizer state persisted server-side, never over Flower) | G/Φ | 5 | .45 | 12.2 | 10 | **10.1** | The only completely untested registered-recipe factor; C-exp's continuous AdamW is the largest structural difference besides pooling itself — prepped, §7 |
| Q-TTB | **Temporal token budgeting ablation** (frames × tokens/frame at fixed budget) | G/I | 3 | .60 | 8 | 12 | **9.0** | Prices our recurring per-inference cost (a few seconds of footage) in one combined slot |
| Q-D | **Server momentum (FedAvgM, β=0.9)**, weighting and μ held constant | G/Φ | 4 | .35 | 12.2 | 6 | **7.7** | Standard fix for drift under heterogeneous clients; ours are task-partitioned, the high-heterogeneity case momentum exists for |
| Q-WA | **Dynamic-weighting ablation: weighting OFF** (pure n-weighted FedAvg) | G/Φ | 3 | .35 | 12.2 | 2 | **7.4** | The weighting was an adaptation never independently validated; a config flag decides whether it earns its place |
| Q-PSW | **Per-sample loss weighting probe** (1/√length + pad-only masking, flow-expert analog, specialist scale) | G/Φ | 2 | .30 | 4 | 6 | **6.0** | A free knob from the NAT recipe; cheap to kill or adopt at small scale before any FL-scale test |
| Q-INT4 | **INT4 + QAT recovery** — *gated: only if Q-INT8's PTQ delta >5 pp on precision tasks* | G/I | 3 | .50 | 6 | 24 | **5.0** | Exists only as Q-INT8's contingency; ~0.25 GB weights if it works |
| Q-LR | **Loss-slope-adaptive learning rate** — round-to-round AdamW LR set by the slope of the median client policy loss instead of the fixed round-indexed schedule — *conditional: behind tonight's diagnostic and the first structural readout (Q-A/Q-B/Q-STEPS per §6); score conditional* | G/Φ | 3 | .35 | 12.2 | 12 | **4.3** | With per-round AdamW reset, the round-level LR is one of the few cross-round control surfaces the server actually has; a slope-driven rule is a second-order refinement until the bigger levers have readings (protocol §5) |
| Q-HPO | **Continuous hyperparameter stage** (Optuna TPE over client LR, FedProx μ, server LR) — *dependency-gated on Q-A/Q-B/Q-D readouts* | G/Φ | 4 | .40 | 32 | 8 | **4.0** | Tuning continuous knobs around a structurally unsettled recipe wastes trials; runs once the structural factors are decided |
| Q-T2S | **Teacher→student distillation as a standing recipe stage** (public/sim data only; decoupled Top-K objective) | G/Φ | 4 | .45 | 12 | 40 | **3.5** | Buys intelligence federation alone can't reach and doubles as the dataset-admission auto-labeler; P is the honest unknown at 450M |
| Q-MAT | **MatFormer-style nested sizes for heterogeneous node inference** | G/A | 2 | .50 | 10 | 40 | **2.0** | Train once, extract smaller sub-models for weak nodes — inference only; does not solve federated training |
| Q-VLM | **VLM swap (Qwen3-VL-2B)** — *trigger-gated: token-space policy plateaus on perception/grounding* | G/A | 4 | .35 | 20 | 60 | **1.8** | Nothing measured indicts SmolVLM2 yet; a swap invalidates the certified census, stats, and Study 001 comparability |
| Q-NAT-B | **Slot B — home-grown NAT-style causal tokenizer on SmolVLM2** | G/A | 5 | .30 | 30+ | 120+ | **1.0** | The strategic moonshot: Impact 5 if token-space federates, but the federation-payload problem (no backbone/expert split) is unsolved and the build is 120+ eng-h |
| Q-SPEC | **Speculative decoding for action tokens** — *gated: only if Q-NAT-B wins its slot* | G/I | 3 | — | — | — | **gated** | No tokens to speculate on in a flow-expert policy; priced in advance by the survey |

**In flight, feeding the queue (not queue items):** the final eval
batch — sequential-chain row, five expert-only specialist rows, and
the **FL rounds 10/20 diagnostic + round-by-round loss curve**. Its
readout is re-sort input #1 (§6).

## 5. Per-item protocol notes

The §1 standing protocol applies in full to every item. Each item
adds exactly one variable, its metric, and its specifics:

- **S0** — No experiment. Deliverables: server entry in the substrate
  config points at the study dataset (or is removed); an explicit
  server-eval-OFF assertion in every queue-run manifest; the queue
  gate manifest derived from `study_prep/fl_rerun_study_manifest.json`;
  the training loop's batch diagnostic fixed (it read a nonexistent
  key — `batch_img=N/A` for all 15,000 steps of Study 001's run) with
  a regression test (§8, G-4). Judged by: gate drill — the original
  Study 001 config FAILs, the hardened config PASSes, on miniature
  artifacts.
- **S1** — No experiment. One Optuna study per queue item; the
  objective wrapper records config hash, gate report id, judged
  score, and artifact hashes for every trial. Judged by: the first
  GPU item's record is queryable end-to-end.
- **Q-A** — Variable: optimizer-state policy reset → carry. Full
  prep package in §7. Judged by: judged completion on the frozen
  100 vs FL 26/100 baseline.
- **Q-B** — Variable: round segmentation 100×30 → 300×10, total
  local steps fixed at 3,000/client. The per-client LR schedule
  stays a cosine over the client's own 3,000 steps (realized via the
  same `lr_at` rule); only segmentation changes — the schedule
  semantics are part of the registered variable, stated in the
  item manifest so the sweep's LR-horizon flag stays visible.
  Judged by: frozen 100 vs 26/100.
- **Q-STEPS** — Variable: total steps 15,000 → 18,000. Continue the
  certified FL run from its round-30 checkpoint for 6 more rounds
  (36 total) via the run's existing resume mechanism
  (`initial_model_path` / `round_offset`, D-11); recipe, datasets,
  substrate identical — the only change is where training stops.
  **Trigger (evaluated at re-sort #1, §6):** activates only if the
  rounds 10/20 diagnostic shows judged completion still climbing
  from round 20 to round 30 by a real margin; if judged scores are
  flat r20→r30, the item drops to low priority — falling loss
  without rising completion means steps are not the binding
  constraint. **Budget asymmetry, labeled:** FL at 18,000 steps vs
  C-exp at 15,000 is NOT an attribution comparison; it answers the
  practical question "can FL get closer given more time?", and
  every report of its result carries both step counts.
  **Watch item:** the eye_drops client's loss rose from round 21 of
  the original run; the continuation's per-client losses are
  reported round by round, and further divergence is a finding,
  not noise. Judged by: frozen 100 vs 26/100, reported as
  FL @ 18,000 steps.
- **Q-D** — Variable: server update gains FedAvgM β=0.9; dynamic
  weighting and FedProx μ=0.1 held exactly as certified. Judged by:
  frozen 100 vs 26/100.
- **Q-WA** — Variable: dynamic weighting off (weights = n_examples
  only). Runs after Q-DW's audit sets its final P. Judged by:
  frozen 100 vs 26/100.
- **Q-VLR** — Two short specialist-scale runs (encoder frozen vs
  unfrozen at ½ LR), judged on the specialist completion table vs
  the frozen-encoder counterpart. Result amends the default
  fine-tune recipe queue-wide.
- **Q-PSW** — Specialist-scale probe first; an FL-scale run is a
  separate re-sort decision, never automatic.
- **Q-NAT-A** — One fine-tune + eval of `ahad-j/nat-vla-450m` as
  released, same 5 tasks / 50 demos / step budget, same table.
  Reference only: the LFM Open License ($10M ceiling) bars it as the
  shared base regardless of result. Dependency: NAT tokenizer repo
  license checked and recorded before any tokenizer weights are
  touched.
- **Q-CSD / Q-DW / Q-PROXY** — Analysis items on existing artifacts;
  each lands as a harness metric or a written audit in the public
  record, with its own verification pass.
- **Q-KV / Q-CD / Q-INT8 / Q-TTB / Q-INT4** — Inference items, all
  behind the standing regression gate: no per-task judged-completion
  regression beyond 5 pp vs the unmodified checkpoint; secondary
  metrics ms/action-chunk, control Hz, checkpoint size, peak memory.
  Q-CD distills each shipped checkpoint as a post-training stage
  (steps 10 → 2–4). Q-INT4 exists only on its gate.
- **Q-HPO** — Opens only after Q-A, Q-B, Q-D have read out. TPE over
  client LR ∈ [5e-5, 2e-4], μ ∈ [0, 0.3], server LR ∈ [0.5, 1.0];
  each trial is a full protocol run; the ≥10 pp promotion rule (§6)
  applies per trial.
- **Q-LR** — Variable: how the round-to-round LR is determined.
  The certified recipe sets each round's LR from the round index
  (per-round `lr_at`(mid-round), flat within-round, per D-11);
  Q-LR replaces that determination with a **loss-slope-driven
  rule**: the next round's LR is a function of the recent slope
  of the training curve. Everything else identical. **Signal:**
  the per-round **median client policy loss** — the series we
  actually have (0.391 → 0.322 → 0.246 → 0.222 over rounds
  1/10/20/30) — with the slope estimated over a smoothing window
  (e.g. the last 3 rounds), because 5 heterogeneous clients make
  single-round deltas noisy. The controller must **NOT** read the
  server composite eval loss: that per-round value is broken
  (rounds-10/20 prep gap G1 — every round's `metrics.json`
  carries round 0's 0.5773 verbatim, masquerading as a per-round
  value); its fix is owned by S0. **Pre-registration
  requirement:** the exact rule — window, slope threshold,
  multiplier, LR floor and ceiling — is frozen in the item's run
  manifest **before launch** and becomes part of the config the
  equivalence gate checks. A data-dependent LR makes the run's
  config partly endogenous; the frozen rule is what keeps the
  test honest and reproducible. No post-hoc tuning of the rule
  mid-run. **Why it is a clean lever here:** with per-round
  AdamW reset (the certified recipe), optimizer state does not
  cross rounds, so the round-level LR is one of the few
  cross-round control surfaces the server actually has.
  **Interaction:** if Q-A (AdamW carry) ever becomes the recipe,
  Q-LR's design must be revisited before it runs; the two are
  tested as separate single-variable items, never combined in
  one run. **Position:** conditional and second-order — behind
  tonight's diagnostic and behind the first structural readout
  (§6). Smoke addition to the §1 protocol: the controller is
  exercised on synthetic loss series before the miniature fit,
  including a flat series (rule must hold LR at its floor/ceiling
  behavior as registered, not drift) and a diverging-client
  series (median must not chase one client's divergence).
  Judged by: frozen 100 vs 26/100.
- **Q-T2S** — Teacher = strongest available checkpoint (centralized
  model or a larger released VLA) on public/sim data only;
  contributor data never meets the teacher. Also delivers the
  offline auto-labeler for dataset admission.
- **Q-MAT / Q-VLM / Q-NAT-B / Q-SPEC** — Bottom of the queue by
  arithmetic, not disrespect. Each carries its gate in §4; a gate
  firing is a re-sort event.

## 6. Sequencing and the diagnostic conditional

**DGX reality (2026-11-22):** sequential chain until ~17:20 CT →
expert-only specialists until ~02:00 CT Nov 23 → final eval batch
(sequential + spec_exp rows + FL rounds 10/20 diagnostic + loss
curve). **No queue training launches until the final eval batch
lands and the diagnostic is read.** Parallel-track items (S0, S1,
Q-CSD, Q-DW, Q-PROXY prep) proceed in slack now — prep only, no GPU.

**The conditional (re-sort #1, decided by the diagnostic):**
Read FL judged completion at rounds 10 and 20 against round 30's
26/100, with the per-round server composite loss curve beside it.

- **PEAKED branch** — if max(score@10, score@20) − score@30 ≥ 10 pp,
  or the loss curve shows the composite rising over the final third
  while client losses fall (aggregation destroying fit): the
  averaging story leads. **GPU order: Q-B → Q-D → Q-A.**
- **WEAK-PEAK lean** — a 5–9 pp mid-run advantage: same order as
  PEAKED, logged as weak evidence (inside the claim noise floor,
  sufficient for ranking only).
- **NEVER-TOOK-OFF branch** — scores@10/20 within ±5 pp of 26 with
  an early loss plateau: the optimization-continuity story leads.
  **GPU order: Q-A → Q-B → Q-D** (the prepped default, §7).
- **AMBIGUOUS** — neither signature clean: default order
  Q-A → Q-B → Q-D, ambiguity logged in the changelog.

**Q-STEPS trigger (same readout):** if judged completion is still
climbing from round 20 to round 30 by a real margin, Q-STEPS (§5)
activates and slots into the GPU order as a short run (~2.2 h
training) ahead of the structural three; if judged scores are flat
r20→r30, it drops to low priority and the branch order above
stands unchanged.

**Q-LR position (added 2026-11-22 16:37 CT):** the
loss-slope-adaptive LR item (§5) is conditional and
second-order: it does not enter the GPU order until tonight's
diagnostic has been read **and** the first structural readout
(whichever of Q-A / Q-B / Q-STEPS fires first under the branch
order and trigger above) has landed. Its rule is pre-registered
in its run manifest and gate-checked like any other config
field. If Q-A (carry) ever becomes the standing recipe, Q-LR's
design is revisited before it runs — the two are never combined
in one run.

After the structural three: Q-WA (post-audit), then Q-HPO unlocks.
Cheap GPU probes (Q-NAT-A, Q-VLR, Q-PSW) slot into short windows
between structural runs when the DGX calendar allows; inference
items follow their own dependency order (Q-KV ∥ Q-CD → Q-INT8 →
Q-TTB; Q-INT4 on gate). Moonshots stay at the bottom until evidence
re-prices them.

**Standing re-sort rule** (after every completed item, eval, or
external result): (1) re-judge P for every item sharing a mechanism
with the completed one, reason logged — no silent re-ranks;
(2) recalibrate effort with realized DGX-hours; (3) any ≥10 pp
judged effect spawns a confirmation item at the top of the GPU
track; (4) gated items re-score only when their trigger fires;
(5) update §4's table and the changelog (§9) in the same commit.

## 7. Prep package — Q-A AdamW carry (provisional GPU #1, launch-ready, NOT launched)

**Exact config diff vs the certified Study 001 FL recipe — one
field:**

| Field | Certified recipe (as-run) | Q-A |
|---|---|---|
| Optimizer-state policy across rounds | Fresh AdamW per fit (reset) | **Per-client AdamW state loaded at fit start, saved at fit end** |
| Everything else | — | **Identical**: same client datasets + P0 episode lists; 30 rounds × 100 steps × batch 32; AdamW preset lr 1e-4, β (0.9, 0.95), ε 1e-8, wd 1e-10, grad-clip 10; per-round LR values from the same `lr_at`(mid-round) rule, flat within round — the scheduled LR is re-applied to param groups *after* state load, so a stale LR cannot ride in with the state; FedProx μ = 0.1; dynamic weighting clip(median/loss, 0.5, 2.0); native bf16/fp32 precision mix; seeds 1000 + client_id; expert scope census 155 tensors / 99,880,992 params; S0-hardened server template; server eval OFF |

**Code change spec** (study branch, DGX workdir
`~/muse-work/zk0-study001-fl-20261121`):

- `src/training/training_setup.py` — the optimizer is built via
  `lerobot.optim.factory.make_optimizer_and_scheduler` from
  `policy.config.get_optimizer_preset()`. Add an optional state
  path: after construction, if a state file exists, load
  `optimizer.state_dict()` (moments + step counts), then re-apply
  the round's scheduled LRs to the param groups.
- `src/training/train.py` — the per-fit entry (setup call at its
  head): accept the state path, pass it through, and at train end
  save `optimizer.state_dict()` atomically (temp file + rename) —
  a kill mid-save can never corrupt the last good state.
- `src/client/client_core.py` — `fit()` (line 166): derive a
  per-client state directory from the client/partition identity +
  run id, under the run's output tree; pass the path into training.
  **State never travels over Flower parameters** — the exchanged
  parameter census is asserted identical to baseline in the smoke.
- No other files change. The diff lands as its own commit on the
  study branch before the smoke, per the document + commit rule.

**Gate inputs.** A queue-run manifest derived from
`study_prep/fl_rerun_study_manifest.json` with the optimizer-state
policy field set to `carry` as the item's registered variable (the
P1 checker's dump list already includes cross-round optimizer-state
policy, per the postmortem spec). Gate must PASS on the miniature
artifacts; cross-arm equivalence vs the certified recipe must show
exactly this one explained difference.

**Smoke plan** (full-path miniature, before any full run): 2 clients
× 2 rounds × 10 steps on the study datasets — (a) state file exists
after round 1 and is loaded at round 2 (AdamW step counter reads 10
at round-2 start, 20 at end — logged, asserted); (b) kill-and-resume
drill mid-round-2: resume restores state and completes with a
continuous record; (c) P2 standard-load test on the miniature
checkpoint; (d) gate PASS report filed in the run manifest;
(e) Flower payload census identical to baseline.

**Launch package (prepared, not executed):** `launch_dgx_job.sh`
study mode with the Q-A manifest, guard + mirror + breaker logonly,
thermal monitor armed, expected ≈10.7 h training + ≈1.5 h judged
eval. Launch follows the §6 conditional — Q-A is first only in the
NEVER-TOOK-OFF / AMBIGUOUS branches; in the PEAKED branch Q-B's
prep (a config-only variant of this package) goes first instead.

## 8. Gap & regression-test register

Standing rule (§1.3): every verifier gap becomes a permanent test.
Status checked against the 2026-11-22 gatefix re-audit.

| ID | Gap (source) | Regression closure | Status |
|---|---|---|---|
| G-1 | Server template declared 2 cameras vs 3-camera design (postmortem Incident 1) | P1 config-equivalence checker incl. server template + saved-checkpoint config; drilled FAIL on the original config | **CLOSED** — re-audited PASS 2026-11-22 |
| G-2 | Packaged checkpoint failed the standard load (defect A5) | P2 standard-load test in the fast tier, on the miniature, before any full run | **CLOSED** — in gate; artifact repaired + verified |
| G-3 | Eval judged arms through unequal declared inputs (postmortem D9) | `eval_gate_check.py`: identical declared inputs across a comparison set, enforced in the harness (refuses, exit 3); drill FAILs a set containing the original two-view FL checkpoint | **CLOSED** — re-audited PASS 2026-11-22 |
| G-4 | Training loop's batch diagnostic read a nonexistent key (`batch_img=N/A`, all 15,000 steps) — no per-step batch evidence exists for Study 001 | Fix the diagnostic + a unit test asserting batch keys are logged on a miniature fit | **OPEN** — owned by S0 |
| G-5 | Server-eval landmine: server eval wired to the leftover two-camera dataset, inert only because it was disabled (sweep D2) | Manifest eval-OFF assertion (S0) + a checker field asserting the server dataset identity in every queue-run manifest | **OPEN** — owned by S0; fail-closed until then (no queue run may enable server eval) |
| G-6 | Stale two-view FL records still sit in the eval active set (gatefix re-audit residual) | Move stale records out of the active set before the final eval batch judges; eval gate fails closed meanwhile | **OPEN** — owned by the final-eval-batch work; queue evals inherit the cleaned set |
| G-7 | Launcher gate was opt-in and armed nowhere (pipeline audit F1) | Wrapper detects study launches itself; refusal drilled with nothing set | **CLOSED** — re-audited PASS 2026-11-22 |
| G-8 | Gate covered the FL arm only (pipeline audit F3) | Checker diffs resolved configs for every arm; negative control (corrupted LR) failed by name | **CLOSED** — re-audited PASS 2026-11-22 |
| G-9 | One-off 3-camera judging copy is ungated (gatefix re-audit residual) | None — it will not be reused; its results were blind-verified 24/24. Queue judging uses only the gated harness | **CLOSED by non-reuse** — recorded so the exception is explicit, not forgotten |

## 9. Changelog

- **2026-11-22 15:43–16:00 CT — Queue system built (this document).**
  Inventory folded in: architecture/recipe backlog note +
  efficiency survey, the grid-search design note, the pre-run
  config-equivalence sweep, the config postmortem, MEMORY's queue
  entries. Supersessions in §3 (camera factor dropped; factorial →
  one-variable-at-a-time per the 15:42 directive; clean re-run
  removed — superseded by the verified 3-camera re-eval + A5
  repair; design-note Q1 closed). Framework: Optuna v5 as record
  layer (S1). Provisional GPU #1 = Q-A AdamW carry, prepped to
  launch-ready (§7), launch held for the §6 diagnostic conditional.
  Standing protocol (§1) adopted from the founder's 15:43 addendum,
  incl. the gap register (§8) seeded from the postmortem + audits.
  Continuation: cron `study001-flqueue-continuation-20261122`
  watches for the final eval batch and performs re-sort #1.
  This document is committed to `ivelin/zk0` @
  `study001/fl-substrate-20261121` in the same step that creates it.
- **2026-11-22 16:17 CT — Item added: Q-STEPS (FL +20% steps
  continuation).** Founder asked whether FL had converged or was
  still learning at round 30: median client policy loss was still
  falling (0.246 @r20 → 0.222 @r30), so a 6-round continuation from
  the round-30 checkpoint (36 rounds, 18,000 steps, via the D-11
  resume mechanism) is registered as a conditional item.
  Provisional score **19.5** (Impact 3 × P .50 ÷ 7.7 effort =
  3.7 DGX-h + 4 eng-h), explicitly conditional on its trigger: it
  activates only if tonight's judged diagnostic shows completion
  still climbing r20→r30; flat judged scores demote it to low
  priority. The budget asymmetry (18,000 vs C-exp's 15,000 steps)
  is labeled in the item — a practical "more time" test, not an
  attribution comparison. The eye_drops client's divergence (loss
  rising from round 21) is carried as a named watch item.
- **2026-11-22 16:37 CT — Item added: Q-LR (loss-slope-adaptive
  learning rate).** Founder suggestion: if the learning curve is
  flattening, dynamically adjust the AdamW learning rate as a
  function of the steepness of the learning curve between rounds
  rather than holding the round-indexed schedule fixed.
  Registered as a conditional, second-order item. Signal is the
  per-round median client policy loss (the series we have:
  0.391 → 0.322 → 0.246 → 0.222), slope over a smoothing window —
  explicitly NOT the server composite eval loss, whose per-round
  value is broken (rounds-10/20 prep gap G1: every round's
  metrics.json carries round 0's 0.5773 verbatim; fix owned by
  S0). The exact rule (window, threshold, multiplier, LR
  floor/ceiling) must be pre-registered in the item's run
  manifest before launch and is gate-checked — a data-dependent
  LR makes the config partly endogenous, and the frozen rule is
  what keeps the test honest and reproducible. Provisional score
  **4.3** (Impact 3 × P .35 ÷ 24.2 effort = 12.2 DGX-h + 12
  eng-h), conditional on its position: behind tonight's
  diagnostic and the first structural readout (Q-A/Q-B/Q-STEPS).
  Interaction noted: with per-round AdamW reset the round-level
  LR is a clean cross-round lever; if Q-A (carry) becomes the
  recipe, Q-LR's design is revisited — tested separately, never
  combined.

---

*Sources: certified Study 001 record (eval + verifications,
2026-11-22, this folder); `fl-grid-search-design-20261122.md`;
`fl-rerun-sweep-20261122.md` + `fl-rerun-study-manifest-20261122.json`;
`study001-config-postmortem-20261122.md`;
`zk0-architecture-recipe-backlog-20261122/note.md` +
`efficiency-survey.md`; gatefix re-audit 2026-11-22; MEMORY.md queue
entries (12:06, 15:42, 15:43 CT).*
