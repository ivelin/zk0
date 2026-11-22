# Study 001 — FL execution record, 2026-11-21

**What this document is.** The complete public record of one day of
federated-learning (FL) execution work on the zk0 project: three
training-machine failures, a certified root cause, a controlled
reproduction, a fix-verification soak, and the re-launch of Study
001's federated arm on a corrected execution substrate. It is written
to be self-contained: a contributor with a DGX-class machine, this
repository, and the branch `study001/fl-substrate-20261121` should be
able to re-run every stage described here.

**Status at writing (2026-11-21, ~19:45 CT).** The re-executed FL run
is in flight: **13 of 30 rounds checkpointed, round 14 training**,
launched 2026-11-21 14:56:41 CT. Training losses and stability are
established; whether FL *works* — the study's actual question — is
decided by the evaluation stage, which has not run yet (§10).

**Environment.**

| Item | Value |
|---|---|
| Machine | NVIDIA DGX Spark (GB10, 128 GB unified memory) |
| Kernel / driver | 6.14.0-1015-nvidia / 580.126.09 |
| Flower | flwr 1.23.0 |
| LeRobot | 0.7.0 |
| Base model | `lerobot/smolvla_base`, pinned snapshot `5e8d12a6e2975b0e5fce7c8caf47c371d257b6` |
| Study branch | `study001/fl-substrate-20261121` |
| Shipped FL checkpoint | `ivelin/zk0-smolvla-fl` (see §8 for the revision to pin) |

All GPU jobs on the shared machine ran under a job-guard wrapper:
a pre-launch whole-box utilization check (never above 70% across all
tenants), resource sampling every 15 s into the run directory, and an
automatic pause (SIGSTOP) of our job when the rolling 5-minute GPU
average crosses the 70% ceiling, resuming (SIGCONT) when it clears.

---

## 1. Three machine deaths and the certified root cause

The DGX Spark hard-powered-off three times, each time with no usable
log trail from inside the OS (consistent with an electrical /
embedded-controller cut, which is inherently unloggable from Linux):

| # | When (CT) | Circumstances |
|---|---|---|
| 1 | Wed 2026-11-18 ~09:59 | Under a full fine-tune (FT-base). The box had run ~6.8 h at GPU 92–96% earlier — a ceiling violation — but was near-idle at the moment of death. Causation for death #1 alone was never established. |
| 2 | Sat 2026-11-21 01:40 | ~21 min into Study 001's original FL run. |
| 3 | Sat 2026-11-21 08:05 | ~26 min into the resumed FL run. |

Single-process training on the same box (five specialists plus the
centralized arm, ~25 h cumulative) never killed it. Only the FL runs
did, twice in a row, at 21 and 26 minutes.

**Certified root cause.** Study 001's original FL execution used a
hand-rolled in-process driver (`fl_driver.py`): one Python process,
no Flower runtime. Its fit loop *was* serial — but all five clients
were constructed up front and stayed resident for the entire run:
five full SmolVLA policies, five AdamW optimizers, and five dataset
handles co-resident in a single CUDA context, with an in-process data
path (per-item video decode inside the training process). Serial
training, simultaneous residency.

An independent verifier, working from the raw guard telemetry CSVs
(not from the investigators' summaries), certified the claim on
2026-11-21. Three decisive signatures separate the fatal profile from
the serialized Flower profile:

1. **RAM hold vs. release.** Serialized Flower runs release memory at
   every client handover — single-sample drops of 3,364 MB (Stage A)
   and 7,939 MB (Stage B). The in-process miniature (Stage C) and both
   fatal runs show **zero drops > 2 GB across 172 combined samples**;
   the fatal runs released 57–60 MB from peak to end (≈ 0%). This is
   architectural, and it is scale-identical between the miniature and
   the fatal runs.
2. **Starved-GPU samples.** Samples with GPU utilization ≥ 90% while
   drawing < 40 W: fatal runs 8/97 samples, Stage C 2/15, serialized
   stages **0**.
3. **Temperature ratchet.** Stage C climbed 56 → 84 °C during fits at
   7.4 °C/min and was still climbing when killed. Serialized stages
   peaked at 66 °C (A) and 68 °C (B) and shed 15–17 °C within one
   sampling interval of a burst ending.

Two candidate explanations were tested and excluded:

- **fp32 precision does not contribute** (Stage B, §2): forcing the
  serialized substrate to fp32 raised every level (peak burst 92.7 W,
  RAM peak 54.1 GB, peak temp 68 °C) and changed nothing's character —
  RAM still stepped and released, clocks stayed steady.
- **"Roaming SM clocks" was struck** as a signature: every loaded run
  roams; it discriminates nothing.

A documented GB10 field-defect class (units hard-powering-off under
sustained GPU load, with a GPU clock cap as the field-tested
mitigation) is consistent with deaths #2/#3 as the *proximate*
hardware mechanism; the *controllable* cause — the thing our code
chose — is the execution model above. The fix is therefore the
execution model, verified in §3.

## 2. The reproduction: baseline first, one variable at a time

Method, fixed before any run: reproduce the founder's shipped
baseline first at miniature scale; vary exactly one row at a time;
write each stage's expectation *before* launch; hard stops registered
in advance (GPU temperature > 90 °C sustained, power-band anomaly,
any box/SSH instability → kill our job only, preserve telemetry). The
baseline is the repository's own shipped FL setup:
`train-fl-simulation.sh` →

```
flwr run . local-simulation-serialized-gpu
```

Flower simulation (SuperLink + Ray), federation
`local-simulation-serialized-gpu` in `pyproject.toml`:
`num-supernodes = 12`, `num-gpus = 1.0`,
`options.backend.max-parallelism = 1` — clients are constructed fresh
per execution in separate actor processes; one model on the GPU at a
time. The shipped run on this substrate: 1 round, 2 local epochs
(a *step* count — 2 optimizer steps per fit over a cycled loader),
FedProx μ = 0.01, batch 16, trainable-params-only exchange.

Three adaptations to the baseline copy were needed for library-era
reasons only; each was founder-approved and none touches the
execution model:

- **#1 — tensor-count gate 506 → 500.** The code's exchange gate
  expected 506 arrays; today's `lerobot/smolvla_base` has exactly 500
  tensors. The trainable scope within them is **155 tensors /
  99,880,992 parameters** (the action-expert subset; the VLM backbone
  stays frozen).
- **#2 — dead `WandBConfig` kwarg dropped.** LeRobot 0.7.0 removed
  the WandB config class the baseline passed (`enable=False`).
- **#3 — SmolVLA preprocessor in fit/evaluate.** LeRobot 0.7.0's
  SmolVLA requires its processor pipeline to prepare batches; the
  baseline loop never invoked it (fits failed with
  `KeyError: 'observation.language.tokens'`). The copy runs batches
  through the canonical preprocessor — batch preparation only.

Datasets for the miniature stages were converted LeRobot v2.1 → v3.0
with the official converter.

**Stage A — serialized baseline, native precision (bf16 mix).**
Expectation: benign, load-following profile. Result: the round
completed with real training — client losses **0.7483 / 0.5986**, zero
skipped steps, real server evaluations (composite 0.8035 / 1.2440),
aggregation over 2 valid fits. Power followed load (idle ≈ 5 W, one
bounded burst to ~74 W at 96% util); host RAM stepped to a 49.6 GB
peak and **released**; SM clocks steady 2,262–2,444 MHz; temperatures
45–66 °C. The anti-fatal profile.

**Stage B — Stage A + fp32** (via `policy.float()`, the retired
driver's own operation). The round completed (losses 0.8171 / 0.5996,
zero skipped). fp32 raised levels and changed no character. **Verdict:
precision cleared.**

**Stage C — the study's unmodified in-process driver** (sha256
`0eccaaf8…`), miniature: 2 nodes, 1 round, 60 steps/node, batch 32.
It **reproduced the fatal profile in ~4 minutes**: power swung
26 ↔ 94 W sustained; temperature ratcheted 80 → 82 → 84 °C (exactly
fatal run 1's maximum, still climbing); RAM held a 29–30 GB plateau
with **no release at the node handover**; clocks roamed. It was
killed deliberately at ~4.2 min on the pre-registered stop
conditions; the box was verified healthy and idle afterwards. One
divergence recorded honestly: the fatal runs' 0%-utilization stalls
appeared in Stage C as busy-but-starved waves (96% util at 26–41 W) —
the oscillation reproduced in power character, not in util-zero form.

**Stage D — ruled unnecessary.** A 5-resident-client stage would only
re-test an endpoint already in evidence: the fatal runs themselves
are that endpoint.

## 3. The soak: fix verification at study scale

The fix claim — *the serialized substrate is safe at study scale* —
was verified before any study re-execution, against a pre-registered
gate: continuous running for at least 2× the longest fatal survival
window (≥ ~52 min).

Configuration: serialized substrate, pool restored to 5 supernodes
with 5-of-5 sampled per round (`fraction-fit = 1`), baseline local
work, native precision (Stage B's fp32 variation reverted), Hub push
disabled (see §8 for why this is called out).

Result — **SOAK PASS**: **70.3 continuous minutes**, 23 rounds,
**115/115 fits**, zero failures or skipped steps, 23 local
checkpoints. Telemetry: load-following power (mean 19.7 W vs. the
fatal runs' 56.9 W); **40 handover releases > 2 GB**; temperature
maximum 73 °C, shedding between rounds; 3 isolated starved-GPU
samples in 279 (below the abort line). Gate line: Study 001's FL arm
cleared to re-execute on this substrate.

## 4. The substrate decision — and why

**Decision: the founder's serialized Flower execution model is the
substrate; the in-process driver is retired.** This was decided by
the evidence in §§1–3, not by preference: the two models were run
head-to-head at miniature scale under identical guard telemetry, the
difference was certified by an independent verifier from raw data,
and the chosen model then passed a study-scale soak at 2× the fatal
survival window.

The canonical FL cycle this implements: many supernodes may live on
one computer (the baseline ships 12; 20+ is fine). Per round, the
aggregator **samples a subset** (here 5 of 5), then takes turns —
selects one supernode, waits for it to finish local training,
receives its weights, picks the next — until all sampled supernodes
return. Then it aggregates, records metrics, and starts the next
round with a fresh sample. **One supernode trains at a time, always.**

Standing rule, from the founder, 2026-11-21: *run one aggregator and
iterate client nodes serially with the correct Flower setup; never
run multiple client nodes training simultaneously on one GPU.* Node
**count** is not the problem; **concurrency and co-residency** are.

## 5. Study 001: how "does FL work?" gets an objective answer

Study 001 (BENCH-003) compares training regimes for one shared
SmolVLA policy on five manipulation tasks, starting from
`lerobot/smolvla_base`. The design was preregistered and sealed
before training; the success criteria are a separate frozen document
(criteria SHA-256 `f6bee175…`; preregistration as amended by D-11,
SHA-256 `4c8c9ff3…`).

**Data.** Five tasks — `block_stack`, `eye_drops_to_basket`,
`ring_insert`, `tool_insert`, `cable_clip` — each with 50
teleoperated demonstrations (250 episodes total; per-task frame
counts 12,975 / 9,127 / 11,679 / 20,806 / 27,647), recorded on an
SO-101 arm at 20 fps with front/top/wrist cameras. The partition is
deliberately maximally non-IID: FL node *i* holds only task *i*'s
demos. Three further held-out tasks exist for a separate confirmation
benchmark and are never trained on here.

**Shared recipe.** AdamW lr 1e-4, β (0.9, 0.95), weight decay 1e-10,
gradient clip 10; cosine decay with warmup 100; batch 32; action
chunk 50; seed 1000. The unit of compute is K = 3,000 steps per task;
every arm spends the same total budget of 15,000 steps.

**Arms.**

| Arm | What it is | Status (§10) |
|---|---|---|
| Base floor | The untrained base checkpoint, evaluated first (Decision 61: the baseline evaluation runs *before* any training verdict is read, on the exact starting checkpoint) | Eval pending |
| Specialists (5) | One full fine-tune per task, 3,000 steps on that task's demos only — the per-task reference ceilings | Trained |
| C-exp (centralized) | One run on the pooled union-250 data, 15,000 steps, trainable scope = the same certified 155-tensor action-expert subset FL trains (deviation D-10 dropped the full-FT centralized arm before it ran, so the centralized arm differs from FL *only* in federation) | Trained |
| FL (federated) | FedProx μ = 0.1 (constant), 30 rounds × 100 local steps/node, dynamic aggregation weighting `eff = n_examples × clip(median/loss, 0.5, 2.0)` | In flight |
| Sequential chain | Tasks added one at a time in the order block_stack → eye_drops_to_basket → ring_insert → tool_insert → cable_clip, 3,000 expert-only steps each from the previous checkpoint; measures backward transfer / forgetting (a judged drop > 5 pp on a learned task = a forgetting event) | Not yet run |

**The read.** The federation tax **Φ = C-exp − FL** is the
attribution read: same scope, same data, same normalization (both
arms use the pooled union-250 preprocessor statistics, deviation
D-9); only federation differs. **Verdict bar:** FL parity within
5 percentage points of C-exp, and no per-task regression > 5 pp.
Evaluation is on the real arm, with per-trial judged completion
against written per-task completion definitions — never from
training success flags — and both shared models are reported with a
full 5-task retention matrix.

Why D-9 matters, in one paragraph: with the base checkpoint's
shipped preprocessor statistics, these datasets' actions/state are
essentially unnormalized (normalized action std 51.5; pilot losses
~1,800–2,100). Under pooled dataset statistics the identical policy
and batch give loss 0.835 (std 1.19), in line with the other arms.
Per-node statistics were rejected: nodes would train in different
normalization frames and the aggregated weights would not be
commensurable.

## 6. Deviation D-11 and the six founder decisions

Deviation D-11 (registered in the preregistration's deviation log
before the re-execution) moves the FL arm from the retired in-process
driver to the serialized Flower substrate — the substrate the
preregistration had specified all along ("Flower/FedProx over 5
nodes"); the driver was the unregistered departure. No registered
design quantity changed: 30 rounds × 100 steps/node @ batch 32,
by-task partition, 155-tensor scope, μ = 0.1, dynamic weighting,
pooled preprocessor, `lr_at` schedule, seed 1000, pinned base
snapshot, final-round checkpoint evaluated, no Hub push.

Six founder decisions, all taken 2026-11-21 13:48–13:51 CT:

1. **D-11 + adaptations #4–#10 approved** (the gap closures below).
2. **Native precision mix** (the base policy's bf16/fp32 mix, as
   soak-cleared) over an fp32-everywhere variation.
3. **Per-round AdamW reset.** Substrate clients are stateless; each
   fit builds a fresh AdamW. Supporting finding from a record search
   the same day: **no controlled carry-vs-reset experiment exists in
   this project's history.** Reset is the entire production history —
   a fresh optimizer inside every fit since 2025-10-11, plus an
   explicit per-round LR-scheduler reset. Carry existed only in a
   vendored September-2025 quickstart that was never run. Decision 3
   therefore matches all published FL evidence from this project;
   carry is the untested variant.
4. **Model-only checkpoints + a kill-and-resume drill as a hard
   pre-launch gate** (a registered realization in place of the
   full-state letter of the project's checkpoint standard; round
   cadence ≤ ~30 min of compute is satisfied).
5. **5/5-fits round acceptance** — a round aggregating fewer than 5
   valid fits voids the run — **and server-side evaluation off**
   during the run.
6. **The ~10–12 h machine window**: build, drill, then launch.

**The eight gaps (G-1…G-8)** between the stock substrate and the
registered design, and how each was closed (per-adaptation detail:
`STUDY001_ADAPTATIONS.md` on this branch):

| Gap | Closure |
|---|---|
| G-1 Dataset wiring: stock loader takes a bare hub repo id per client — no local `root`, no episode restriction | Client config carries `root` + a P0 episode list; all 5 clients verified loading exactly their 50-episode subsets |
| G-2 FedProx μ: stock `compute_fedprox_parameters` decays μ (halving every 10 rounds, or an eval-trend factor) — never the registered constant 0.1 | `fixed_fedprox` strategy path returning the constant μ |
| G-3 Dynamic weighting: stock substrate has plain FedAvg only | Strategy-level `eff = round(n_examples × clip(median/policy_loss, 0.5, 2.0))`; factors logged per round; unit-checked against the retired driver's `aggregate_round` to < 6e-8 |
| G-4 LR schedule: stock is a per-fit cosine reset that discards the registered per-round LR | Per-round `initial_lr = lr_at(mid-round offset)` (warmup 100 → 1e-4, cosine → 2.5e-6 at the 3,000-step per-node budget), flat within the round |
| G-5 Seeding: none in the substrate | Per-client seed 1000 + client_id |
| G-6 Preprocessor: adaptation #3 builds *per-client* statistics — exactly what D-9 forbids | Pooled preprocessor built once pre-launch by the driver's own construction, serialized, loaded by every client; pre-launch hard gate on the stats hash `a0455b5b…` — reproduced through the real client path |
| G-7 Checkpoints/resume: substrate saves model-only per round and has no resume path | Resume wiring (`initial_model_path` + round offset) proven by the kill-and-resume drill (§7) |
| G-8 Packaging: substrate copies the *base* preprocessor into checkpoints | Post-run step writing the hash-verified pooled preprocessor (plus tokenizer artifacts and postprocessor state, which substrate checkpoints also lack) into `checkpoint_round_30` so it loads standalone |

Two realization findings worth recording for anyone repeating this:
(i) LeRobot's pipeline save/load degrades preprocessor statistics to
float32 (max diff 1.8e-6) and can never reproduce the D-9 hash, so the
pooled statistics are serialized bit-exactly (float64 safetensors)
and the pipeline is reconstructed in-process per client; (ii) the
substrate names image features after the dataset, so study-mode
clients build the policy from the base checkpoint exactly as the
retired driver did, with the camera rename applied in a study-gated
DataLoader collate.

Verified at build time: trained scope set-identical to the certified
155-tensor census (the substrate exchanges the full 500 tensors on
the wire, but frozen tensors are bit-identical across clients and
average to themselves — semantically equal to trainable-only
exchange); `hf_repo_id` absent from the study configuration (§8).

## 7. Build verification and launch

**Smoke — PASS.** 2 rounds × 5 clients × 2 steps in a miniature copy
under the guard: `steps_completed = 2` per fit, finite losses,
dynamic-weight aggregation records arithmetically correct for both
rounds, checkpoints saved, packaging dry path passed.

**Kill-and-resume drill — PASS on attempt 2 (hard gate).** Attempt 1
was invalidated by operator errors, recorded here because they are
instructive: the kill signal hit only the CLI parent, so the
simulation child finished round 2 — which then aggregated only 3 of
5 fits after the Ray actors were killed, a live demonstration of
exactly the short-round failure mode the 5/5 acceptance rule exists
to void. Attempt 2: segment 1 was SIGKILLed (whole process tree)
mid-round-2; frozen state = `checkpoint_round_1` + round-1
aggregation only. Segment 2 resumed from `checkpoint_round_1` and
produced **rounds 2 and 3 under their registered numbers**, exit 0.
Continuity {1} ∪ {2,3}, 5 clients per round, and the weight chain
base → ckpt1 → ckpt2 verified distinct and chained.

**Launch.** The full 30-round run launched **2026-11-21 14:56:41 CT**
under the job guard (whole-box 70% ceiling enforced mechanically).
Round 1: 5/5 fits × 100 real steps; per-node policy losses after 100
steps — tool_insert **0.316**, ring_insert **0.323**, cable_clip
**0.391**, block_stack **0.445**, eye_drops **0.491** (median 0.391);
dynamic-weight factors 0.795–1.235; round LR confirmed at the
registered mid-round value `lr_at(50) = 5e-05`; ~8.5 GB RAM release
observed at a client handover.

One operational note, because it will look alarming in the logs: at
15:31 CT the guard paused the run for **91 seconds** when the rolling
5-minute GPU average crossed the 70% ceiling (71%), then resumed it.
A STOP/CONT duty-cycle pause at the ceiling is the guard working as
designed — normal operation, not a failure. (An automated monitor
initially misclassified any pause as a failure; its rules were
corrected the same afternoon to report only real failure, a pause
stuck > 10 min, sustained ≥ 85 °C, or completion.) The run does run
hotter than the soak predicted — peaks touching 85 °C — because
100-step fits are sustained load rather than bursts; the character
remains the benign one (temperatures shed 10–14 °C between bursts,
RAM releases at handovers), and guard duty-cycling doubles as a
thermal limiter.

**Status at writing:** 13/30 rounds checkpointed, round 14 in
progress, pace ≈ 22 min/round including startup and guard pauses.

## 8. The Hugging Face incident and restore

During soak preparation it emerged that the substrate's
`save_and_push_model` reads its checkpoint gating from `pyproject`
(save interval 1, rounds 20) — **not** from `--run-config`
overrides. As a result, Stage A and Stage B each saved a checkpoint
and **pushed it to the live repository `ivelin/zk0-smolvla-fl`**:
two commits carrying 1-step miniature weights (the second a 1.8 GB
fp32 miniature), replacing the curated model card and reverting the
preprocessor to legacy feature names. The stage records' "no
checkpoint / no Hub push" claims were wrong; that correction is part
of this record.

The shipped revision was never lost — it remained fully addressable:
`f062dedbfe7801dde10a56a9a2d8bb21e52acaff` (2026-11-17; shipped
weights 1,197,789,256 B, LFS sha256
`693dfdeb68e322ac194d28923ccb9e971e23b38c07cec743705a3b09fbcc7ec4`;
1 round, FedProx μ = 0.01, 12 datasets, eval loss 0.15415).

**Restore, executed 2026-11-21 13:52 CT** with the founder's
approval: a new commit on top of the accidental history (history
preserved, both accidental pushes still visible) — new HEAD
**`b400eea1cded125884ff7c5c6ae03b356822180e`**. Verified via the Hub
API: the tree is identical to `f062ded` across all 9 files — weights,
README, metrics, config, and preprocessor byte-identical.

Standing rules from this incident:

- **Pin the revision.** Anyone pulling the FL checkpoint uses
  revision `b400eea` (or `f062ded`) explicitly — never unpinned HEAD.
- **Prevention:** `hf_repo_id` stays unset / Hub push disabled by
  default in this codebase; the study configuration was verified to
  have it absent before launch.

## 9. Guard circuit-breaker extension — built, validated, not deployed

As a second line of defense behind the execution-model fix, a
circuit-breaker extension to the job guard was built and validated
offline on 2026-11-21. It watches the guard's own 15 s telemetry for
the three certified fatal signatures and kills *our* job (never the
machine, never anyone else's work) if one fires:

- **T1 — temperature:** ≥ 85 °C once, or ≥ 3 of the last 12 samples
  ≥ 80 °C with the current sample ≥ 80 °C. (A climb-rate rule was
  tested and rejected: even the benign soak spikes ~20 °C/min over
  60 s.)
- **T2 — RAM hold:** whole-box memory used ≥ 20 GB for 32 consecutive
  samples (8 min) with no ≥ 2 GB release, our job holding the GPU in
  ≥ half the window, **and** our process group's RSS ≥ 10 GB —
  whole-box RAM alone never fires it.
- **T3 — starved GPU:** ≥ 3 starved samples (util ≥ 90% @ < 40 W)
  **and** ≥ 12 power swings > 30 W within the last 40 samples.

Replay validation against recorded telemetry: it **fires on fatal
run 1 at 1m31s** (T1; T2 at 8m19s and T3 at 9m04s would fire
independently), **fatal run 2 at 8m18s**, and **Stage C at 3m02s** —
earlier than the human kill. It **never fires** on Stage A, Stage B,
or the soak (279 samples). Honest limits: T2/T3 do not fire on Stage
C — it lived only 17 samples (T2 needs a 32-sample hold) and
contained exactly 2 starved samples (T3's floor of 3 is forced by
the soak's benign maximum of 2).

Deployment is deliberately pending a founder decision, in steps:
replay the production run's full study-scale telemetry through it
first (the soak's 2-step fits do not settle T1/T2 against 100-step
fits), then a log-only period, then active. Any automation around it
must treat a breaker event as "the breaker killed our job to protect
the box" — never as a cue to auto-relaunch.

## 10. Experiment inventory — what exists, what is left

**Trained and verified:**

| Run | Budget | Result |
|---|---|---|
| FT-base re-run (full fine-tune of `smolvla_base`) | 10,000 steps | Final loss 0.047; post-run verified (500/500 tensors, finite forward pass, action shape [1, 50, 6]) |
| FT-zk0 (full fine-tune of `zk0-smolvla-fl`) | 10,000 steps | Final loss 0.052; same verification |
| Specialists × 5 (full FT, per task) | 3,000 steps each | All rc = 0, 2026-11-20 |
| C-exp (centralized, expert-only scope) | 15,000 steps | Completed 2026-11-21 01:19 CT, rc = 0 |

**In flight:** the FL re-execution (§7) — 13/30 rounds at writing.

**Not yet run:** the sequential chain (5 × 3,000 expert-only steps),
then the **evaluation stage**: base floor first (Decision 61), then
judged real-arm evaluation of every arm, the retention matrix, and
the Φ read.

That ordering is the point of this section: **everything left to
establish whether FL works is evaluation, not more training.**
Training losses in this record (0.316–0.491 after round 1, and the
miniature-stage losses) show the machinery trains; they are not
capability evidence. The verdict comes from judged task completion:
FL within 5 pp of C-exp with no per-task regression > 5 pp, or it
doesn't.

One open founder decision is attached to the inventory: the
specialists trained full-FT while FL and C-exp train the expert-only
scope. The recommendation on record is to **accept the specialists
as-run** — Φ never touches them, they serve as per-task reference
ceilings (where full FT is the natural definition), and re-running
all five expert-only would cost ~15 h of machine time for a
comparison the primary read does not use. The scope difference is
registered either way; the decision was still open at writing.

## 11. Reproducing this work

1. Check out branch `study001/fl-substrate-20261121`. The substrate
   adaptations for the study build are documented in
   `STUDY001_ADAPTATIONS.md` (same directory as this file); the
   baseline entry point is `train-fl-simulation.sh`.
2. Reproduction stages (§2): copy the baseline tree; apply
   adaptations #1–#3; run Stage A as-is (expect client losses in the
   0.6–0.75 range at miniature scale and visible RAM releases at
   handovers); Stage B = A + `policy.float()`; Stage C requires the
   retired in-process driver design described in §1 — reconstruct it
   only behind the hard stops in §2, at miniature scale, and expect
   to kill it within minutes.
3. Soak (§3): 5 supernodes, 5-of-5 sampling, baseline local work,
   native precision, Hub push disabled; gate = ≥ 52 continuous
   minutes with handover releases present and temperature shedding
   between rounds.
4. Study run (§§5–7): pin the base snapshot
   `5e8d12a6e2975b0e5fce7c8caf47c371d257b6`; verify the trained-scope
   census (155 / 99,880,992 of 500 tensors); verify the pooled
   preprocessor stats hash `a0455b5b…` through the real client path;
   pass the miniature smoke and the kill-and-resume drill before the
   full launch; launch under a whole-box utilization guard.
5. Any use of the shipped FL checkpoint: pin revision `b400eea` or
   `f062ded` (§8).

*Record written 2026-11-21. Numbers are as measured and verified on
that date; the FL run's final state and the evaluation verdict will
be appended as a follow-up record when they exist.*

---

## SIDE NOTE — for readers on DGX Spark / GB10-class hardware: the freezing problem and what mediated it

> This section is a hardware field note, separate from the study
> record above. It documents a failure pattern we hit three times on
> a DGX Spark (GB10, 128 GB unified memory), what our forensics
> established, and the mediation that worked — in case it saves
> another operator a frozen box.

**The pattern.** Under sustained FL training load, the machine
hard-froze / powered off three times:

- **2026-11-18** — during a full fine-tune.
- **2026-11-21 ~01:41 CT** — ~21 minutes into an FL run.
- **2026-11-21 ~08:05 CT** — ~26 minutes into the resumed FL run.

Each time the box was dead until a **manual power-cycle**. Each time
there was **no useful in-OS log of the cut** — nothing in the journal
explains the stop, which is the signature of a cut that happens
below the OS (embedded-controller / power level), faster than Linux
can record anything. Single-process training on the same box ran
~25 cumulative hours without an incident; the FL execution model
below killed it twice in one morning.

**What the forensics established.** The workload that froze the box
had a distinctive telemetry profile, certified by an independent
re-derivation from raw 15-second guard samples (§1):

- unified RAM climbed and then **never released at client
  handovers** (zero releases > 2 GB, vs. 3.4–7.9 GB releases in the
  benign profile);
- **starved-GPU samples** — utilization ≥ 90% while drawing < 40 W;
- a **temperature ratchet** to ~84–85 °C that did not shed between
  bursts.

The serialized execution model showed none of this: memory released
at every handover, power followed load, and temperature shed 10–17 °C
within one sampling interval of a burst ending. The fatal ingredient
was not the model, the data, or the precision (fp32 was tested and
cleared) — it was an execution model that kept five full training
clients resident in one process and one CUDA context for the whole
run, so the machine never got a release point.

**The mediation that worked.**

1. **Serialized execution.** One aggregator; per round it samples a
   subset of supernodes and trains them **one at a time**, each
   client constructed fresh in its own process; memory is released
   at every handover. Node count was never the problem — concurrency
   and co-residency were.
2. **A whole-box utilization ceiling, mechanically enforced.** Every
   job runs under a guard that samples the whole box every 15 s and
   pauses *our* job when the rolling GPU average crosses 70%,
   resuming when it clears. In practice the duty-cycle pauses double
   as a **thermal limiter**: the production run's temperature peaks
   shed between bursts instead of ratcheting.
3. **Prove stability before the long run.** A 70.3-minute soak at
   full study scale (2× the longest fatal survival) with 115/115
   fits and a 73 °C maximum preceded the production launch.

**Honest caveats.** A hard power-off under sustained GPU load is a
**documented field issue for GB10-class units** in the community,
independent of our workload; a **GPU clock cap** is a known field
mitigation for it (we have not applied one — it needs a privileged
path we chose not to improvise). The workload profile above was the
controllable trigger in our case, and fixing it was sufficient for a
stable study-scale run — that is not a claim that the hardware
behavior does not exist. Separately, our guard **circuit-breaker
extension** (§9), which kills our own job on the three fatal
signatures, is built and replay-validated but **not yet deployed**.
No blame attaches anywhere in this note: the failure was a
combination of a young platform, a documented hardware behavior, and
an execution model we wrote — and the execution model was the part
that was ours to fix.
