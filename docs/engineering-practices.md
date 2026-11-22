# zk0 Engineering Practices — Single Source of Truth

This file is the canonical record of **how zk0 work is done**: the
architecture invariants, the experiment methodology, and the operating
procedures every contributor — human or agent — follows. System
architecture in depth lives in [ARCHITECTURE.md](ARCHITECTURE.md); this
file governs practice. Study records (preregistrations, execution
records, postmortems, sweeps) live alongside the code on this branch and
are indexed in §11.

**Status:** canonical on the working branch; flows working → staging →
main like every other change (§5). Reconciled 2026-11-22 from the
project's decision record, its release/verification SOP, and the
Study 001 record. Where an older document disagrees with this file, the
newest founder decision governs and the resolution is recorded here.

---

## 1. Architecture invariants

These do not move. A design that violates one is not a zk0 design.

- **One shared model.** No per-node personalization. Variation is
  handled by per-arm calibration/canonicalization and history-based
  adaptation, never by forking the model per operator.
- **Data stays local; only model updates move.** Training data never
  leaves the operator's machine.
- **Federation scope (certified configuration):** the VLM backbone is
  frozen; only the action expert is federated.
- **Dual-mode model.** A parallel language/reasoning head is active in
  training and evaluation and bypassed at real-time execution; actions
  never causally depend on reasoning tokens.
- **Temporal input.** The model consumes a few seconds of recent
  footage, not a static snapshot.
- **Size cap.** HW3-class across Mac, Linux, and Windows: target 1B to
  a few B parameters, maximum ~10B.
- **Sim-first, one swappable sim/real contract.** Engine roles are
  asymmetric and evidence-based: Isaac Sim is the lab tier; Drake is
  the eval gate; MuJoCo is the eval cross-check and the near-term
  contributor grasp engine; Genesis is the contributor front door on
  probation (re-tested per release; never an eval gate until it passes
  at true scale). Game engines are not an eval path.
- **Telemetry contract.** Train only on signals the real arm reports;
  vision is inside the contract under the same firewall as every other
  signal.
- **Provenance marks never blur:** *sim-trained* · *sim-validated* ·
  *real-validated* — three distinct marks, in data, UI, and claims.
- **Safety imports:** a non-overridable hard safety layer; shadow mode
  before promotion; trigger-based capture; every verified failure
  becomes a permanent regression task; canary rollout with rollback.

## 2. Comparative-experiment methodology — apples to apples

The comparison is the product. A result whose arms differed in anything
but the variable under study is not a result.

- **Pre-flight equivalence (founder rule, 2026-11-22).** Before ANY
  comparative training run (fine-tuned specialist vs centralized vs
  federated pooling), compare ALL important parameters, configuration
  settings, and hyperparameters across every arm in advance: input
  features/cameras, preprocessor and its statistics hash, trained-
  tensor census, data episode lists, optimizer and schedule semantics,
  and the full budget accounting — **steps × batch × views = total
  images seen**. All else equal, so the comparison isolates the one
  variable under study. The only permitted differences are that
  variable plus registered waivers. **No launch on an unexplained
  difference.** Cheap checks on miniature runs first — never
  discoveries after expensive ones.
- **Registered design first.** Every study has a preregistration and a
  machine-readable study manifest, human-approved before launch.
  Deviations are appended as new registered rows; the frozen original
  is never edited.
- **Base-floor first.** The untrained base model's baseline evaluation
  runs BEFORE any training starts, on the exact starting checkpoint.
- **Config-equivalence gate (mechanical).** Resolved configurations
  are dumped from the actual constructed objects — for federated runs
  that means the clients, the server template model, AND the saved
  miniature checkpoint's own declared config — and diffed against the
  manifest. An unexplained difference fails the gate, and the launch
  wrapper refuses to launch a study run without a passing gate report
  covering the current code/config tree. (Study 001's camera miss —
  §11 — is the incident this gate exists to prevent: training used
  three cameras while the server template, built from a leftover
  two-camera dataset entry, made every saved checkpoint declare two.)
- **Full-path smoke before long runs.** Before any run expected to
  exceed ~30 minutes, a miniature end-to-end run exercises EVERY
  production code path — training, checkpoint save, reload, hashing,
  signing, upload — and the artifacts are verified to exist and load.
  A pre-flight that disables a destructive path is not a pre-flight.
- **Evaluation integrity.** The evaluator asserts identical declared
  input features across all checkpoints before judging; a measurement
  taken through unequal inputs is labeled as such wherever it appears.
  Offline or judged metrics are named for what they are, never
  presented as physical or rollout success.

## 3. Federated execution on one machine

- **One aggregator, client nodes iterated SERIALLY** — exactly the
  serialized Flower setup that shipped the federated checkpoint. The
  aggregator samples a subset of supernodes per round, then takes
  turns: select one supernode, wait for its local training to finish,
  receive its weights, pick the next — then aggregate, evaluate,
  record metrics, and start the next round. One supernode trains at a
  time, always.
- **Node count is not the problem; concurrency is.** Five, twenty, or
  more supernodes may live on one computer. Multiple client nodes
  training simultaneously on one GPU, and in-process resident drivers
  that hold all clients in one process, are prohibited — that
  execution profile is the certified root cause of two machine deaths
  (Study 001 record, §11).
- **Soak before scale.** A new substrate or recipe proves itself in a
  soak run at study scale before a production run is scheduled.

## 4. Shared-machine protection

Training machines are shared with teammates and other agents. Play
nice is mechanical, not aspirational.

- **70% ceiling.** Whole-box utilization across ALL tenants — compute,
  GPU, memory, disk — never exceeds 70%, enforced mechanically: a
  pre-launch whole-box check plus an in-run monitor that pauses our
  job at the ceiling.
- **Job guard, single launch path.** Every job launches with explicit
  CPU/GPU/memory caps through the guarded launch wrapper: pre-launch
  ceiling check, continuous whole-box telemetry sampled into the run
  directory and mirrored off-box (no mirror, no launch), large GPU
  jobs under the fleet GPU lock — exactly one lock layer per launch.
- **No boot services or auto-start of any kind**, for any workload.
- **Thermal monitoring** stays armed for the whole of any
  heat-generating run; monitors and wrappers carry liveness contracts
  (heartbeats) and may stand down only on verified run completion,
  never on a failure report.
- **Circuit breaker.** A guard-side breaker watches for the certified
  fatal signatures (temperature ratchet, memory that never releases at
  client handover, starved-GPU oscillation). It currently runs in
  log-only mode; kill authority is earned per-trigger with healthy-run
  evidence, not assumed.
- **Incident posture.** Connection loss means a few bounded probes,
  then silence and preserve evidence. After any incident, the first
  pass is read-only — state, processes, previous-boot logs, autostart
  inventory — reported before anything restarts.
- **Backup discipline.** Work done on a shared machine is synced to
  the remote working branch at stage boundaries (code/config only —
  never environments, dataset caches, or checkpoints). Significant
  checkpoints are mirrored to our Hugging Face org with hashes
  recorded; no checkpoint exists on only one machine.

## 5. CI/CD — fast checks, staged promotion

- **Fast regression tier first.** A quick suite runs in minutes BEFORE
  any expensive training launches. Every verified incident becomes a
  permanent regression test in the suite; the suite grows tighter over
  time, never looser.
- **Pipeline stages, in order, each gated:** (1) fast deterministic
  tests, green in one command; (2) coverage floors, with real
  round-trip tests (save → reload → hash; sign → verify → tamper) on
  persistence, signing, aggregation, and safety paths — mocks of an
  irreversible step do not count; (3) full-path smoke (§2); (4)
  preregistered eval batteries — a halt gate firing stops the study
  and is never argued around; (5) staging verification — replica-first,
  the change rehearsed and verified exhaustively on a staging copy;
  (6) independent verification and sealing (§8); (7) gated production
  rollout with a recorded rollback point.
- **Promotion path.** Nothing reaches production directly — not a full
  training run, not a public push, not a published artifact. Code
  flows working branch → staging → main; artifacts flow staging →
  public; each promotion records its verification evidence in the run
  manifest before it proceeds.
- **Packaging standard-load test.** Every packaged checkpoint is
  loaded in a fresh process through the standard load path, exactly as
  an external consumer would load it, before it is published or
  evaluated. Packaging that only our own pipeline can read is a
  defect.
- **Publish gating by construction.** A run stage that must not
  publish (to a hub, a repo, or a site) proves it from its resolved
  configuration for EVERY stage — smoke, soak, and production alike —
  and the pre-flight gate verifies no publish occurred.
- **Local CI while hosted CI is unavailable.** When hosted CI minutes
  are exhausted, the same suite runs locally on the executing machine
  through one documented entrypoint before any PR; hosted CI is
  re-enabled the moment budget returns. Local results are reported as
  local, never described as hosted CI.

## 6. SRE practices

- **Monitoring with liveness contracts.** Every monitor, wrapper, and
  recurring job heartbeats; silence past its threshold is itself an
  alarm. A dead run pages within minutes, not at the next check-in.
- **Incident loop.** Incident → blameless postmortem → regression test
  or guard. Postmortems reconstruct timelines from logs and records,
  never from memory, and name the mechanism, not the person.
- **Postmortem cycles.** End-of-run (every run, clean or failed:
  outcome vs plan, timeline, deviations, action items with owners);
  end-of-day (decisions recorded the day they are made; incidents
  logged; nothing left only in chat); end-of-week (action-item
  burn-down, audit status, hygiene).
- **Full audit after major disappointments** — lost work or
  significant compute destroyed, a wrong claim delivered, a gate or
  preregistration failure, a user-visible shipped defect: an
  independent audit of the affected pipeline in a separate context,
  not just a fix.
- **Checkpoint + resume standard.** Long runs never depend on a final
  save: full-state checkpoints (weights + optimizer + scheduler + RNG
  + step/data position) at a cadence bounding lost work to ~30
  minutes of compute, written atomically and verified after save
  (hash + reload). Evals persist per-unit results incrementally and
  resume by skipping completed units. Resume is proven, not assumed:
  the pre-run smoke includes a kill-and-resume drill. A registered
  exception (e.g. a deterministic substrate with model-only
  checkpoints) is recorded in the study record with its reasoning;
  the drill remains a hard gate.
- **Backups are restore-tested.** A backup that has never been
  restored in a drill is a hope, not a backup.

## 7. Node release gate

External nodes are other people's machines and reputations. Node
software goes live rock solid or not at all.

- **Terms gate (hard).** External production federation stays blocked
  until contributor terms are published with a working signed-in
  human acceptance flow. A gate is an external condition, not a config
  flag.
- **Staging tests from all perspectives, before go-live:**
  - cold install from zero on every supported platform class
    (Mac / Linux / Windows, HW3 tier);
  - user personas exercised end-to-end: novice contributor,
    experienced operator, low-bandwidth / unreliable-network operator,
    operator mid-run during an aggregator restart;
  - failure injection on staging — kill mid-round, disk full, network
    drop — with recovery proven, not described;
  - canary promotion: founder nodes first, then a small external
    cohort, with rollback ready at each step.
- **Honest gate status.** Wherever a release is described, the gate's
  actual status is stated. The gate is passed only when every element
  above has evidence; until then the release is a candidate, not a
  launch.

## 8. Verification system

- **Builder ≠ verifier.** Any new claim, pull request, change, or
  experiment result is independently verified and certified by an
  independent agent in a separate context before it is delivered or
  shipped. The verifier is briefed with the raw artifacts and the
  acceptance criteria — never the builder's conclusions. When builder
  and verifier disagree, the disagreement surfaces to the founder; the
  builder does not resolve it.
- **Claim sealing.** A verified claim is sealed to its exact input set
  — assumptions, facts, files, objects, parameters, and the
  verification package itself. A sealed claim is reused without
  re-verification while its seal is intact; quoting it outside its
  sealed inputs makes it a new claim. Whether a change is material
  enough to break a seal is a recorded judgment belonging to a third
  role, the independent auditor.
- **Verify before asserting.** State-of-world claims (sent vs draft,
  paid vs unpaid, merged vs open, passed vs failed) are checked at the
  authoritative source before they are stated, and deliverables carry
  a claims-evidence line: source · field · timestamp. An unverifiable
  fact is reported as unverified, never filled in.
- **Capability claims require measurement.** "Cannot", "beyond", and
  "impossible" require a direct measurement of the limiting quantity.
  A gate failure is evidence about quality, never about capability.
- **Replica-first.** Any change that writes to production is first
  executed and verified exhaustively against a replica or staging
  copy. Testing never writes to production.

## 9. Licensing and survivable copies

- **Permissive licenses only.** Everything zk0 uses — datasets,
  checkpoints, code, dependencies — must be under a permissive
  open-source license, verified at the source before use. Anything
  else is a red flag, flagged before it is touched, never after.
- **Survivable copies by reference.** Every published experiment must
  be reproducible even if the original authors delete their repos.
  The mechanism is duplication-efficient: a fork under our Hugging
  Face org at a pinned revision (content-addressed storage dedupes
  the bytes), plus a reference file with hashes in this repo naming
  the upstream original, the fork, the exact revision, and the exact
  subset definition. This covers datasets AND checkpoints.
- **Our own licensing.** Shared checkpoints and code are published
  under Apache-2.0, inbound = outbound, commercial use included.
  Datasets we publish are under CC BY 4.0.

## 10. Documentation placement — this file's own rules

- **One canonical home per topic (DRY + MECE).** Check whether it
  exists before creating it; extend the existing home instead of
  duplicating. This file is the home for engineering practice;
  [ARCHITECTURE.md](ARCHITECTURE.md) is the home for system
  architecture; study records (§11) are the home for experiment
  evidence.
- **Public-first.** zk0 is an open-source project: methods, configs,
  results, postmortems, and engineering practices belong in this repo,
  written so outside contributors can reproduce the work entirely.
- **Private material stays private — and unpointed.** Material that is
  not publishable (personal information, secrets, non-publishable
  strategy or operations detail) is maintained privately by the
  maintainer. Public files never reference it: no links, no document
  IDs, no folder names, no summaries of private documents.
- **Conflicts resolve by newest founder decision**, and the resolution
  is recorded in the winning document. A conflict that recency cannot
  settle is surfaced to the founder, never silently picked.

## 11. Study records (index)

- **Study 001 (BENCH-003)** — the first full federated-vs-centralized
  comparison on the ArmnetBench SO-101 tasks:
  - `STUDY001_ADAPTATIONS.md` — registered adaptations to the design.
  - `STUDY001_FL_EXECUTION_RECORD_2026-11-21.md` — the full execution
    record of the federated run.
  - `study001-config-postmortem-20261122.md` — the configuration-miss
    postmortem: eight incidents, and the P1–P5 prevention gates now
    folded into §2 and §5 of this file.
  - `fl-rerun-sweep-20261122.md` — the pre-rerun config-equivalence
    sweep: camera census, the eval-side premise correction, the
    classified cross-arm diff, license census, and mirror plan.
  - `STUDY001_DATASET_REFERENCE.md` — the dataset reference:
    provenance, pinned revision, exact subset definition and hashes —
    everything an outsider needs to reconstruct the training data.
  - `fl-grid-search-design-20261122.md` — the unified FL setup search
    queue: one dynamic queue of recipe, architecture, and pipeline
    bets, best bets first, re-ranked as results land.
