---
title: "zk0 Roadmap: Pool Compute and Data for Physical AI"
description: "The zk0 community and product roadmap: prove the federated experiment first (M0 calibration protocol and the M1 four-arm test), then clusters, bonds, and institutional tiers — each phase gated by traction, with honest status on what is and is not yet measured."
---

# zk0 Roadmap: Pool Compute and Data for Physical AI

zk0 is an open-source project building one shared, generalist robot model for the stationary SO-101-class arm, trained by a community of arm owners and GPU owners whose data and compute stay distributed. This roadmap is deliberately phased: prove the core experiment first with simple, visible trust mechanisms, then add clusters, then economic bonds, then institutional controls — each phase unlocked by a traction signal, not a date. We would rather publish what we have actually measured, including what has not worked yet, than promise what federation might do in principle.

## Positioning

Compute is increasingly hard to come by. Robot data is even scarcer.

No single small lab, startup, or enthusiast can collect enough diverse, real robot trajectories — across tasks, objects, environments, materials, and slightly different arms — to train a robust generalist model alone. A few massively funded companies can try. Even they struggle with the long tail.

zk0 pools what is distributed:

- **Compute** — not just one training run. Covering the variation space takes program compute: many rounds, many tasks, many evaluations, retraining, and re-checks, accumulated over months.
- **Data** — the value of private robot episodes, trained on where they already sit, next to the compute that is already there.

The product of the pool is **one shared model**, in the SmolVLA class (450M parameters today, with room to grow within a Tesla HW3-class envelope of roughly 1–2.5B parameters), for a stationary single arm. It should work on many tasks **out of the box** — no per-task fine-tune-and-forget cycle.

That last point is a design rule, not a slogan:

- A candidate checkpoint is promoted only if it passes the accumulated task battery with **no regressions**.
- Replay mixing across prior tasks is mandatory in round training.
- Breadth is a headline metric: a tasks × sites matrix of zero-shot success, tracked per round.

A dedicated local fine-tune will still beat the shared model on any single task in the near term. Breadth is the network's claim. Depth remains available locally.

**Privacy is a bonus, and an enabling condition — not the headline.** Robotics companies are competitive, and for good reason: they cannot upload customer or deployment data, yet early customers alone cannot provide the diversity a non-collapsing model needs. Keeping data local is the mechanism that lets the most valuable data — real deployments — enter a pool at all. For the first milestone, we do not need the most privacy-sensitive operators. We need a community that wants a shared strong model more than it worries about this, gets strong results, and then branches into hierarchical tiers with stronger legal, contractual, and algorithmic privacy controls.

The honest pitch is not a slogan about data location. It is **reciprocity, with audited, value-weighted contribution accounting**: you contribute, the shared model improves on other people's held-out tasks, that improvement is measured, and your credit reflects the measured transfer — not raw episode counts.

There is no token. There is no chain. Mechanisms borrowed from public-blockchain research — commit-reveal, transparency logs, reputation, stake bonds — are used where they solve a real problem, without riding a chain.

Scope, for now, is narrow on purpose: the widest useful task set for a **stationary single arm**. Mobility and multi-arm systems come later, only after the core thesis is proven on this embodiment.

## Where we are, honestly

This section is a feature of the document, not a disclaimer buried at the end.

**What is established:**

- The federated learning loop has converged stably over 250 rounds on real, heterogeneous community datasets (2–4 clients), at the level of policy loss. In zk0's actual configuration — frozen SmolVLM-2 backbone, action-expert-only updates of roughly 100M parameters, 20 local epochs, tiny client sets — FedProx with dynamic scheduling outperformed FedAvg in our own published runs.

**What is not established:**

- **Zero rollout success rates have been published yet.** Every number in the published record so far is policy loss. Loss is not task success, and any mapping from loss to an implied success rate is unmeasured. We state this plainly.
- **No local-only baseline has been published.** The federated-vs-local comparison the thesis depends on has not been run in public, by us or, on real hardware at our scale, by anyone in the form we need.
- Earlier language about learning "without catastrophic forgetting" was overstated. The record shows bounded, stabilized forgetting: unseen-task evaluation loss rose substantially versus the pretrained baseline, and the frozen backbone explains much of the retention. Those numbers are the strongest argument for the no-regression promotion gate above.

**What the wider literature says:**

- Federation has **never beaten centralized pooling on real robots in print.**
- The only real SO-101 federation study we have (decent-vla, UCL/Cisco, September 2026) found federated training trailing centralized by 15.8 points at 2 sites and 24.5 points at 3 sites.
- The measured cause was not the aggregation algorithm. It was **per-arm calibration heterogeneity** — wrist-roll distributions more than 100 degrees apart across arms, non-overlapping. Different arms, mechanically, are different robots until calibration and action spaces are standardized.
- The same literature is split on aggregators, and the split is configuration-dependent: FedAvg beat FedProx in decent-vla and ForgeVLA under their configurations; FedProx with dynamic scheduling beat FedAvg in zk0's configuration. We do not resolve that by argument. We measure it, in our configuration — see M0/M1 below.
- LoRA-only federation cost SmolVLA about 20 points in decent-vla. We do not federate LoRA adapters alone for this model.

If a checkpoint, a claim, or a phase in this roadmap is not backed by a measured rollout result, we will say so.

## The phased plan

Growth philosophy: start slow and simple, get tangible traction, and optimize for **acceleration** — compounding — rather than a forced early spike that plateaus. A small retained core compounds: a better shared model brings more contributors, more diverse data makes a better model, and reputation stock accumulates. Each phase's infrastructure makes the next phase cheaper.

Planning number, stated honestly: **a retained core of 10–20 committed operators beats vanity node counts.** Most recruited nodes go dark within months in projects like this. We plan for the stubborn core.

Phases are gated by traction signals, not calendar dates.

### Phase 1 — Prove It

*Community tier. Free. Now.*

Trust machinery, kept as light as the stakes allow:

- Signed provenance manifest for every checkpoint, plus a signed Merkle transparency log — tamper-evident lineage from round 1.
- Commit-reveal updates: a node commits a hash of its update before aggregation, and reveals after. Copying another node's update stops being free.
- Calibration attestation plus statistical quality control from update fingerprints — a wrist-roll distribution far off the fleet median is visible in update statistics without seeing anyone's episodes.
- Transfer-verified credit: visible, non-monetary — badges, model-card lines. Credit accrues when a contribution measurably improves *other* nodes' held-out evaluations.
- A reputation ledger starts accruing, quietly, from the beginning.

**Unlock signal for the next phase:** the decisive local-only vs. federated vs. centralized experiment (M1) is published, and a retained core of nodes is still running rounds.

### Phase 2 — Cluster It

*Near-peer, competitor-free clusters.*

When standing starts to be worth gaming, verification hardens:

- Reputation-weighted influence on aggregation.
- Slow-vesting standing: reputation is earned over months of verified contributions, not in a weekend.
- A per-class canary evaluation battery run every round.
- Dataset-possession challenges: commit-at-capture dataset hashes, plus random sampling challenges that a node holds the data it committed to.
- Frontier-model screening as a *screening layer, not proof*: task-label and video consistency checks, junk and mislabel detection, and embedding-based near-duplicate detection against the known public corpus — a direct counter to replaying public data as if it were private. Measured transfer to other nodes' held-out evaluations remains the final arbiter. A judge can be fooled by plausible data; transfer cannot be argued with.
- Spot-check re-execution on a consented sample.

Clusters are formed so direct competitors are not in the same cluster. Companies do not pool with their rivals; they pool with near-peers in different verticals or geographies.

**Unlock signal:** clusters are forming on their own, and standing is valuable enough that people try to game it.

### Phase 3 — Bond It

*Paid tier.*

When money flows, adversaries finally have something worth stealing, so defenses get economic teeth:

- Operator stake bonds, **slashable only on proof of malicious behavior** — not on poor performance, not on honest bad data, not by vote alone. Proof.
- Reputation gates admission to paid clusters. Reputation is slow to earn and fast to lose: months of verified contributions to climb, one proven malicious act to fall to zero.
- ZK norm/range proofs on updates: prove an update's norm is within bounds without revealing the update. This is the practical bridge between privacy and inspection.
- TEE attestation where hardware allows.
- Aggregator correctness proofs: the server proves it averaged faithfully.

All of this is borrowed from public-blockchain Sybil and staking research. No chain. No token.

**Unlock signal:** paid clusters exist, and the value at stake exceeds the friction cost of bonds and proofs. Proofs and bonds appear exactly when that is true — never before.

### Phase 4 — Institutionalize

*Hierarchical tiers / enterprise.*

- Threshold checkpoint signing: no single party signs a promoted checkpoint alone.
- A steward council with defined authority, pre-committed release criteria, an appeal window, and published dissenting evaluations.
- Insurance and formal audits.
- Differential-privacy tiers, branching by need: legal, contractual, and algorithmic controls, including customer-level differential privacy for enterprise participants.

**Unlock signal:** enterprise and legal demand that actually requires it.

A note on what cryptography can and cannot do, because the plan depends on the distinction: cryptography can prove provenance integrity, possession, norm bounds, and that an aggregation was computed faithfully. It **cannot** prove that trajectories came from a real, calibrated arm rather than a replay or a simulation. That is the oracle problem, and no proof system removes it. Legitimacy in zk0 comes from commit-at-capture, hardware-signed calibration traces, screening, and above all measured transfer. Full zero-knowledge proofs of training are infeasible at current proving costs for a model of this size, and we do not plan on them.

## M0 — Fix the defaults (now)

Before any new scale, fix what the evidence already shows:

1. **Calibration standard + action normalization as a protocol-blocking node-join requirement.** A shared calibration procedure, a versioned calibration registry, and normalized action spaces are a precondition for a node's data or updates to count in a round. An uncalibrated or unnormalized node cannot join a counted round. This is the one intervention aimed directly at the measured 15–25 point real-robot federation gap.
2. **Full action-expert round payloads.** Rounds carry full fine-tunes of the action expert — affordable at this model size — not LoRA-only adapters.
3. **Aggregator A/B, under zk0's configuration.** FedProx with dynamic scheduling is the incumbent, because it won in zk0's published configuration. FedAvg is the challenger, because it won in other configurations. The harness A/Bs them; the result, whatever it shows, is published.

Exit test: the protocol document is published, and the harness enforces it.

## M1 — The decisive experiment

The experiment the field has not published on a real fleet, and the asset this whole roadmap is organized around.

**Design:**

- Four arms, same starting checkpoint, same total data, same task family (sorting + stacking):
  1. Local-only
  2. Federated
  3. Centralized pool
  4. Federated + calibration normalization
- An aggregator A/B is nested inside the federated arms, under zk0's configuration.
- Simulation gate first, on held-out position/task splits, before spending real rollouts.
- Primary metric: **success on held-out site + held-out objects**, with paired trials and confidence intervals reported. Minimum trial counts and the full protocol are fixed in the pre-registration before results are seen.

**Pre-registered bars:**

- **≥ +10 percentage points over the best local-only arm** on held-out sites/objects.
- **Parity within ~5 percentage points of centralized = success.** Superiority over centralized is not claimed, and is not the bar.

**Honest framing of M1:** most of the data available for M1 is already public. Anyone can download that corpus tonight and train centrally, with full curation. So M1 is **not** "unlocking unshareable data." M1 is three things: a rehearsal of the full federated protocol under measurement, a compute pool in action, and the live, un-uploaded tail — data generated this week, on real arms, that never gets posted anywhere. Marketing M1 as anything larger would cost credibility with exactly the technical audience zk0 needs.

**Kill / pivot criterion — pre-committed:**

> If federated + calibration normalization still lands more than 10 percentage points below centralized pooling on held-out sites, federation repositions to a privacy wrapper, and centralized community pretraining becomes the engine.

We decide that on M1 data, not on enthusiasm. The criterion is written here, before the experiment runs, so it cannot be quietly moved afterward.

Recruitment scale for credible evidence is 8–12 sites for M1 (within a broader design range of 8–20 sites, 50–225 episodes per site, narrow task family) — not a three-node smoke test presented as evidence.

## Tier-1 integrity commitments

These are cheap now and nearly impossible to retrofit later. They apply from round 1.

1. **License allowlist + provenance/credit manifest.** Only permissively licensed or explicitly granted datasets count in counted rounds. Every checkpoint ships a machine-readable manifest pinning dataset IDs, revisions/commit hashes, licenses, and contributor credit lines. Without this, a shared checkpoint is legally unusable by any company, and the question "was my data in round 47?" is unanswerable.
2. **Evaluation integrity.** Held-out evaluation sets are firewalled from the training pool. Headline comparisons are paired (same site, A vs. B). We publish *n* and confidence intervals with every success number. There is **no public cross-site leaderboard** until physical evaluation kits and spot audits exist — self-reported success inflates the moment it is published as a ranking.
3. **Calibration is an expiring state.** Servos drift within a session on this hardware class. Calibration is not a join-time ritual: attestation expires, and nodes re-verify periodically. A node whose verification has lapsed does not submit counted updates.

Institution minimums ride along with these: contributor terms with data-ownership and lawful-collection representations, and a named second admin/deputy — the project should not be one person as its entire institution.

## Product stages, mapped to the phases

The product build is staged so each stage is tangible, testable, and demo-able in minutes, and each opens a new contributor door. This is a summary mapping only — the stage specs live with the implementation work.

| Product stage | What it proves | Phase it serves |
|---|---|---|
| Stage 0 — Arm truth | Pinned auto-calibration, calibration registry + verify gate, independent safety envelope (hard joint/torque/workspace limits, e-stop, watchdog), all behind one MCP surface. The arm calibrates itself and refuses an unsafe command. | Prove It |
| Stage 1 — Agent teleop + judge | An agent teleoperates over MCP, records LeRobot-format episodes, a VLM judge labels success/failure (whole chunks, not steps), with a failure note per episode. Judge agreement vs. human labels is measured and published. | Prove It |
| Stage 2 — Local training + the go switch | Local SmolVLA training, a frozen local regression set, and the user-controlled switch from agent control to the on-device policy — with a shadow phase first, where the candidate predicts but does not actuate. | Prove It |
| Stage 3 — First federation | Light aggregator, connector/OAuth join + node manifest, first counted rounds between independently built arms. No raw data moves — verifiably. | Prove It → Cluster It |
| Stage 4 — Trust machinery | Shadow mode for candidate checkpoints, trigger-based capture (human grab, judge/policy disagreement, anomaly spike, aborted chunk), the permanent regression battery (every verified failure becomes a task forever), canary rollout + checkpoint rollback, aggregator honesty checks. | Cluster It |
| Stage 5 — Reasoning head | A parallel language/reasoning head decoded from shared latents — active in training and evaluation, bypassed at real-time execution, never causally upstream of actions — plus the eval-explain-iterate loop on slightly novel tasks. | Cluster It |
| Stage 6 — Ghost-sim + scale-out | Idle-time simulation grown from real failures (load-capped), round mechanics for stragglers/dropouts, aggregation grouped by embodiment class. Compute-only contributors (GPU, no arm) join here. | Cluster It → Bond It |

Two standing rules cut across all stages: **simulation first** — nothing goes to physical arms until it is proven end-to-end in simulation, under one swappable sim/real contract, so a contributor can run the whole stack against a simulated arm and later swap the backend to a real arm with no other change; and a **non-overridable safety envelope** — no model, agent, or checkpoint can override hard limits, the e-stop, or the watchdog. Unattended operation makes that a precondition, not a feature.

## Who we are recruiting

Two contributor classes, now:

- **(a) SO-101 owners with private episodes.** Your arm, your demonstrations, your data stays on your machine. Bring episodes; leave with a live node and a shared model that does more than your local model alone — measured, not asserted.
- **(b) Compute contributors with a 10GB+ VRAM GPU.** SmolVLA fine-tuning fits in roughly 10–16GB of VRAM. Ordinary consumer GPUs count. You do not need an arm to contribute compute, especially as idle and simulation workloads grow.

Later doors, as the stages open: code contributors (calibration and harness work, including joint-range data that helps land upstream auto-calibration), red-team contributors (adversarial data and nodes are a recognized contribution class), sim-environment builders (the environment contract is a public plugin interface), and explanation raters.

Steady state, again: we are optimizing for a retained core of **10–20 committed operators**, not a large number of registered nodes that go dark.

**Join the Discord: discord.gg/dhMnEne7RP**

## Weekly public milestones

Momentum you can check, not a launch spike. We publish **one tangible, publishable milestone per week** — a spec, a verified check, a kit, a design, or a result with numbers, whatever the numbers show.

The week-by-week plan (W1–W8, starting the week of Monday, November 16, 2026: roadmap publication, the M0 protocol spec, a verified node quickstart, the external-node onboarding kit, the data-contribution sprint kit, the provenance/transparency design, aggregator A/B results in simulation, and the M1 pre-registration with recruitment for 8–12 sites) is maintained alongside this roadmap as the weekly milestone plan. If a week slips, the milestone slips with it and we say so — the artifact is the commitment, not the date.

## Workshops

A Monday workshop series runs alongside the milestones. Each session is working, not a lecture: bring an arm or a GPU, leave with the week's stage working.

Sessions:

1. **Node setup** — assemble, connect, perceive, first calibration.
2. **The calibration standard** — the registry, the verify gate, and why normalization is protocol-blocking.
3. **Data contribution sprint** — the sorting/kitting task family, episode format, judge labelling.
4. **Local training & first rounds** — train locally, pass the regression set, join a counted round.
5. **Reading results & eval integrity** — paired comparisons, confidence intervals, what a success number does and does not mean.
6. **The M1 experiment & clusters** — the pre-registered design, site recruitment, and how clusters form.

Dates and registration for each session are published with the weekly milestones and in Discord. Session details live in the workshop specs, not in this roadmap.

---

*Claims discipline for this document, and for everything zk0 publishes: no unproven success claims; no unqualified "privacy-preserving" or "safe" claims; no claim that federation beats centralized pooling; no token or chain promises. If we cannot measure it yet, we say we cannot measure it yet.*
