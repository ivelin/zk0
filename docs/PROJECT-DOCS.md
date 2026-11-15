# zk0 project documents — where the canonical records live

Project documents (roadmap detail, design decisions, specs, research) live in the
team's systems of record — Google Drive first — not in this repo. This repo
carries code and short pointers only.

**Canonical design decisions (2026-11-14):**
[zk0 — Node, Connector & Runtime Design Decisions (2026-11-14)](https://docs.google.com/document/d/1E8dslWsnOjyQlS2JDf9S7sCmfoQbFUj7BlHYbIIh5fg/edit?usp=drivesdk)
(Google Drive, folder `zk0.bot/`)

Headline decisions recorded there, settled 2026-11-14:

- **Local-first connector:** the arm is controlled by a local zk0 runtime on the user's PC (USB default; Pi/WiFi bridge optional); the hosted connector at mcp.zk0.pirin.ai is coordination only — never a proxy for arm control or episode data.
- **One company domain, separate repos:** pirin.ai subdomains per project (Bootstrap pattern); zk0 connector code lives in this repo.
- **Runtime packaging:** small stable zk0-core + a pinned LeRobot/Flower train pack behind zk0-owned adapters; one `zk0` launcher per OS; signed atomic updates with rollback and canary rollout.
- **Design principle:** "Ask your bot to DO it" — every user-facing flow must be executable end-to-end by the user's agent; humans do only the physical and the consent acts (terms acceptance is non-delegable, rule SM-5B).
- **Agent-as-teleoperator:** staged path — agent-run calibration, agent repetition for known tasks, human-seeded new tasks, zero-shot output counted only as candidate data after the same verification as human data.
- **Legal structure:** company Terms/Privacy canonical on pirin.ai (Cocoonhive LLC dba pirin.ai); Apache-2.0 in this repo covers code and public shared model weights; contributors actively accept the terms before counted participation (inbound = outbound).
- **Hardware baseline:** SO-101 is the default arm.

Public roadmap summary: https://zk0.bot/docs/ROADMAP.html (see docs/ROADMAP.md).
If anything in this repo conflicts with the Drive doc, the Drive doc wins — flag the conflict.
