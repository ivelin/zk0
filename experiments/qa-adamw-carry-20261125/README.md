# Experiment Q-A — AdamW-carry FL rerun on certified data

**Branch:** `experiment/qa-adamw-carry-20261125` (this branch is the experiment's record home — founder rule 2026-11-25 ~14:37 CT: every experiment gets a dedicated branch; the log, decisions, and decision traces live here. A GitHub issue points here as a summary; it is not a second home.)
**Base:** `study001/fl-substrate-20261121` @ `6301d104906d14e872e6c433b3fb43c879ecbd45` (tip after the PR #52 + PR #53 merges, 2026-11-25). This branch is not a PR and will not become one.

## The experiment

Study 001's federated arm (fl_r30) scored 26/100 vs the centralized expert-only arm's 70/100 (Φ = −44 pp) — on data whose completion axis had never been certified. Q-A reruns the FL arm with **AdamW optimizer state carried across rounds** (the diagnosed suspect) on the **fully certified** run set. Whatever the rerun shows is the result: the first FL-vs-centralized number on certified data, reported honestly either way (F50 Physical AI Summit, UT Austin, Thu Dec 3, 2026; results lock EOD Mon Nov 30, 2026).

Two variables differ from fl_r30 — AdamW carry **and** the dataset (250 → 239 certified). The headline must never claim carry alone was isolated. The eval denominator is **90** (20×4 + cable 10), not 100.

## Pinned run set — 239 episodes

Certified PASS pool is 249 (246 CLEAN + 3 RECOVERED; ep604 is the only certified FAIL, excluded). Training excludes the 10 certified cable episodes drawn as the held-out eval set (founder re-split 2026-11-23, seed 20261123), leaving **239 train** (236 CLEAN + 3 RECOVERED). Full composition: `run_set_manifest.json`; per-list hashes: `payload_sha256.txt`; the lists: `episode_lists/`.

| Task | n | sha256 |
|---|---|---|
| block_stack | 49 | `dac978f5f0ebfb5c38f891e79d50330e82e02f72dd2660d4921d073df6ebf845` |
| eye_drops_to_basket | 50 | `b28fe1680627ad0c2368e2651724b26e684a9ced506ff1f6f568088fba58c3e4` |
| ring_insert | 50 | `54e5562ef8e0501d13adff00ea1328d24952d2b8511e29c912bd78e3a917fbb2` |
| tool_insert | 50 | `840b288ba37589c003ad5232d4e91fcde454a5995dda0de619d1cf492ddeb6d0` |
| cable_clip | 40 | `34071c00c3fe58b37cff2e81e93d7216fec58f339a4c34bbdf84a54afbb3f772` |

## Records in this folder

- `RUN_LOG.md` — the living run log (entries appended as the run progresses)
- `DECISIONS.md` — decision traces for this experiment
- `GATE_SMOKE.md` — gate + smoke status and reports
- `run_set_manifest.json`, `payload_sha256.txt`, `episode_lists/` — the pinned run set

## Status (2026-11-25)

- Mirror proof: **PASSED** 2026-11-25 14:33 CT (see `GATE_SMOKE.md`)
- Config-equivalence gate: **pending** — must pass with overrides unset
- Full-path smoke: **pending**
- Launch: on gate + smoke pass, no calendar wait (founder acceleration 2026-11-25 ~14:28 CT)
- Run home on the DGX: `~/muse-work/zk0-qa-relaunch-20261125/` (guard telemetry mirror staging stays on-box there; this branch is the durable off-box record)
