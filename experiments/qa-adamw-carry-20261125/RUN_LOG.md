# Q-A run log — living document

Entries appended as the run progresses. Times are America/Chicago (CT).

## 2026-11-25

- **~14:22 — Launch approved.** Founder approved the F50 results sprint and the Q-A relaunch on the voice call: "get solid numbers for federated learning on certified data." Q-A had been stopped 2026-11-23 pending the certified set (done) and the founder's word (given today).
- **14:33 — Mirror proof PASSED.** Three consecutive job-guard mirror syncs (startup / periodic / exit) all OK through the real launch path (`launch_dgx_job.sh`, active mode, breaker T3′ armed, preflight green). The five pinned episode lists + `run_set_manifest.json` were staged into the guard path and mirrored; sha256 of the mirrored lists match the register-derived certified lists exactly (eye/ring/tool also reproduce the 2026-11-23 gate report's hashes digit-for-digit). Details: `GATE_SMOKE.md`.
- **~14:26–14:35 — Run set corrected 238 → 239.** The reconciliation audit (item C1) found ep573's register `human_status` field stale: the founder adjudicated ep573 PASS on 2026-11-23 16:10 CT, recorded in the register's own notes, the training verdicts_master, and the 2026-11-23 daily log. Register corrected at the source; ep573 IN (block 49). The audit was independently verified CERTIFIED the same day (counts recomputed to the digit). See `DECISIONS.md` #4.
- **~14:36–14:37 — Log home decided.** Founder: the run's log copy lives remotely for the experiments; refined minutes later — every experiment gets a dedicated GitHub branch holding the log, decisions, and decision traces; a GitHub issue is a pointer/summary only. This branch is that home. The guard's telemetry mirror itself keeps pointing at the DGX staging dir (`~/muse-work/zk0-qa-relaunch-20261125/mirror`): the originally intended off-box target (the Muse VM) is structurally unreachable from the DGX — proven today with a live listener (connection refused at the platform layer; the VM has no SSH server). Durability off-box is provided by these committed records.

## Pending

- Config-equivalence gate (overrides unset) — gate workstream running
- Full-path miniature smoke incl. kill-and-resume drill
- Launch (on gate + smoke pass), 30 rounds at the registered budget, full-state checkpoints ≤30 min apart
- Canary judging (40-episode canary, hardest first) → final eval + two-judge certification → results lock Mon Nov 30 EOD
