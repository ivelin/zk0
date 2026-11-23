# Twin-suite v2 rollout harness (Stage 1.4 part 1)

- `sim_runner.py` — closed-loop rollout runner (MuJoCo 4.0.1 sim venv).
  One rollout per invocation; appends to `<out>/rollouts.jsonl`
  (fsync) and skips rollouts already recorded complete.
- `policy_service.py` — SmolVLA policy server (FL venv, CPU). Spawned
  by the runner; JSON-lines over a pipe, one request per 50-action
  chunk, frames as .npy files.
- `run_smoke.sh` / `run_smoke_final.sh` — the Stage 1.4 smoke
  (base checkpoint, block_stack ep411+453, nominal + v0).
- `grasp_probe.py` (recorded-pose grasp+lift), `grasp_probe2.py`
  (close-sweep), `grasp_probe3.py` (scripted IK-centred grasp) —
  diagnostics behind the Stage 1 grasp blocker; see the report at
  ~/workspace/research_notes/zk0-bench003-design-20261119/twin-suite-v2-docs/rollout-harness-v2.md

Run all DGX work through ~/muse-work/tools/launch_dgx_job.sh + job
guard. v1.1 files are frozen: this harness only reads them.
