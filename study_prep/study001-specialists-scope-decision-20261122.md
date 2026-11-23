# Study 001 — Specialists Scope Decision (2026-11-22)

**Decision (founder, Ivelin, zk0 chat, 2026-11-22 ~11:22 CT):** run the FAIR
specialist baselines — the five specialists retrained **expert-only**, exactly
matching the training scope of C-exp and the FL arm, so every comparison in
Study 001 is apples-to-apples. The existing full-FT specialists are **kept as
the ceiling reference** and are not touched, moved, or overwritten.

**Why:** as registered (D-10 / Decision 61), the specialists trained the full
model (all 500 tensors, 392,904,096 trainable params) while FL and C-exp train
only the certified action-expert subset (155 tensors / 99,880,992 params). The
founder judged that scope mismatch a mistake for attribution: a specialist-vs-FL
gap could come from the scope difference rather than from federation. The
primary read Φ = C-exp − FL was already scope-matched; this decision extends
the same fairness to the per-task specialist baselines.

## Exact run configuration

| Field | Value |
|---|---|
| Arm (run_train.py) | `specialist_exp` (new arm, see Code change) |
| Tasks | block_stack, eye_drops_to_basket, ring_insert, tool_insert, cable_clip |
| Datasets | The same per-task P0 subsets as the full-FT specialists: 50 episodes each from `armnet/armnetbench_v01_lerobot_so101` (episode-list SHA-256 per task recorded in each run manifest; block_stack list identical to the full-FT specialist's, verified by check-only: eps_sha `db0445408083…`) |
| Init | Pinned base snapshot `lerobot/smolvla_base` @ `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` — every run independent from base, NOT chained (the sequential arm is a different arm) |
| Scope | Expert-only: `train_expert_only=True`, `freeze_vision_encoder=True` |
| Census (verified pre-launch, per task, CPU) | Trainable = certified subset exactly: **155 tensors / 99,880,992 params**; total 450,046,176. `SPECEXP_CENSUS_RESULT PASS` (`run/p1/specexp/census.log`, script `run/scripts/specexp_census.py`) |
| Steps / batch | 3,000 steps each, batch 32, save-freq 375 |
| Recipe | The study recipe via `study_cfg.build` (identical code path to C-exp): AdamW lr 1e-4, betas (0.9, 0.95), wd 1e-10, grad-clip 10, cosine decay, warmup 100, decay to 2.5e-6 at step 3,000, seed 1000, `push_to_hub=false`, video backend pyav, pooled-stats handling as in the study machinery |
| Outputs | `$W/run/train/spec_exp_<task>/out/checkpoints/003000/pretrained_model` (W = `/home/openclaw/muse-work/zk0-study001-20261119`); done-markers `$W/run/train/spec_exp_<task>.done`; run manifests carry amendment tag `SPEC-EXP-20261122` |
| Expected duration | ~1.7 h per run at the expert-only rate → ~8.5 h for five |

## Code change

`run/scripts/run_train.py` gained the `specialist_exp` arm by a strictly
additive patch (original preserved at `run_train.py.preSpecExp20261122`):
new argparse choice; `expert_only` true for the new arm via the same D-10
scope mechanism; manifest amendment tag `SPEC-EXP-20261122` for the new arm
only. Regression check-only runs after the patch confirm the existing arms
are unchanged (specialist: full-FT flags false, same episode hash;
sequential and centralized: unchanged recipes).

## Sequencing and safety

- **No concurrency with the sequential chain.** The driver
  (`study_prep/spec_exp_driver.sh`) waits for `SEQ CHAIN COMPLETE` in
  `run/train/sequence.log` before any GPU work; if the chain fails
  (`BLOCK FAILED seq_*` or the chain driver dies), it aborts and trains
  nothing.
- **Full-path smoke first** (inside the driver, on the freed box): 8 steps
  via the production entry point, checkpoint save at steps 4 and 8,
  fresh-process reload verified at 500 tensors; any failure aborts before
  the production blocks. (Prep-time verification was config-level —
  check-only ×5 + the CPU census — because the GPU was occupied by the
  chain; the GPU smoke runs at miniature scale immediately before the
  ~8.5 h run, per the full-path smoke rule.)
- **Launch:** `launch_dgx_job.sh` (outer job guard, active mode, mirror
  target set), `JOB_LOCK=0` at the wrapper and the fleet
  `gpu-large-model` lock taken per block inside the driver — exactly one
  lock layer, the discipline the sequential chain launch settled on after
  its double-lock deadlock. Whole-box 70% ceiling enforced by the guard.
- **Thermal monitor:** a 30-minute monitor cron is armed for these runs
  under the founder's standing thermal rule (2026-11-22): silent unless
  failure, a guard pause >10 min, sustained ≥85 °C across 3+ consecutive
  samples, or any sample >88 °C; it stays armed until all five runs
  finish, reports completion, and only then stands down.
- Blocks are resumable (the sequencer's `ft_block` machinery): re-running
  the driver skips DONE blocks and resumes an interrupted block from its
  latest complete checkpoint.

## Completion criteria

All five `$W/run/train/spec_exp_<task>.done` markers present with final
checkpoints at step 3,000; per-task final losses recorded; the five
checkpoints backed up to Hugging Face under the `zk0bot` org as dedicated
public study-artifact repos (hash-verified; the shipped repo
`ivelin/zk0-smolvla-fl` is never written); scripts/configs pushed to
branch `study001/fl-substrate-20261121`.

## Completion record

- Completed: 2026-11-23T07:50:57Z (chain marker spec_exp_chain.done); completion work verified 2026-11-23 ~03:20 CT.
- All five expert-only specialist blocks completed 3,000 steps from the pinned base (lerobot/smolvla_base), rc=0 each.

| Task | Final training loss | Wall time (s) | Checkpoint |
|---|---|---|---|
| block_stack | 0.100 | 6086.1 | /home/openclaw/muse-work/zk0-study001-20261119/run/train/spec_exp_block_stack/out/checkpoints/003000/pretrained_model |
| eye_drops_to_basket | 0.088 | 6066.1 | /home/openclaw/muse-work/zk0-study001-20261119/run/train/spec_exp_eye_drops/out/checkpoints/003000/pretrained_model |
| ring_insert | 0.077 | 6224.5 | /home/openclaw/muse-work/zk0-study001-20261119/run/train/spec_exp_ring_insert/out/checkpoints/003000/pretrained_model |
| tool_insert | 0.118 | 6092.8 | /home/openclaw/muse-work/zk0-study001-20261119/run/train/spec_exp_tool_insert/out/checkpoints/003000/pretrained_model |
| cable_clip | 0.142 | 6231.2 | /home/openclaw/muse-work/zk0-study001-20261119/run/train/spec_exp_cable_clip/out/checkpoints/003000/pretrained_model |

Final losses are the last loss logged per job in spec_exp_guard2/guard/command.out (no trainer_state.json is produced at the 003000 training_state path; training_state holds optimizer/scheduler/rng state only).

### Hugging Face backups (public, zk0bot org; remote size + LFS sha256 verified equal to DGX originals)

| Repo | HEAD revision | model.safetensors sha256 |
|---|---|---|
| zk0bot/study001-specialist-exp-block-stack | 9439751ca48cd59e059e311218f40f992221ab1c | e2bf7b87f5c2ab297a5b3850431135f60410fe2f15172e583507586ba5eb8170 |
| zk0bot/study001-specialist-exp-eye-drops | e1c92f67f5e8128ba859808d1669eade99991726 | b8a6cbe0a9e020346bc2986d24d7ec489e1f7a576d704ac4fbd2d2d369f5e701 |
| zk0bot/study001-specialist-exp-ring-insert | 5e4bc2b3df8a1e8a8e5bbea7c92beccd7d0ceeb9 | 3160b57a7009daaa7e46b8eacef6048d5ca0a9b6311fe749c84995e9cd32ba97 |
| zk0bot/study001-specialist-exp-tool-insert | fc68c8587db98451bca38178659aba2d498d57ce | e8ef39a578e26f1d492dbbbd48be8302008b050e73f98849ef0dc1022c5fca05 |
| zk0bot/study001-specialist-exp-cable-clip | 1b4519b56b95c97a25d61d71ee3e314084ff5b0c | 540808b67a410d928263d18e4e344fcb3dce3ba4f79923899f6e8c7904caa59d |

Each model.safetensors is 906,712,552 bytes locally and remotely. Full backup summary: hf_backup_summary_specexp_20261122.json (same directory as this note on the DGX; copy in the fl repo under study_prep/).

### Telemetry

- Guard telemetry (spec_exp_guard2/guard/telemetry.csv, 3,411 samples): max GPU temp 88 °C.
- Guard log: 94 WATCHDOG STOP events and 94 WATCHDOG CONT events (log-only circuit breaker, per founder decision 2026-11-22).

### Shipped model guardrail

- Verified via the Hub API after the backups: ivelin/zk0-smolvla-fl remains at HEAD b400eea1cded125884ff7c5c6ae03b356822180e — untouched.
- The full-FT specialists remain the ceiling reference for Study 001; these expert-only specialists are the fair baselines.
