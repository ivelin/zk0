# Study 001 adaptations on the serialized Flower substrate (D-11)

Branch `study001/fl-substrate-20261121`, forked 2026-11-21 from the
reproduction/soak tree (`repro/fl-baseline-dgx-20261121` tip) after
the serialized-substrate soak PASSED (70.3 min, 23 rounds, 115/115
fits — reproduction protocol, 2026-11-21). This branch carries ONLY
the adaptations below plus the study configuration; the repro
baseline branch is untouched. Authority: deviation D-11 to the
BENCH-003 Study 001 preregistration (founder decisions 2026-11-21
13:48-13:51 CT), whose §4 carries the full text.

Adaptations #1-#3 (the substrate's own, from the reproduction) are
inherited unchanged: #1 serialized client concurrency (the client
app module forces Flower's thread pool to width 1), #2 native
precision (no fp32 variation — founder decision 2), #3 the
`build_batch_preprocessor` externalized batch preparation.

## #4 — Dataset wiring (G-1)
`src/configs/datasets.py`: ClientConfig gains optional `root` and
`episodes_file`. `src/common/utils.py`: `load_lerobot_dataset`
gains `root`/`episodes` kwargs, forwarded to `cfg.dataset.root` /
`cfg.dataset.episodes` (lerobot's own DatasetConfig carries both).
`src/client_app.py`: the simulation branch parses the episodes file
(module-level `_load_episode_list`, accepts the P0
`{"subset": ..., "episodes": [...]}` shape) and loads the client
dataset with its root + episode list. The pyproject `[tool.zk0.
datasets]` clients are the five study nodes over the P0-verified
on-disk tree (frames 12,975 / 9,127 / 11,679 / 20,806 / 27,647).

## #5 — Fixed FedProx path (G-2)
`src/server/strategy.py`: when the app config carries
`fixed_fedprox = true` (set), `compute_fedprox_parameters` returns
the constant registered pair (proximal_mu = 0.1, initial_lr)
directly — the substrate's default dynamic path decays μ by 0.95
per round toward 0.01, which contradicts §2's constant μ = 0.1. The
default path is unchanged when the flag is absent.

## #6 — Dynamic weighting (G-3)
`src/server/strategy.py`: `aggregate_parameters` is replaced by the
registered weighting, implemented in the module-level pure helper
`dynamic_weighted_average(entries)`: factor
f = clip(median_loss / policy_loss, 0.5, 2.0) with the driver's own
guards (f = 1.0 when a loss or the median is <= 0), effective weight
eff = round(n_examples x f), weighted average over eff. The
substrate client's `policy_loss` metric (pure policy loss, separate
from its FedProx-inclusive `loss`) is the weighting loss; the
driver's enrolled-NodeKey hash/signature verification has no
substrate counterpart (the substrate's client identity is its
Flower registration) and is not reproduced — the FL arm is ours
end-to-end in this study. Factors and effective weights are logged
per client per round and written to
`<save_path>/server/aggregation_round_<n>.json`. Unit-checked
against the driver's `aggregate_round` (which runs its real hash +
signature verification) on synthetic inputs incl. clip-bound and
zero-loss cases: `study_prep/test_dynamic_weighting.py`.

## #7 — LR schedule (G-4)
New `src/server/lr_schedule.py`: `lr_at(s)` transcribed verbatim
from the driver (warmup 100 -> peak 1e-4, cosine -> 2.5e-6 at
s = 3000). `configure_fit` sends `lr_at((round-1) x 100 + 50)` (the
mid-round offset) as `initial_lr`, replacing the static overwrite.
`src/training/training_setup.py`: with `lr_within_round =
"constant"` in the app config, the within-round cosine is made flat
(eta_min = initial_lr), so the LR is constant within a round as
registered.

## #8 — Seeding (G-5)
`src/client_app.py`: the simulation branch seeds torch and numpy
with 1000 + client_id per client construction (the driver's
convention). The substrate's data pipeline exposes no sampler seed
control; residual variance is inherent to the substrate change and
is carried by D-11.

## #9 — Pooled preprocessor + camera rename (G-6)
`src/training/training_setup.py`: when the app config carries
`pooled_preprocessor_path`, `build_batch_preprocessor` constructs
the D-9 pooled preprocessor instead of building from the client's
own dataset statistics. Mechanism (measured, 2026-11-21):
lerobot's pipeline-level save_pretrained/from_pretrained
round-trip degrades the normalizer stats to float32 (max diff
1.8e-6 on action/state mean/std) and squeezes scalar stat shapes,
so a reloaded pipeline can NEVER hash to the D-9 value. The prep
step (`study_prep/build_pooled_preprocessor.py`) therefore
serializes the pooled normalizer statistics bit-exactly (float64
safetensors) from the driver's own `build_pooled_preprocessor`
construction, and `_construct_pooled_preprocessor` rebuilds the
pipeline from them through the driver's own factory call (same
base-config handling, same rename override argument). The gate is
applied to the object clients actually train with: the prep
constructs through the real client path and requires the driver's
stats hash == a0455b5ba7d75b9c3d743754828020ca751f29fe93447a10620
7553b89d7a39d (D-9). The pipeline-level serialization is still
written (it is the G-8 packaging form, consistent with how every
lerobot checkpoint - including the centralized arm's - carries
its preprocessor).
Camera rename (row 19): lerobot 0.7.0's SmolVLA pipeline does
not rename observations - its rename step carries an empty map
(the certified pilot checkpoint's own policy_preprocessor.json
shows rename_map {}; the driver's `overrides=` kwarg is silently
ignored by this lerobot version's factory, and the from-scratch
factory path forwards no overrides at all). The driver renames in
its data path (make_batches); the substrate equivalent is a
DataLoader collate (`src/client_app.py`: `_study_rename_collate`,
gated on the study configuration) applying front->camera1,
top->camera2, wrist->camera3 to every batch before the
preprocessor - the same net transformation at the same point in
the path.

## #10 — Resume wiring (G-7)
Model-only per-round checkpoints (substrate native, founder
decision 4). `src/server/server_utils.py`:
`initialize_global_model(initial_model_path=None)` loads the
initial global parameters from a checkpoint directory when the run
config carries `initial_model_path`; `src/server_app.py` passes it
through. `src/server/strategy.py`: `self.round_offset` (run config,
default 0) is added to the substrate's segment-local round numbers
in `configure_fit` (LR offset + config) and `aggregate_fit`
(aggregation record filename), so a resumed segment continues the
registered round numbering. Proven by the kill-and-resume drill
before the production launch (hard gate).

## G-8 — Packaging
Post-run step (`study_prep/package_final_checkpoint.py`): copies
the hash-verified pooled preprocessor files into
`checkpoint_round_30` (the substrate's `save_model_checkpoint`
copies BASE preprocessor files in), then verifies the packaged
directory's preprocessor hashes to the D-9 value and loads via
`SmolVLAPolicy.from_pretrained` + `make_pre_post_processors`.

## Configuration (pyproject.toml)
30 rounds; local-epochs 100 (= steps/fit); batch 32; fraction-fit
1 (5-of-5 — founder decision 5: any short round voids the run);
fraction-evaluate 0 and eval-frequency 31 (server evals off except
the substrate's round-0 evaluation, logged and ignored — decision
5); proximal_mu 0.1 with fixed_fedprox; checkpoint_interval 1;
use-wandb false. **The hf repo key is ABSENT**: pushes are disabled
(the runE/runF accidental pushes are the cautionary tale); local
checkpoints only.

## Verification (pre-launch)
- `study_prep/preflight_gates.py`: resolved smolvla_base snapshot
  == 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 and its
  model.safetensors SHA-256 == the sign-off value; census through
  the substrate construction == 155 trainable tensors / 99,880,992
  params of 500; per-client loads == 50 episodes and manifest frame
  counts, fps 20, chunk 50.
- `study_prep/test_dynamic_weighting.py`: #6 vs the driver.
- Full-path miniature smoke (2 rounds) + kill-and-resume drill
  (hard gate) before the production launch.
