# Sim harness provenance

The two Python modules in this directory are **vendored copies of
certified zk0 sim code**, vendored 2026-11-17 for the zk0 Node package
so the package is self-contained. The certified originals are untouched
and remain the reference; the copies differ from them only by the
declared patches listed below.

| File | Certified original | Original md5 | Patch |
|---|---|---|---|
| `multitask.py` | `zk0-multitask-20261115/multitask.py` (DGX Spark, `~/muse-work/`) — the MuJoCo multi-task benchmark whose T1 green-box run carries the certified gated verdict of 2026-11-16 (SSOT decision 27) | `87b1bad6cdda448d79deedf6933c6a64` | The hardcoded build-directory paths `W`/`MODEL` are read from the `ZK0_SIM_MODEL_DIR` environment variable (set by the package runner to its staged model directory). No task, pilot, IK, or scoring logic changes. |
| `grasp_gate.py` | `zk0-multitask-20261115/grasp_gate.py` (byte-identical copy also in `zk0-rung3-rematch-20261117/harness/`) — the grasp-quality gate of SSOT decision 27 | `833086a77d1c4f1a9ca8d1d49d9909e2` | `GateRecorder.sample()` additionally records per-body `max_single` (largest single-contact normal force in the sample), needed by the Tier 1 damage criterion of the benchmark preregistration. Read-only with respect to physics and to every pre-existing field; `evaluate_cycle` is unchanged. |

Arm model files are **not** vendored into git (spec §1.3: assets are
pulled from the Hugging Face collection at pinned versions with
content hashes). The runner stages them into the runtime asset cache:

| Asset | Pinned source | File | md5 |
|---|---|---|---|
| SO-101 arm, realistic variant | HF `zk0bot/so101-realistic` v0.2.0 (model files byte-identical to v0.1.0) | `so101_realistic.xml` | `8f3937f109fd512d027935b56689fff8` |
| SO-101 arm, stock (control runs only) | MuJoCo Menagerie `robotstudio_so101` HEAD, as used by the certified multitask benchmark | `so101.xml` | `1470b61558f0f18762e27e7431ac51ce` |
