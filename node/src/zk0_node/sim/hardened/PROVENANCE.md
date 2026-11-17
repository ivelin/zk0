# Vendored: the certified noise-hardened realistic pilot (SIM-2)

Build slice 4 (Stage 7 trajectory admission) vendors the certified
rung-3-rematch machinery into the package so the admission stage runs
the *same code* the certification ran — not a re-implementation.

Source (Spark, certified workdir):
`~/muse-work/zk0-rung3-rematch-20261117/` (`harness/` + `code/`),
the machinery behind the V-026 SEALED rung-3 rematch (the
noise-hardened control law passes the telemetry-contract ladder
16/16 Tier 1, incl. 4/4 at rung 3, under the PROVISIONAL placeholder
noise σ_q = 0.005 rad, σ_duty = 0.01).

| file | source md5 | vendored md5 | patches |
|---|---|---|---|
| harness.py | 4df23b0f93b76ba8368a6e0342303412 | feda844477e762eeaeaf2e3408843b55 | 2 (paths) |
| scenes.py | 29183344c9f6ed1147a5cbe03936f36b | 29183344c9f6ed1147a5cbe03936f36b | 0 |
| objects.py | b3b6527f35f5349881fc7c756c3bd9c0 | b3b6527f35f5349881fc7c756c3bd9c0 | 0 |
| grasp_gate.py | 833086a77d1c4f1a9ca8d1d49d9909e2 | 833086a77d1c4f1a9ca8d1d49d9909e2 | 0 |
| bam_pilot.py | 2100219d2c964a189ae937de71e104c9 | e3817b927c2a964b9684156dbaa2e6e6 | 1 (paths) |
| hardened_pilot.py | ebf1d1a3671ac9de4ee0946222925cec | dd4a7755600b905e24e1270d79b33f7f | 1 (paths) |
| wall_pinch.py | 65a384618b23c9e9bdece4ceaccdde7c | 2539d12e7f5dca61e19a8b284adcd5c3 | 3 (paths + mass draw) |
| calibration.json | c1a48721c13143fe597a49542045db52 | c1a48721c13143fe597a49542045db52 | 0 |

Patches (each marked in-file with a `[slice-4 vendoring patch]`
comment block; a diff against the certified source shows nothing
else):

1. **Paths only** (harness.py, bam_pilot.py, hardened_pilot.py,
   wall_pinch.py): absolute Spark workdir constants become
   package-relative paths. `harness.MODEL` is env-overridable
   (`ZK0_HARDENED_MODEL_DIR`) so the runner can point it at a
   writable copy of the staged arm model — the harness writes
   compiled scene XML into its model dir. The punkfab servo law
   resolves to the package's own vendored copy (`../bam/arm.py`,
   slice 3, punkfab/robot-actuators @ fb59c95, MIT).
2. **Mug asset package** (wall_pinch.py): `PKG` was an absolute
   workdir path; the package stages the byte-identical asset package
   itself and passes it via `ZK0_MUG_PKG`
   (`zk0_node.sim.assets.mug_dir()`). Byte identity vs the certified
   workdir copy, verified 2026-11-17: params.json md5
   981589df76b70441a0f66a5f4a6efad8, visual.obj md5
   8ff104788e7691e586fdfb30500a31f2, and the 32 hull_*.obj files'
   aggregate md5 identical.
3. **Mass draw** (wall_pinch.py): `ZK0_MUG_MASS`, when set, scales
   the mug's mass (diaginertia scaled by the same ratio) — the
   Stage 7 admission envelope's mass axis. Unset (the certified
   configuration) the scene is exactly the certified one.

Arm model: the certified harness's live `model/so101.xml` is
byte-identical to the package's staged model-realistic `so101.xml`
(md5 8f3937f109fd512d027935b56689fff8, verified 2026-11-17), so no
model files are vendored; the runner copies the staged model dir.

NOT vendored: `run_rematch.py` (the certified matrix runner). The
package's own runner is `zk0_node/sim/hardened_run.py`; it mirrors
run_rematch's configuration and Tier 1 computation (including its
DetailRecorder, copied there with attribution) and adds the
aim-offset / mass-draw parameters admission needs.
