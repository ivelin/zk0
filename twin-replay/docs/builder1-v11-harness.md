# Builder 1 — twin-suite v1.1 harness report (2026-11-22)

Scope: harness code only (criteria, secondary thresholding, fps, cable
solver-failure, peg jitter, pose sampler, param hash). Full re-validation
was NOT run — poses and scene templates are being fixed in parallel.
All work CPU-only, `nice -n 10`, ≤1 process at a time, DGX training
undisturbed. Python: MuJoCo 4.0.1 venv
`/home/openclaw/muse-work/zk0-sim-v01-20261115/.venv`.

**Inherited-process disclosure (required):** the validation set doubles as
the tuning set — v1's global changes were iterated against the same 100
episodes used for certification, and v1.1 inherits that. Stated, not hidden.

## Fix 1 — Primary criteria (F-01) — `code/replay.py:evaluate()`
Rewritten to use the previously dead `contacts_with()` helper:
- **block_stack**: success iff upper cube in contact with lower cube,
  centre within lower top footprint inset 25%, and upper's contact set is
  exactly {lower cube} ("no other support contact"). Face gap/height are
  now diagnostics only (recorded in `checks`).
- **eye_drops_to_basket**: centre inside basket interior footprint AND
  support test — contact with `basket` body present, no contact with
  `platform` body or world (mat/floor/walls).
- **ring_insert**: engagement boolean at the physical hole clearance
  (RING_R_IN − PEG_R = 3 mm; v1 silently added +5 mm +2 mm) and
  0 ≤ ring_bottom − board_top ≤ 1 cm (v1 credited 5 mm interpenetration).
- **tool_insert**: centre-in-slot now computed in the slot frame (yaw
  applied; v1 used world axes), axis ≤20° kept, silent z-gate replaced by
  the registered "resting in the tray" = contact with the `tray` body.
- **cable_clip**: closure mechanism (documented in code + changelog #8):
  the clip is a fixed-geometry sprung clip — the two static jaw geoms'
  gap IS the channel and cannot articulate — so closure is registered
  mechanically as RETENTION: a segment inside the channel volume between
  the jaws AND in contact with the `clip_block` body at settle.
Registered tolerances/thresholds otherwise unchanged.

## Fix 2 — Secondary criterion (F-02)
`evaluate()` now emits `secondary_pass` + `secondary_threshold_m` per
variant: planar ≤3.0 cm (block/eye/tool), ring ≤2.0 cm when not engaged
(engaged → boolean governs, secondary_pass=True), cable exempt (null).
`run_validation.py` propagates `nominal_secondary_pass` per episode and
`nominal_secondary_passes` per task.

## Fix 3 — FPS (F-05)
Dataset metadata `/home/openclaw/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101/meta/info.json`
says **fps = 20**. v1 ran at 20; v1.1 runs at 20 — value unchanged,
provenance fixed: `twin_params.py` now reads FPS from that file at import
(fallback 20.0). The preregistration's "30 fps" text is superseded by the
dataset's own rate. Also fixed twin_params' stale non-inverted closedness
doc (F-06 bookkeeping) and bumped SUITE_VERSION to twin-suite-v1.1.

## Fix 4 — Cable solver failure + explosion cause
- Detection (in `evaluate()`, constants in `twin_params.py`): a variant is
  `solver_failure` if any cable segment ends non-finite, at planar radial
  distance **>1.0 m** from the arm-base origin, or z outside
  [−0.05, +0.50] m. Justification: workspace/table within ~0.6 m of the
  origin; total cable length 0.72 m; v1 results are bimodal (intact
  ≤0.63 m; the 6 nominal explosions from changelog #7 at 1.15–13.8 m).
  `run_validation.py` excludes solver-failure variants from agreement
  denominators, logs them as defects, and reports the count per task
  (standing explosion-rate disclosure).
- Cause investigation (`/tmp/twin11/cable_dt_test.py`): ep1939 v0 max
  radial 13.758 m @ dt 0.002 → 0.458 @ 0.001 → 0.208 @ 0.0005; ep1969 v0
  12.072 → 2.530 (still exploded) → 0.464. Diagnosis: integrator
  instability of the stiff ball-joint chain (24 ball joints + weld
  solref 0.005) at 500 Hz. **Global fix applied: TIMESTEP 0.002 → 0.0005
  (2000 Hz) for all tasks/variants** (~4× physics cost; full set was
  5–8 min, expect ~20–35 min). Evidence base is 2 episodes — flagged for
  the verifier. Detection + exclusion remain as the standing guard.

## Fix 5 — Peg+board jitter removed (F-11)
`make_draws()` no longer emits `fixture_jitter`; `build()` no longer
applies it. Outcome reference stays at nominal per Amendment 1 A2.
**Process incident:** a parallel builder's `build_scenes.py` rewrite
(18:56, +4.4 KB, v1 base) clobbered Builder 1's first application of
fixes 5–7; all three were re-applied onto the current file preserving
the parallel template changes, verified `fixture_jitter` refs = 0
(changelog #15). replay/twin_params/run_validation were not clobbered.

## Fix 6 — Pose jitter sampling (F-13)
Gaussian pose jitter is now clipped at ±1σ (xy and yaw) — Amendment 1's
"within Builder B's per-pose error estimate". Friction/mass sampling
already conformed and is unchanged.

## Fix 7 — param_table_hash (F-10)
Now hashes (name + contents, fixed order): twin_params.py,
build_scenes.py, replay.py, run_validation.py, calibration.py,
finalize.py, docs/physics-parameters.md — list recorded in changelog #13.
Hash at report time: `4d8d350b6907427c` (will legitimately change as
parallel builders land code; it is stamped at scene-build time).

## Smoke-test evidence (nominal variants, new criteria, 2000 Hz)
| Episode | Primary | Secondary | Key mechanical evidence |
|---|---|---|---|
| block ep411 | FAIL | PASS (0.86 cm) | upper contacts = {world} only — no stack; F-02 divergence now visible |
| block ep531 | FAIL | FAIL (5.54 cm) | upper contacts = {world} |
| eye ep1609 | FAIL | FAIL (11.8 cm) | contacts {basket, platform, world} — support test correctly rejects |
| eye ep1757 | **PASS** | FAIL (7.27 cm) | contacts exactly {basket} — genuine basket rest |
| ring ep7 | FAIL | FAIL (3.94 cm) | engaged=False at 3 mm clearance, planar 3.94 cm |
| ring ep11 | FAIL | FAIL (3.13 cm) | engaged=False |
| tool ep622 | FAIL | FAIL (8.26 cm) | tray contact=True; centre/axis (60.6°) fail |
| tool ep859 | FAIL | FAIL (17.2 cm) | tray contact=True; centre/axis (79.7°) fail |
| cable ep1939 | FAIL (= recorded FAILURE → agreement) | exempt | **solver_failure=False, max radial 0.208 m** (v1: exploded, 13.76 m) |
| cable ep2020 | FAIL (= agreement) | exempt | intact, max radial 0.247 m |

Outcomes are consistent with v1's patterns (pose accuracy is the known
bottleneck, owned by parallel builders); the criteria now derive from
contact/support state and every field is auditable in `checks`.

## Determinism
block ep411 + ring ep7, all 5 variants regenerated twice in-process with
final code: **byte-identical XML both times** (SHA-256 prefixes recorded
in the run log); draws contain no `fixture_jitter`; every pose-jitter
draw within ±σ.

## Files changed
- `code/replay.py` — evaluate() rewrite (criteria, secondary, cable detection)
- `code/twin_params.py` — FPS-from-metadata, TIMESTEP 0.0005, secondary/cable
  constants, stale closedness doc, SUITE_VERSION twin-suite-v1.1
- `code/build_scenes.py` — param_table_hash extension, clipped sampler,
  peg/board jitter removal (applied twice; see incident above)
- `code/run_validation.py` — solver-failure exclusion, secondary +
  explosion-rate reporting
- `validation/param-changelog.md` — entries #8–#16, real DGX timestamps,
  written contemporaneously
- `docs/builder1-v11-harness.md` — this report
- Smoke side effects: nominal (v0) scene XMLs regenerated for the 10 smoke
  episodes; per-episode smoke results under /tmp/twin11/smoke_out (ephemeral)

## Not done / caveats
- Full re-validation not run (per brief; parallel builders own poses/templates).
- Timestep-fix evidence is 2 exploded episodes; verifier should re-check
  the full cable set at 2000 Hz during re-validation.
- Cable clip closure remains a documented modelling device (retention
  test), because the clip jaws are static geometry in the scene template;
  an articulated-jaw model would be a scene-template change, out of
  Builder 1's lane.
