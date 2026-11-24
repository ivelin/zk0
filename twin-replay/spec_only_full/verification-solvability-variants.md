# Verification — Solvability Variants (V2b)

Independent certification (separate context) of the spec-only FULL gate suite variant-level canonical solutions — twin-suite v2, Amendment 3. Founder rules 2026-11-23 13:05/13:09/13:12 CT: one canonical solution per (scene, task); a solution counts only when independently certified.

**Overall verdict: CERTIFIED — 27/27 reproduced.**

## Scope

The 27 scenes in the builder's register (`work/spec_only_full/register.json`, desktop Pop!_OS) with `solution_within_envelope: true` and `proven_nominal: false`, i.e. not proven at the nominal variant but carrying a claimed executed pass at an envelope variant (sampled k=1..4 or pose corner k=5..8). Claimed counts verified against the register: block_stack 8, eye_drops_to_basket 14, tool_insert 5, cable_clip 0, ring_insert 0 = 27. All 27 register entries carried `certification.status: "pending"` as stated.

## Method (independence)

- Enumerated the 27 scenes, controllers, variant_k values and builder check records by reading the register directly — not from the builder's summary.
- Staged my own copies of the 27 canonical variant scene files from `work/spec_only_full/scenes/` into my own workdir on the desktop (`~/muse-work/zk0-verify-v2b-20261123/staged/`), rewriting ONLY the arm-include line to the desktop arm copy (`~/muse-work/zk0-spec-only-20261123/work/scenes-v2/arm/so101_new_calib_padded.xml`). Verified programmatically for all 27: the canonical file's sha256 equals the suite-manifest sha256, and each staged file differs from its canonical file in exactly one line (the include line). Staging report: `staging_diff_report.json` in the verifier workdir.
- Re-executed each canonical solution with the builder's driver `run_reference_spec.py` (frozen controller code from `work/reference/`, `work/code/`, `work/harness/` — read-only, unmodified), `SPEC_VARIANT=<k>`, `SPEC_MANIFEST` = the suite manifest, `SPEC_STAGED` = my staged dir, `PYTHONPATH=~/muse-work/zk0-spec-only-20261123/pylibs` (MuJoCo 4.0.1).
- Runs executed under the job guard (`job_guard.sh <rundir> v2b-verify active`, mirror target `~/muse-work/zk0-spec-only-20261123/telemetry-mirror`), 3 concurrent, `nice -n 10`, per-run timeout 1500 s. Guard preflight passed; command exited rc=0; final mirror OK. All 27 runs completed rc=0.
- Verdict per scene: certified iff my re-execution returned `sim_success=true` AND my checks matched the builder's recorded checks exactly (floats compared at 4 decimal places, contact lists as sets).

## Per-scene results

| Scene | variant_k | Controller | Verdict | Checks |
|---|---|---|---|---|
| block_stack__spec01 | 7 | frozen (refcore.py + run_reference.py) | PASS — reproduced | match |
| block_stack__spec07 | 2 | frozen (refcore.py + run_reference.py) | PASS — reproduced | match |
| block_stack__spec09 | 3 | frozen (refcore.py + run_reference.py) | PASS — reproduced | match |
| block_stack__spec11 | 1 | frozen (refcore.py + run_reference.py) | PASS — reproduced | match |
| block_stack__spec14 | 1 | frozen (refcore.py + run_reference.py) | PASS — reproduced | match |
| block_stack__spec16 | 1 | frozen (refcore.py + run_reference.py) | PASS — reproduced | match |
| block_stack__spec17 | 4 | frozen (refcore.py + run_reference.py) | PASS — reproduced | match |
| block_stack__spec18 | 1 | frozen (refcore.py + run_reference.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec00 | 1 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec01 | 1 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec02 | 1 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec03 | 3 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec06 | 3 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec07 | 4 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec08 | 6 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec09 | 4 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec11 | 3 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec12 | 6 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec14 | 4 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec15 | 3 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec17 | 1 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| eye_drops_to_basket__spec19 | 2 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| tool_insert__spec08 | 8 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| tool_insert__spec09 | 3 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| tool_insert__spec12 | 2 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| tool_insert__spec16 | 7 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |
| tool_insert__spec19 | 3 | rev2.1 (refcore2.py + run_reference3.py) | PASS — reproduced | match |

My per-scene result files are on the desktop in `~/muse-work/zk0-verify-v2b-20261123/results/` named `<scene_id>_<version>_k<k>_verify.json` (the `record` values in `certification-v2b.json`).

## Discrepancies

None. Every scene reproduced: `sim_success=true` on re-execution and every recorded check field matched the builder's canonical-solution record exactly — including the continuous fields (block_stack `face_gap_m`, tool_insert `axis_angle_deg` / `tool_z`, eye_drops `carton_z`), not just the boolean criteria.

## Notes

- These certifications are variant-level: each scene is certified solvable within its envelope at the named variant_k, under the frozen mechanical criteria. They do not assert nominal-variant solvability (all 27 are `proven_nominal: false` by construction of this set).
- Builder's directories were accessed read-only; nothing in them was modified. All verifier artifacts live in `~/muse-work/zk0-verify-v2b-20261123/` on the desktop (targets.json, staging_diff_report.json, results/, rundir/guard telemetry).

**CERTIFIED — 27/27 reproduced.**
