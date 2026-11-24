# Independent Verification — Spec-Only FULL Gate Suite (Generator / Scenes)

- **Artifact class:** generator + scenes, zk0 twin-suite v2, Amendment 3
- **Verifier:** independent subagent, separate context (builder ≠ verifier)
- **Date:** 2026-11-23
- **Suite under test (desktop, read-only):** `~/muse-work/zk0-spec-only-20261123/work/spec_only_full/` (`suite_manifest.json` + `scenes/`)
- **Reference files:** VM `~/workspace/spec-only-full-20261123/generate_spec_only_full.py` (sha256 `a87b768584ed267da2bb110628586738814124df775f565124f7e81c27a3be33`), `specs_full.json` (sha256 `5bb24b1dadbbd87c1c96092ac8a662ddbb25984934678e1553c30715eff50234`); DGX frozen corners `controls_battery/envelope-corners-v2.json`; DGX demo specs `spec_only/specs/specs.json`; DGX frozen demo generator `spec_only/generate_spec_only.py` (sha256 `c96fcec4c3f7c062941b442ebbe615e645d16634691b044459234ffb75d0cd60`).
- **Method:** code audit, semantic JSON diff, full local regeneration in an isolated scratch dir on the VM, exhaustive XML parsing of all corner variants. No writes were made on the DGX or the desktop.

## Claim 1 — Generator provenance / inputs only from specs: **PASS**

- The generator's only file read is `(HERE / "specs" / "specs.json").read_text()` (line 58). Full-text audit: no other `open`/`read` of data files; imports are Python stdlib only (`argparse, hashlib, json, math, os, random`); the only "episode/pose/recording" mentions are the docstring/manifest statements that none is used. Envelope-corner values are frozen constants in the code, not read from any file at runtime.
- One nuance (not a violation): `ARM_INCLUDE` can be overridden by the env var `SPEC_ARM_INCLUDE`; it was unset during the verification regeneration, and the default is the canonical string (see Claim 6). It is a path string, not scene data.
- Derivation check: diffed against the frozen demo generator on the DGX (sha256 `c96fcec4…`, matching the hash quoted in the full generator's docstring). Total diff 268 lines; every change is accounted for by the declared FULL-1…FULL-5 changes (docstring, `SCENES_PER_TASK` 4→20, corner tables + friction-override plumbing, variant plan k=5..12, manifest kind labels, canonical arm string). Placement sampling, variant draw machinery, per-task scene builders and criteria targets are byte-identical apart from the friction class arguments.
- Desktop copies are identical to the VM copies: desktop `generate_spec_only_full.py` sha256 = `a87b7685…` and desktop `specs/specs.json` sha256 = `5bb24b1d…`, exactly matching the VM files above.
- Regeneration in an isolated scratch dir containing only the generator + `specs/specs.json` reproduced the entire suite (see Claim 3) — proof by construction that no other input is needed. Manifest field `episode_data_used: false`; manifest `master_seed: 20261127`.

## Claim 2 — Output counts and variant structure: **PASS**

- Desktop `scenes/` contains exactly **1,140 XML files** (direct `ls | wc -l`).
- Desktop manifest: 5 tasks × 20 scenes = **100 scenes**. Per-scene variant counts: block_stack 13, eye_drops_to_basket 13, ring_insert 13, tool_insert 13, cable_clip 5 → 4×260 + 100 = **1,140 variants**.
- k-values: precision tasks k=0..12, cable_clip k=0..4 only. Kind labels per precision task (×20 each): nominal, sampled ×4, corner:pose_+x/−x/+y/−y, corner:friction_low, corner:friction_high, corner:mass_low, corner:mass_high. Cable: nominal + sampled ×4 only.
- Manifest `master_seed` = **20261127**.

## Claim 3 — Deterministic regeneration / manifest hashes: **PASS**

- Re-ran the generator on this VM in `/tmp/regen` (generator + `specs/specs.json` copied from `specs_full.json`, `--out` used): output "generated 100 scenes, 1140 XMLs total, seed 20261127".
- Regenerated `suite_manifest.json` is **byte-identical** to the desktop manifest (and parsed-JSON equal).
- All **1,140** regenerated scene files match the desktop manifest's sha256 values: 0 mismatches, 0 missing, 0 extra.
- Independently, on the desktop itself, all 1,140 on-disk scene files were re-hashed against the desktop manifest: **1,140 checked, 0 mismatches**.

## Claim 4 — Corner values vs frozen envelope-corners-v2.json: **PASS**

Frozen file located at `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/controls_battery/envelope-corners-v2.json`. Its values (pose sigma: block 0.025, ring 0.025, eye 0.035, tool 0.035; global floor 0.5/0.96, mat 0.7/1.08, walls 0.4/0.6; per-task friction and mass tables, incl. block cube low 0.005) match the generator's frozen tables exactly. Exhaustive XML checks on the regenerated files (byte-identical to the desktop suite per Claim 3):

- **Pose corners (k=5..8):** all 320 scene-variants (4 tasks × 20 scenes × 4 corners) checked body-by-body against k=0: every manipulated body (cube_a+cube_b, carton, ring, tool) offset by exactly ±sigma in x or y, z unchanged, and **no non-manipulated body moved** — 0 errors.
- **Friction corners (k=9/10):** **2,240 geom checks** across all 160 scene-variants, covering every geom emitted in the scene files (floor, mat, all 4 walls, cubes, carton, platform, basket, ring segments, board, peg, tool, tray/slot/rims) — every sliding component equals the frozen absolute low/high value for its class, and torsional/rolling components are unchanged vs k=0 in every case — 0 errors, 0 unclassified geoms.
- **Mass corners (k=11/12):** 200 checks — cube_a/cube_b each 0.005/0.035 kg, carton 0.02/0.045, tool 0.05/0.1, ring total (sum of 12 segment masses) 0.004/0.015; all non-manipulated masses unchanged vs k=0 — 0 errors. Block cube low = 0.005 confirmed present (the registered OI-2 foam corner).
- **Single-dimension discipline:** friction- and mass-corner variants (320 variants) have body positions identical to k=0; block cubes retain nominal mass 0.02 in friction corners.
- Finding (disclosed limitation, not a claim failure): the envelope file's **gripper** friction corner (1.0 → 0.8/1.2) is not present in any scene file, because the gripper lives in the shared arm include. The generator docstring and the manifest's `friction_corner_note` both disclose this. Claim 4 as stated covers the per-geom tables + global floor/mat/walls, all of which pass.

## Claim 5 — specs_full.json vs demo specs: **PASS (with a wording clarification)**

- Recursive semantic diff of DGX `spec_only/specs/specs.json` vs VM `specs_full.json`: exactly **2 differences**:
  1. `master_seed`: 20261126 (demo) vs 20261127 (full).
  2. `suite_note`: **absent** in the demo specs; present (added) in the full specs with the Amendment 3 text.
- Everything else — all task specs, placement, variant protocol, workspace, gaps register, and the identical `seed_note` string — is equal. So the claim holds in substance: the only differing fields are `master_seed` and `suite_note`; strictly, `suite_note` is an addition rather than a changed string.

## Claim 6 — Canonical arm include path in every scene file: **PASS**

- All **1,140** regenerated scene files (byte-identical to desktop, Claim 3) contain `<include file="/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/scenes-v2/arm/so101_new_calib_padded.xml"/>` — 0 files missing it.
- No scene file contains a local/machine-specific path (`/home/ivelin`, `/tmp/`): 0 files. Output is machine-independent as claimed (modulo the documented `SPEC_ARM_INCLUDE` override, not used here).

## Overall verdict

**CERTIFIED-WITH-FINDINGS**

All six claims PASS with exhaustive (not sampled) evidence: 1,140/1,140 files regenerated byte-identically, 1,140/1,140 desktop files matching their manifest, 320/320 pose-corner variants, 2,240/2,240 friction geom checks, and 200/200 mass checks with zero mismatches. The two findings are documentation-level clarifications, not defects: (1) `suite_note` is added in the full specs rather than changed from a demo-spec string; (2) the envelope's gripper friction corner is not expressible in scene files (arm is a shared include) — a limitation the builder already discloses in the generator docstring and manifest.
