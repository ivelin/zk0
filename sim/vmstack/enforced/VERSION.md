# Enforced sim stack — snapshot record

**Snapshot date:** 2026-11-27. **Purpose:** the exact stack that produced
and certified the campaign's first Decision-129-certified sim solution
(block_stack ep453), published so a guest with public assets only can
reproduce it end to end.

| Component | Identity |
|---|---|
| Harness | `enforce_replay.py` — **protocol v2 P1+P3+D2+P4+P5** (P5 = full actuation: all 5 arm joints + gripper driven through STS3215 position actuators via interpolated ctrl targets at every substep; no qpos writes, no velocity zeroing after run-start placement) |
| Grip drive | `grip_drive.py` — **gripdrive/1.0-20261127** (kinematic gripper-position channel with measured closedness anchors) |
| Physics engine | **MuJoCo 4.0.1** (PyPI `mujoco==4.0.1`, Apache-2.0) |
| Arm model | **v4** — `arm_padded_v4cam.xml` (camera-fitted) / `arm_padded_v4.xml`, built by `build_arms_v4.py`; carries the `<contact><exclude base×shoulder>` correction (the pair's construction overlap otherwise generates phantom contact forces) |
| Scene | `scenes-v4/block_stack__ep453_v0__v4cam.xml` (block_stack twin of real episode 453) |
| Drive configuration | `config/folded_home_standard.json` — every scene starts from the folded-home keyframe |
| Seat | `driver-rotation-20261127/` — the teleoperation driver-seat contract (`DRIVER_SEAT.md`, incl. the P5 erratum) + seat server/client |
| Calibration inputs | `../../staging/scenes/calibration_v2.json`, `calibration_fk_affine.json` (read by `build_scenes.py` at import) |

## File hashes (sha256, of the files as published here)

| File | sha256 |
|---|---|
| `arm_padded_v4cam.xml` | `be8096dd2455ef9b7f25bab2ef7bb56fb7b371bc850c44a0ea4244f88aebbcfb` |
| `arm_padded_v4.xml` | `0d2436bf7e8a33eda8f1c2d5d191e06d6b6aa998121d6246199824197aff947a` |
| `build_arms_v4.py` | `54a2a18e2786063822e55884ed6dd9d0df6d09a4659d26574714844b2670645a` |
| `enforce_replay.py` | `0a5f87a57bd81b34ccc3432937ff5b76e5143f0d3e696b20373aeaf3ad6f3029` |
| `grip_drive.py` | `27c804fa21bf895d709c550bc4124bd2fb7d3c0885175cd9ba9b451a60a68d14` |
| `pilot453_v4.py` | `dffd8e0e24af31f68411184b5c208b3ff7f9b3ab63e1c7acfc3c07d575613b7d` |
| `admission_gate.py` | `6cc91d7848481e67c2735eaa1e43bd3e7cac5b3ec21394ddc664ecb6f4617463` |
| `canary_drills.py` | `63d263543a5e555589e472c099fc7474bbd48cac0676845006e38d2590f86702` |
| `build_scenes_v4.py` | `68a6eccf8f47adffdd52c7864e35217e0181b506bac95fa3941da05b13390904` |
| `scenes-v4/block_stack__ep453_v0__v4cam.xml` | `ad5996c9b9c2f57354347fe2b023618d14b51323930a23a0ff60c8cc98dddff2` (published copy — see disclosed diff 1) |
| `scenes-v4/block_stack__ep411_v4__v4cam.xml` | `4d4c00bbb92364bdb3e3027549b4dc279fe9304d3d82da9f33c7abef97a8727e` (published copy — see disclosed diff 4) |
| `scenes-v4/block_stack__ep489_v0__v4cam.xml` | `1dca2c7cb4163cf4faeab05fa70e4b8a77629f3186b4c562f3ecc2fb92afc4c9` (published copy — see disclosed diff 4) |
| `config/folded_home_standard.json` | `f9cf9928cd973278986ad9b8c31d94186f5e84606383596dca3e454428f09125` |
| `driver-rotation-20261127/DRIVER_SEAT.md` | `463225dbcd0f093fd08dff6a3e68d3ccc4339c2e31c0a8ec50e0a0d8ef17c2a6` (replaced with the live corrected text 2026-11-27 — see the addendum below) |
| `driver-rotation-20261127/seat.py` | `c6cdc8cd5ec198bd88052b9ed276f10dbbe2b086c169ca8cb9de52bd2de34eaf` |
| `driver-rotation-20261127/seat_server.py` | `de4bb270d2372327e5d1ab98fbd41e3f9088fe043739f57f4b5aafd9c8852acd` |
| `../build_scenes.py` | `e8b79c1a6370866cd89201154949ed3736fa9df05677d5158a2570afffac2c1a` |
| `../replay.py` | `c83a8e9fa0f741e63498da33565dfdb66f4b525df36c32e695e76ce94c70a839` |
| `../twin_params.py` | `9b7ea2656e7abacec6d3eab218e6da698fbff0e38edca75a542ded4a5c739a79` (published copy — see disclosed diff 2) |
| `../../arm_fk.py` | `86fd5c0fbd75e9bae29f112e24df608f64cccb597ff79c94b57da9becca83e21` |
| `../../staging/scenes/calibration_v2.json` | `cc6735d41926e5b10a31bad64fea462ab569b7d5da5a12dca51db6cd06e3fdd0` |
| `../../staging/scenes/calibration_fk_affine.json` | `8a80a7e198f28b2c2f8e2df735015839f211883452619e7ea2353638055cc77f` |
| `../../notes/block_stack__ep453_scene_brief_and_playbook.md` | `6ab2a66aec22125637b0645680abfe1c2d2c59b275c3d78cf50ae5f9a4ef97ee` |
| `../../notes/block_stack__ep411_driver_notes_source.md` | `dcb912ebb134d65bf87639042ae8944bf72fc2c0342ec69ca14c35167c69a7d8` (published copy — see disclosed diff 5) |
| `../../notes/block_stack__ep489_driver_notes_source.md` | `2159a7a5891100efdae7220bb3a28989456c0b4b19b20ce5c28aadd40852cf13` (published copy — see disclosed diff 5) |
| 16 STL meshes in this directory | listed in the snapshot manifest of the publication record; each is a real file here (the working tree holds symlinks to a vendored asset folder) |

Every file not named in the disclosed diffs below is **byte-identical**
to the canonical working-tree file that produced the certification
(sha256-verified at snapshot time).

## Disclosed diffs (the only differences from the canonical files)

1. **Scene include (1 line).** The canonical scene names the arm by an
   absolute path on the author's machine. The published copy includes it
   relatively: `<include file="../arm_padded_v4cam.xml"/>`. Canonical
   scene sha256 `9bf24a99c82826bb48a1ebdf853b382221385fb5251806e115b474fdbb1e53e0`;
   published `ad5996c9…` (above). Nothing else in the file differs.
2. **`twin_params.py` path constants.** The canonical module
   (sha256 `6d190dcbe0fdfd8832d3dbb8aea17bfbb6d0c49c826beeee74337ab1f064bc99`)
   points `TWIN` / `RECORDS` / `SPLIT` / `ARM_XML` / `ARM_ASSETS_TARGET`
   / `DATASET_META` at one machine's working tree. The published copy
   re-points exactly those constants (and two comments that quoted the
   paths) at this snapshot's own layout. **Every criteria and physics
   constant is byte-identical** — the diff touches path lines only.
3. **Driver notes header.** The published playbook carries a 1-line
   HTML-comment provenance header; the body is verbatim from the
   session-2 driver notes.
4. **ep411/ep489 scene includes (1 line each).** The two scenes added
   in the 2026-11-27 pin pass receive the same transformation as
   disclosed diff 1: the canonical scenes name the arm by an absolute
   path; the published copies include it relatively
   (`<include file="../arm_padded_v4cam.xml"/>`). Canonical scene
   sha256: ep411 v4 `ec7bb5adec579f9098184c937169e076a3acd1d28e69f2c205b952f9d079cd29`,
   ep489 `271eb528b6bc2f294406432af60b5eae846721570437f16bbc6d007110f55e34`;
   published copies hashed in the table above. Nothing else in either
   file differs.
5. **Driver notes traceback path (1 line in each notes file).** The
   ep411 and ep489 driver-notes sources quote a seat traceback that
   names `enforce_replay.py` by a machine-local absolute path; the
   published copies name it by its path in this repo
   (`sim/vmstack/enforced/enforce_replay.py`). Source sha256: ep411
   notes `d941c5fc3b0eb96e5d852a684fdb5d4d9ac9ad96c7efbd93ad48a2ab9755e646`,
   ep489 notes `7080f9763fad0ec6d115cf066e5a9fb5b4137ad00a12d68a33cedcca85f3a4c4`;
   published copies hashed in the table above. Nothing else in either
   file differs.

## Addendum — 2026-11-27 pin pass (PR #61 merged)

PR #61 (this snapshot) merged 2026-11-27 as merge commit
`92e1edc4364589e52ef6fff63498400394e19934`. In the same pass:

- the scenes and driver notes for **block_stack ep411** (the
  fidelity-repaired v4 scene of the session-6 re-certification) and
  **block_stack ep489** were added to this snapshot (hash table +
  disclosed diffs 4–5 above);
- `driver-rotation-20261127/DRIVER_SEAT.md` was replaced with the
  live corrected contract text (item 7 in the protocol SSOT §2.7
  panel-rule formulation, plus the input-parity standing rule
  restated 2026-11-27 13:46 CT), and `../../README.md`'s
  certification wording was refreshed from the retired
  different-families formulation to the same panel rule
  (independent verifier finding F1, 2026-11-27);
- no stack code changed: arm, harness, grip drive and engine are
  byte-identical to the snapshot above.

## Guest verification (run at snapshot time, 2026-11-27)

The snapshot was copied to a bare directory with only the published
episode package as input and the certification replay re-run
(see `../../REPRODUCE.md` for the exact commands). The resulting record
matched the official certification record
(`agy2_replay.json`, sha256 `37b81117d8006c7ad661410d04ba34158f5c0a6143acf5efafddbe79ae69420c`)
on **every field except the two input-path fields** (`scene`,
`trajectory`): harness string identical, 374 frames, `aborted: false`,
no enforcement fault, alarm silent, weld never active, grip channel OK,
`sim_success_criteria_read: true`, final cube_b
`[0.25693170991895303, 0.04630475749868331, 0.04480197082047864]` —
bit-identical.

## Notes

- The seat server (`seat_server.py`) resolves the campaign tree layout
  (`validator-sets-20261123/coverage-topup-work/vmstack/enforced`) below
  its seat root, as its contract describes. To run the seat from this
  snapshot, recreate that layout with a symlink — commands in
  `../../REPRODUCE.md`. The certification replay above needs no seat.
- `arm_tool_v4.xml` (a tool-task arm variant in the working tree) is
  **not** in this snapshot: no scene in the certified set uses it. It
  carries the same base×shoulder pair, still unexcluded there — a known
  open item, recorded here so it is not silently lost.
