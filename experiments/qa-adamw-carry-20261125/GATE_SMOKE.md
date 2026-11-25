# Q-A — gates and smoke

Launch rule (standing): the full run launches only on **config-equivalence gate PASS with overrides unset** + **full-path smoke PASS**. A gate failure stops the launch; waivers are the founder's, by name — nobody waves a gate through.

## Mirror proof — PASSED (2026-11-25 14:33 CT)

Run through the real launch path (`launch_dgx_job.sh` → job guard, active mode, breaker T3′ armed; preflight green: 118 GB mem avail / 1,339 GB disk / load 0.30):

- Three consecutive mirror syncs, all OK — startup 14:33:18, periodic 14:33:34, exit 14:33:49 CT. `mirror_status` = OK, job exit code 0, zero failures (vs the preserved 2026-11-23 run's 110× rc=255).
- Payload: the guard telemetry set (telemetry.csv, guard.log, mirror_status, command.out, exit_code) plus the pinned run set — all five episode lists + `run_set_manifest.json`.
- Hash verification — mirrored = certified, exact on all five tasks:

| Task | n | sha256 |
|---|---|---|
| block_stack | 49 | `dac978f5f0ebfb5c38f891e79d50330e82e02f72dd2660d4921d073df6ebf845` |
| eye_drops_to_basket | 50 | `b28fe1680627ad0c2368e2651724b26e684a9ced506ff1f6f568088fba58c3e4` |
| ring_insert | 50 | `54e5562ef8e0501d13adff00ea1328d24952d2b8511e29c912bd78e3a917fbb2` |
| tool_insert | 50 | `840b288ba37589c003ad5232d4e91fcde454a5995dda0de619d1cf492ddeb6d0` |
| cable_clip | 40 | `34071c00c3fe58b37cff2e81e93d7216fec58f339a4c34bbdf84a54afbb3f772` |

- Caveat (recorded, not smoothed): the reachable mirror target is the on-box staging dir in the run home. The off-box VM target is structurally unreachable from the DGX (proven 2026-11-25). Off-box durability of the records is this branch (see `DECISIONS.md` #8).

## Config-equivalence gate — FAIL 118/119 (2026-11-25 ~14:56 CT), then founder waiver

Overrides-off run (checker `--construct`, all three arms registered, no `--checkpoint-config` / `--save-template-config` / `--waive`, `STUDY_GATE_ALLOW_OVERRIDES` unset; report `overrides: []`, tree_hash 2affb7d0…): **118 PASS, 1 DIFF-FAIL** — `saved_checkpoint.input_features` (fl-artifact, shape-exact rule): the substrate's raw saved checkpoint config declares, after rename normalization, camera1 [3,576,1024], camera2 [3,576,1024], camera3 [3,720,1280] vs the registered [3,256,256] x3. This answers the camera-form question from the checker's own assertions: the 2026-11-23 gate passed this check only through the `--save-template-config` override; placing the raw artifact at the default path does not legitimately eliminate it. No launch followed the FAIL.

**Founder ruling 2026-11-25 14:58 CT (voice): APPLY THE WAIVER.** Registered as `W-rawsave-form` in the gate manifest (sha256 b9e21bec… -> 5bec3de8…); reasoning + scope in `DECISIONS.md` #10 (raw form = the substrate's real behavior for every arm/round; the packaged form is what the eval gate shape-checks; the waiver covers ONLY this form check). Gate re-run with `--waive saved_checkpoint.input_features=W-rawsave-form` follows; the report must show 118 PASS + 1 DIFF-WAIVED and no configuration overrides.

## Full-path smoke — Phase A PASSED; Phases B/C after the gate re-run

Phase A (2026-11-25 14:48-14:53 CT, through the launch path, guard + mirror OK): 1 round x 10 steps x 5 serialized clients on the 239 lists, rc=0; 5/5 per-client optimizer states loadable at step 10; round-1 checkpoint saved + hashed (model sha256 667855ce…, config 0ed91253…; per-client state hashes in the run dir `smoke_manifest.json`). The miniature checkpoint's config.json was placed at the gate checker's default path, as the smoke driver is designed to do.

Phases B (kill-and-resume drill: kill mid-round, verify state intact/loadable, resume to a continuous step record) and C (packaging: pooled preprocessor + A5 metadata config correction on a copy of the miniature, then the shape-exact `input_features` check on the PACKAGED form + standard load test) run after the waived gate re-run passes. Phase C's packaged-form shape check is the proof that makes the waiver sound: if the packaged form failed it, the launch would stop.
