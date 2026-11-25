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

## Config-equivalence gate — PENDING

Must pass with `STUDY_GATE_ALLOW_OVERRIDES` **unset**. The 2026-11-23 gate PASS was produced with overrides=1 and does not carry. The 2026-11-23 overrides are explained by the checker source (both pointed the checkpoint-config checks at the smoke's own round-1 config instead of the default path); the new run eliminates them by placing the miniature artifact at the checker's default path. Open question the gate's own assertions must answer: the preserved round-10 raw checkpoint config declares pre-rename camera names/native shapes (front/top/wrist) vs the manifest's post-rename camera1/2/3 [3,256,256].

## Full-path smoke — PENDING

Miniature end-to-end with every production path on: train → full-state checkpoint save → hash → reload → eval step → kill-and-resume drill (kill mid-run, resume from checkpoint, verify a continuous record), on the pinned 239 lists, launched only via `launch_dgx_job.sh` with the guard.
