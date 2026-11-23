# Independent Verification — Twin-Suite v2 Controls Battery Prep, Part A (Envelope Solvability + Controls Manifest)

**Verifier:** independent subagent, separate context, raw artifacts only.
**Builder's report:** `controls-battery-prep-v2.md` (same folder).
**Governing:** pre-reg `twin-suite-v2-preregistration-20261122.md` §3.2–§3.4, §4, §6; union ruling in `verification-v2-solvability.md` §R5.
**Artifacts:** DGX `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/` (`reference/results-env-frozen/`, `reference/results-env-rev21/`, `controls/controls-manifest.json`, guard rundirs `runs/solvability-env-frozen/`, `runs/solvability-env-rev21/`); verifier re-runs at DGX `/home/openclaw/muse-work/verify-controls-prep-20261123/` (rundir `runs/verify-controls-prep/`, guard exit_code 0).
**Date:** 2026-11-23.

## VERDICT: CERTIFIED

Every Part A claim — envelope batch provenance and guard discipline, per-version and union counts, combined denominators, all three control-checkpoint pins (re-hashed at the source), the Part B matrix arithmetic, and the no-rollout / no-v1.1-modification scope statements — is confirmed against the raw artifacts. Both decisive re-runs reproduced the batch records **byte-identically**. The only standing note is the known off-box mirror infrastructure gap (§1), which the builder reported accurately and which affected this verifier's own batch identically.

---

## 1. Envelope batches — CONFIRMED

- **Single frozen batches, all 20 scenes each.** `batch_env_frozen.sh` is one loop invoking `run_reference.py <task> <ep> results-env-frozen/…` (timeout 900, `nice -n 10`) over exactly the 20 envelope scenes; `batch_env_rev21.sh` is the identical loop with `run_reference3.py` into `results-env-rev21/`. No per-scene scripts, flags, or parameter changes. All 20 record files in each directory have mtimes inside its single batch window (frozen 03:34–03:59 CT; rev2.1 04:00–04:28 CT), and both `batch_env_*_summary.txt` files record rc=0 for all 20 scenes with verdicts matching the JSON records.
- **Guard discipline.** Both rundirs' `guard/exit_code` = **0**; guard logs show single launches via the guard (`cmd=bash reference/batch_env_frozen.sh` / `batch_env_rev21.sh`, logonly, CPU-only), preflight load1 0.25 (frozen). **MIRROR FAIL rc=255 logged 10× (frozen) and 11× (rev2.1)** — ssh connection refused to the precedent off-box target, exactly as reported; telemetry is on-box only. No `JOB_GUARD_ALLOW_NO_MIRROR` / override appears in either guard log. **Zero WOULD-KILL lines** in either log.
- **Controller md5s unchanged** (re-hashed by this verifier after the runs): `refcore.py` `96aa928072e5487b1fe1f077be293132`, `run_reference.py` `a9d29bdb8f07c8b55181d10295f6d457`, `refcore2.py` `ee9ea862d6f4b3f8b133da0f0ad43c33`, `run_reference3.py` `22b4b758d67933ac05fc8ce892262485` — all match the frozen values.
- **Generation marking:** all 40 envelope records carry `"generation": "ORACLE-GUIDED"`.

## 2. Union register arithmetic + decisive re-runs — CONFIRMED

Recomputed from all 40 record files:

| Version | block | eye | ring | tool | cable | Total |
|---|---|---|---|---|---|---|
| Frozen | 4 | 1 (1609) | 0 | 0 | 4 | **9/20** |
| Rev2.1 | 2 (499, 549) | 0 | 0 | 0 | 2 (2068, 2085) | **4/20** |
| **Union** | **4/4** | **1/4** (1609) | **0/4** | **0/4** | **4/4** | **9/20** |

Matches the builder's §1 table scene-for-scene, including the five frozen-only validations (block 527/546, eye 1609, cable 1905/2046) and zero rev2.1-only gains. Spot-checked failure reasons against the records: rev2.1 block 527 `within_inset_footprint: false`; rev2.1 block 546 `no_other_support_contact: false`; rev2.1 cable 1905 `cable_seated_in_clip_closed: false`, `solver_failure: false`; frozen tool 885 `centre_in_slot: false`, `resting_in_tray_contact: false` — all as reported.

**Verifier re-runs** (DGX's own frozen `run_reference.py`, output to verifier scratch, via `launch_dgx_job.sh` + `job_guard.sh`, logonly, `JOB_LOCK=0`, CPU-only; guard exit_code 0, MIRROR FAIL rc=255 logged 2×, no override, zero WOULD-KILL):

| Scene | Batch record | Re-run |
|---|---|---|
| block_stack 527 (frozen-only validated) | PASS | **byte-identical** (md5 `32cde516…` = batch md5) |
| eye_drops_to_basket 1760 (both-fail) | FAIL (grasp infeasible in all yaw branches) | **byte-identical** (md5 `7210fe0b…` = batch md5) |

## 3. Combined denominators — CONFIRMED (arithmetic)

Dev union per the §R5 ruling (block 4/4, eye 3/4, ring 0/4, tool 1/4, cable 3/4 = 11/20) + envelope union (§2 above):

| Task | Dev | Envelope | Combined |
|---|---|---|---|
| block_stack | 4/4 | 4/4 | **8/8** |
| eye_drops_to_basket | 3/4 | 1/4 | **4/8** |
| ring_insert | 0/4 | 0/4 | **0/8** |
| tool_insert | 1/4 | 0/4 | **1/8** |
| cable_clip | 3/4 | 4/4 | **7/8** |
| **Total** | 11/20 | 9/20 | **20/40** |

Nominal gate denominator (§2 of the pre-reg) is 96 (block 20, eye 20, ring 20, tool 16, cable 20); attempted = 40, so **56 gate scenes unattempted** — correctly flagged by the builder as out of Part B scope, and ring_insert's no-gate consequence (A2.2(iii)) and tool_insert's one-scene denominator are stated plainly.

## 4. Controls manifest — CONFIRMED, all hashes re-computed by this verifier

`controls/controls-manifest.json` (schema `twin-suite-v2/stage1-controls-manifest/1`) is the only file in `controls/`, created 2026-11-23 04:35 CT, and carries `no_control_run_has_been_executed_under_this_manifest: true`.

| Control | Pin in manifest | Verifier re-hash / check |
|---|---|---|
| Base | revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`; weights SHA `7cd549ac…a01eb` | Local snapshot path exists at that exact revision; `model.safetensors` SHA-256 = **7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb** ✓ |
| FL round-30 | `checkpoint_round_30_a5repair`; weights SHA `5b1cbf29…7a84b4` | SHA-256 of the actual file at the manifest path = **5b1cbf290b8b4beeeb9a77473f0d03c0e3ffce8c038b7aab56bc6795fc7a84b4** ✓ (identical to the pre-reg §6 pin) |
| C-exp | `run/train/centralized/out/checkpoints/015000/pretrained_model`, 15,000 steps; weights SHA `8f6807ed…d48379` | SHA-256 = **8f6807ed65d997f2820ff68205431c60cf40d5b35f56da1127b8179f65d48379** ✓; run manifest records `"steps": 15000`, `train_expert_only: true`; `checkpoints/last` → `015000` ✓ |

**C-exp identification is unambiguous, as claimed:** `run/eval-traj-20261122/eval-plan.md` names exactly this path as the cexp arm ("centralized expert-only, 15,000 steps"), and `run/eval-traj-20261122/scripts/sweep.sh` runs `run_ckpt cexp $W/run/train/centralized/out/checkpoints/015000/pretrained_model`. No substitution. Reference ordering base 0/100 < FL 26/100 < C-exp 70/100 is recorded in the manifest per A1.1.

## 5. Part B matrix fidelity — CONFIRMED

- **L1 = 300:** 20 validated scenes (block 8, eye 4, tool 1, cable 7 — the §3 combined register restricted to validated scenes) × K=5 variants (§4) × 3 manifest checkpoints = 300. ✓
- **L2 = 120:** envelope-subset validated precision scenes = block 4 (499/527/546/549) + eye 1 (1609) = 5 × 8 non-nominal MuJoCo corners × 3 = 120. The frozen `envelope-corners-v2.json` defines exactly 10 corners per precision task (nominal, pose ±x/±y, friction low/high, mass low/high, `engine_drake`); removing nominal and the Drake corner leaves exactly the 8 used. ✓
- **L3 = 15:** same 5 scenes × nominal instance × Drake × 3 = 15. ✓ **Total 435** (300 + 120 + 15), with the optional Drake-variant extension (+60 → 495) correctly presented as an option, not silently included.
- **Cable excluded from corners:** the corners JSON defines tasks for block_stack, eye_drops_to_basket, ring_insert, tool_insert only — no cable_clip entry — so cable appears in L1 only, exactly as the builder states. Ring contributes no scenes (0 validated). Friction corners carry the [DEFAULT]-at-corner flag in the JSON, matching the builder's flagging note.
- **56 unattempted gate scenes correctly out of scope:** the matrix is defined over the 20 validated scenes only (A2.2 denominator rule); the builder flags the extension as a separate un-run step rather than assuming it.

## 6. Scope discipline — CONFIRMED

- **No v1.1 file modified:** the v1.1 workdir (`/home/openclaw/muse-work/zk0-study001-twin-20261122/`) contains **zero files with mtimes on 2026-11-23**; the envelope scene copies live in the v2 workdir's `scenes-v2/`, as reported.
- **No policy rollout ran in Part A:** after 03:30 CT on 2026-11-23 the only new files in the v2 workdir are the envelope batch records/logs, the two guard rundirs, the `scenes-v2/` copies, and `controls/controls-manifest.json` (04:35 CT). No rollout output, recording, or control-run directory exists; the manifest itself asserts no control run has been executed under it.

## Standing note (not a finding against Part A)

The off-box guard mirror remains broken in this environment (ssh to the Muse VM refused — no sshd): both builder batches and this verifier's batch set `JOB_GUARD_MIRROR_TARGET`, logged MIRROR FAIL rc=255, and used no override. Telemetry exists on-box only. This is the same open infrastructure gap recorded in every prior v2 verification; it does not affect any Part A count, hash, or verdict.

**Verdict: CERTIFIED.** The envelope union register (9/20), the combined validated denominators (20/40), the controls manifest pins, and the 435-rollout Part B matrix may be cited as settled.
