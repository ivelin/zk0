# Twin-Suite v2 — Independent Verification: Finger-Pad Fix

**Date:** 2026-11-22 · **Verifier:** Muse subagent, separate context, raw artifacts only
**Subject:** `fingerpad-fix-v2.md`, DGX `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/`
**Governing:** twin-suite-v2 preregistration §4; Amendment 2 §A2.2. Prior blocker record: `verification-v2-rollout-harness.md` (blocker CONFIRMED).

## Verdict: CERTIFIED-WITH-FINDINGS

Every decisive claim in the fix report reproduces — most of them **exactly**, digit for digit, in verifier re-runs on the DGX (CPU-only, `nice -n 10`, MuJoCo 4.0.1 sim venv, scripts run unmodified; no DGX training disturbed, no workdir written). The findings below concern the completeness/wording of the pad-sourcing lineage (§1) and run-to-run variance in the scripted-pass record (§4); none overturns the fix's central claim or its two honest limits.

**Central claim, as verified:** on the padded model, a fingers-down, face-centred scripted executor pinch-lifts the 30 mm cube and holds it ≥2 s by finger contact alone at descent heights 0.000 m and −0.005 m; at +0.005 m it lifts but drops in hold. The site-centred approach still fails. The full scripted stack pass is **not** proven (place blocked by the wrist_flex limit).

## 1. Pad sourcing — CONFIRMED with two lineage findings

- **Menagerie source exists as claimed:** `/home/openclaw/muse-work/zk0-sim-v01-20261115/menagerie/robotstudio_so101/so101.xml` (mtime 2026-11-15, md5 `1470b615…`) contains `fixed_jaw_box1..7`, `moving_jaw_box1..3`, the tip capsule (`fixed_jaw_box3`), and the split gripper-part collision meshes (`wrist_roll_follower_so101_gripper_part0_v1.stl`, `moving_jaw_so101_gripper_part{0,1}_v1.stl`).
- **Verbatim port verified for every ported geom:** all `fixed_jaw_box1..7` and `moving_jaw_box1..3` pos/size values in `scenes-v2/arm/so101_new_calib_padded.xml` match the Menagerie file exactly; only the class changed (`collision_gripper` → `collision`, group 3), as declared. The part meshes are present, with `maxhullvert="64"` as upstream.
- **Authored pads match the report's table exactly:** `finger_pad_fixed` box centre (−0.0105, 0, −0.080), half-size (0.0025, 0.010, 0.023); `finger_pad_moving` centre (−0.009, −0.048, 0.01875), half-size (0.003, 0.030, 0.006), `euler="0 0 −0.13"`. The in-file comments carry the same labels as the report: fixed face plane / z-span / y-half **[MEASURED]**, thickness **[ESTIMATE]**; moving y-span / z / half-thickness **[MEASURED]**, face plane **[ESTIMATE]**, pre-tilt **[MEASURED-EST]** with the −0.35 → +0.22 correction narrative. No unlabelled authored dimension found.
- **Full-mesh finger geoms visual-only in the copy only:** in the padded copy, both original full-mesh collision geoms (body `gripper`: `wrist_roll_follower_so101_v1`; body `moving_jaw_so101_v1`: `moving_jaw_so101_v1`) carry `group=2 contype=0 conaffinity=0`. The vendored original still carries them as plain `class="collision"` (grep-verified).
- **No v1.1 or vendored file modified:** vendored `so101_new_calib.xml` mtime 2026-11-15 16:10, md5 `64092b46…`; Menagerie file mtime 2026-11-15; under the v1.1 workdir `zk0-study001-twin-20261122` the only file newer than 2026-11-22 20:00 is the v1.1 completion-report doc itself (the same spot-check result as the previous verification). The v2 scenes differ from the v1.1 scenes by exactly one line — the `<include>` swapped to the padded arm copy (diff-verified, ep411). Original `grasp_probe.py` again imports `sim_runner` (the reported sed accident was indeed restored); all `*_padded.py` probes import `sim_runner_padded`.

**Finding F1 — the port is not complete, and the report does not say so.** The Menagerie model also carries six tip spheres — `fixed_jaw_sph_tip1..3` and `moving_jaw_sph_tip1..3` (r = 0.75 mm) — which are **absent** from the padded copy and unmentioned in the report. The upstream part-mesh geoms are also unnamed upstream (`collision_gripper_mesh`) and were given names in the copy (`fixed_jaw_mesh`, `moving_jaw_mesh0/1`) — harmless, but "ported verbatim (pos/size)" should have been scoped to the boxes/capsule/meshes actually ported, with the sphere omission declared. Impact assessment: 0.75 mm tip spheres sit at the finger tips, outside the pad-face contact zone exercised by the acceptance battery; no battery result is plausibly affected. It is a lineage-completeness defect, not a physics one.

**Finding F2 — one lineage sentence is overstated.** The report's claim that "body frames, geom positions and joint ranges in the Menagerie file match the vendored file element-for-element for both finger bodies" does not hold for the `gripperframe` site: Menagerie pos is (0.012, −0.000218, −0.098127) vs vendored/padded (−0.0079, −0.000218121, −0.0981274), and several body quats differ in representation between the two files. The padded copy correctly inherits the **vendored** frames (the right choice — it is a copy of the vendored model), so there is no functional impact; the sentence as written is inaccurate and should be narrowed to the finger geom pos/size and joint ranges, which do match.

## 2. Acceptance battery re-run — CONFIRMED, exact reproduction

Verifier re-ran `harness/fingerpad_probes.py` unmodified (xoff −0.020). Output, verbatim:

```
HEIGHT +0.005: shove_mm 11.0 grip_qpos 0.096 close_contacts [gripper, moving_jaw_so101_v1, world] lift_z 0.076 hold2s_z 0.015 hold_contacts [world] PINCH_LIFT_HOLD FAIL
HEIGHT +0.000: shove_mm 12.5 grip_qpos 0.095 close_contacts [gripper, moving_jaw_so101_v1, world] lift_z 0.082 hold2s_z 0.081 hold_contacts [gripper, moving_jaw_so101_v1] PINCH_LIFT_HOLD PASS
HEIGHT -0.005: shove_mm 13.9 grip_qpos 0.098 close_contacts [gripper, moving_jaw_so101_v1] lift_z 0.097 hold2s_z 0.101 hold_contacts [gripper, moving_jaw_so101_v1] PINCH_LIFT_HOLD PASS
```

Every number matches the report's §4(b) table (shove 11.0/12.5/13.9 mm; stall 0.096/0.095/0.098 rad; lift 0.076/0.082/0.097 m; hold 0.081/0.101 m at the two passing heights). At both PASS heights the hold contact set is exactly `[gripper, moving_jaw_so101_v1]` — finger contact alone, moving jaw present, no `world`. At +0.005 m the cube lifts (0.076 m) and drops during hold (hold z 0.015 m, contacts `[world]`) — exactly the reported "lifts, not held" FAIL. Welds: the scripted-pass runner asserts `eq_active` never becomes true at any step (code-read, and the assertion did not fire in the §4 re-run); the probe scripts never enable welds; verifier independently confirmed the compiled padded ep411 model has `neq = 8`, all `eq_active` false at init.

Supporting probes also re-run, both matching the report:
- **(a) Recorded-pose** (`grasp_probe_padded.py`): no cube contact at the recorded pose; after recorded close + lift, cube z_max 0.0149 m, contacts `[world]` — left behind, unchanged from pre-fix, as reported.
- **(c) Close-sweep** (`grasp_probe2_padded.py`): cube starts in contact with **both** jaws; moving jaw registers at start and at grip 0.162 rad; total cube motion y 0.0197 → 0.0120, x 0.2391 → 0.2460 (≈2–7 mm vs the pre-fix 20 mm sweep); no lift. As reported.

**Pre-fix failure of the same probes: CONFIRMED from the prior verification's records.** `verification-v2-rollout-harness.md` §4 independently re-ran the unpadded battery: recorded-pose probe — no contact, z 0.0149 m, `[world]` only; scripted IK-centred probe — cube shoved −13.5 mm, moving jaw never in any contact set, no lift; close-sweep log — cube swept y 0.030 → 0.010, never lifts. The before/after contrast in the fix report is accurate.

## 3. The honest failures — CONFIRMED

- **Site-centred probe 3 still fails on the padded model.** Verifier re-ran `grasp_probe3_padded.py` unmodified: on close the cube moves x 0.2407 → 0.2274 m (**−13.3 mm shove**; report: −13 mm), the moving jaw **never** appears in the cube's contact set (contacts only `[gripper, world]`, then `[world]`), and the cube ends at z 0.0149 m — no lift. Exactly the reported failure class.
- **Gripperframe site offset re-measured.** In a verifier scratch script, at the face-centred grasp pose (grip 0.095 rad) on the padded ep411 nominal model: site at (0.2177, 0.0294, 0.0144); pad-face centres at (0.2151, …, 0.0326) fixed / (0.2413, …, 0.0406) moving; site-to-pinch-midpoint distance **24.5 mm**; site-to-cube-centre distance **21.4 mm**. The report's "≈20 mm off the pinch centre" is confirmed in magnitude (20–25 mm depending on the exact centre definition).

## 4. Scripted pass record (ep411 nominal) — CONFIRMED, with variance noted

Verifier re-ran `harness/scripted_pass.py` unmodified:

- **Grasp:** grip stall **0.095 rad**, contacts `[gripper, moving_jaw_so101_v1, world]` — as reported.
- **Lift/carry:** cube_a carried to **z 0.0826 m** (report: 0.082 m) with contacts `[gripper, moving_jaw_so101_v1]` only — pinch-grasp solvability in the gate twin without any weld: **proven**, independently.
- **Place:** the arm did not reach the place site — achieved (0.083, −0.0397, 0.0456) vs target (0.0581, −0.0181, 0.0566), a ~34 mm shortfall in this run (report: ≈25 mm); cube_a was nevertheless set down in contact with cube_b's top (z 0.0375 m, contacts include `cube_b`; report: z 0.033–0.037 across its runs) and toppled on release.
- **Final score, this re-run: `sim_success` FALSE, `secondary_pass` TRUE, `solver_failure` FALSE** — identical to the report's best-documented run.
- **Joint limits from the model file:** `wrist_flex` range in the padded model is [−1.65806, +1.65806] rad = **±95.0°** (verifier-computed). The builder's specific IK branch demanded ≈107°; the verifier's own aligned-orientation IK to the same place target converged (err 1.9 mm) only at wrist_flex **119.7°** — a different branch, the same conclusion: the aligned place pose is kinematically reachable only beyond the ±95° limit, so the actuated arm stalls short. The place failure is an arm-workspace/executor limit, not finger geometry — as reported.

**Finding F3 — final resting poses are not run-stable.** The verifier's re-run ends with cube_a at (0.0975, −0.0426, 0.0149) and cube_b pushed to (0.0628, −0.0196, 0.0149), vs the report's recorded (0.063, −0.050, 0.015) / (0.064, −0.015, 0.015). The score triple, the grasp/lift numbers, and the failure mode are identical; the post-topple resting poses differ (contact-rich topple dynamics). The report's §5 numbers should be read as one documented run, not a deterministic record — which is how the report presents them ("best-documented run"), so this is a caveat, not a discrepancy.

## 5. Regression smoke — CONFIRMED

Read directly from `smoke/base_final/rollouts.jsonl` (unpadded) and `smoke/padded_final/rollouts.jsonl` (padded), side by side:

| Episode / variant | Unpadded: success / secondary / solver / horizon | Padded: success / secondary / solver / horizon |
|---|---|---|
| 411 nominal | F / **T** / F / 296 | F / **F** / F / 296 |
| 411 v0 | F / F / F / 296 | F / F / F / 296 |
| 453 nominal | F / T / F / 289 | F / T / F / 289 |
| 453 v0 | F / T / F / 289 | F / T / F / 289 | Totals: **success 0/4 in both, solver failures 0/4 in both, horizons identical** (296/296/289/289 = recorded length + 20 settle). Padded wall times 61.6/71.7/76.3/64.2 s match the report's table. Guard record (`smoke/guard-padded1/guard/`): exit code 0, `SMOKE_PADDED_DONE` in `command.out`. Padded records' `actuator_info.finger_geoms` lists the upstream jaw geoms + both pads, as reported.

**The one difference is exactly as reported:** ep411 nominal secondary flips True → False, and in both runs both cubes finish essentially unmoved at their start poses (final positions agree to <1 mm between suites). So the flip is in the near-miss/secondary criterion path, not in object outcomes. The report's attribution — the policy now sees padded fingers in its wrist/front renders and contact dynamics changed — is **plausible and consistent with the records, but not independently proven** (no render-diff or trajectory-diff test was run by builder or verifier). It should stay labelled as the report labels it: "most plausibly". The conclusion the report draws from it — the padded model is a new suite version needing its own certification, not a silent harness tweak — is correct regardless of the attribution.

## 6. Guard mirror — BROKEN, independently confirmed

The builder's §7 note is accurate. Guard log (`smoke/guard-padded1/guard/guard.log`) records three mirror attempts to `hatch@100.64.115.122:…/guard-mirror-fingerpad`, each `MIRROR FAIL rc=255`, "ssh: connect to host 100.64.115.122 port 22: Connection refused"; telemetry survives only in the on-box guard rundir. **Verifier's own bounded probe from the DGX (this session):** TCP connect to 100.64.115.122:22 → Connection refused; `ssh -o ConnectTimeout=5 hatch@100.64.115.122` → Connection refused, rc 255. The off-box mirror path is indeed broken from the DGX in this environment (the Muse VM exposes no direct tailnet sshd; it is reached via the runtime proxy). No no-mirror override was used by the builder; the same limitation applies to this verification's smoke-free probe runs, which needed no guard (short CPU-only probes).

## Summary of findings

- **F1 (minor, lineage):** Menagerie tip spheres (`fixed_jaw_sph_tip1..3`, `moving_jaw_sph_tip1..3`) omitted from the port and undisclosed; ported mesh geoms renamed. No plausible effect on the battery.
- **F2 (minor, wording):** "match element-for-element" overstates the Menagerie↔vendored correspondence (gripperframe site pos differs); padded copy correctly follows the vendored frames.
- **F3 (caveat):** scripted-pass post-topple resting poses vary run to run; score and failure mode are stable.
- Secondary-flip attribution (§5) is plausible, labelled as such in the report, and unproven — acceptable as stated.

None of these touches the acceptance battery, which reproduced **exactly**. The fix report's verdict stands as written: grasp fixed for a fingers-down, face-centred approach only; site-centred probes still fail; full-pass solvability not proven (wrist_flex workspace limit at place).

**Verdict: CERTIFIED-WITH-FINDINGS** (F1–F3 above; recommend the builder amend §1 of `fingerpad-fix-v2.md` for F1/F2 before this battery is cited as suite-certification evidence).

---

## Delta pass — 2026-11-22 (F1/F2 fix verification; original text above unaltered)

**Verifier:** Muse subagent, separate context, raw artifacts only. **Subject:** the dated fix note appended to §1 of `fingerpad-fix-v2.md`, responding to findings F1/F2 above.

**Verdict: CERTIFIED-WITH-FINDINGS** (delta findings D1–D2 below; neither overturns the fix note's resolution).

### 1. Shipped padded model — the revert is real, CONFIRMED

- `scenes-v2/arm/so101_new_calib_padded.xml` on the DGX: md5 **`0c3f0d1535dc6ea50c6b8ef527a58cf3`** — exactly the md5 the fix note claims for the shipped copy.
- `sph_tip` count in that file: **0** (grep, no lines). The six spheres are not in the shipped model.
- **Post-revert reproduction, independently re-run by this verifier** (scripts unmodified, CPU-only, `nice -n 10`, sim venv): `fingerpad_probes.py` returned shove **11.0 / 12.5 / 13.9 mm**, grip stall **0.096 / 0.095 / 0.098 rad**, lift **0.076 / 0.082 / 0.097 m**, hold **0.015 (dropped) / 0.081 / 0.101 m**, verdicts **FAIL / PASS / PASS** — digit-for-digit the verified (no-spheres) numbers in the fix note and in §2 above. `grasp_probe3_padded.py` re-run: cube x 0.2407 → 0.2274 on close (**−13.3 mm**), contacts `[gripper, world]` only, final z 0.0149 m, no lift — as recorded.

### 2. Fix-note disclosure — CONFIRMED in substance, with D1

- **Six sphere values as ported:** the note's values match the Menagerie source exactly (verifier read from `zk0-sim-v01-20261115/menagerie/robotstudio_so101/so101.xml`): all size 0.00075; fixed pos (−0.0081, 0, −0.101), (−0.0081, 0.0035, −0.0975), (−0.0081, −0.0035, −0.0975); moving pos (−0.012, −0.078, 0.0192), (−0.012, −0.0745, 0.0225), (−0.012, −0.0745, 0.0155). The note also discloses the part-mesh renaming (`fixed_jaw_mesh`, `moving_jaw_mesh0/1`), as F1 asked.
- **Full with-spheres battery table:** present in the note, all four cases, including the **+0.005 m FAIL→PASS flip** (hold 0.015 → 0.077 m) and the −0.005 m hold move (0.101 → 0.086 m = −15 mm, arithmetic checks).
- **Revert decision + pre-stated rule:** stated explicitly (any outcome-class change or material number move → revert), and §1 above confirms the shipped model is the reverted one.
- **Finding D1 (evidence retention):** the fix builder's recorded with-spheres run outputs are **not present on the DGX** — searched the v2 workdir and `/tmp` for battery logs / trial copies / `PINCH_LIFT_HOLD` outputs; none found (only the probe script source). The with-spheres table therefore cannot be cross-checked against recorded run outputs; it stands on the note alone. The sphere *source* values and the *post-revert* numbers are independently verified (above); the with-spheres numbers themselves are unverified-but-uncontradicted. Recommend the builder retain trial-run logs on-box for any future trial-and-revert.
- **Finding D2 (minor arithmetic):** the note's "grip stall systematically +0.018…+0.021 rad" does not match its own table at −0.005 m: 0.114 − 0.098 = **+0.016 rad** (the other two cases are +0.020 each). The systematic-shift claim itself holds; the stated range is off by ~0.002 rad at the low end.

### 3. F2 correction — CONFIRMED accurate

- `gripperframe` site, read from the files: padded copy and vendored `so101_new_calib.xml` both pos **`-0.0079 -0.000218121 -0.0981274`**; Menagerie pos **`0.012 -0.000218 -0.098127`**. The fix note's superseding statement quotes both correctly.
- Jaw collision geoms match: all `fixed_jaw_box1..7` (incl. the tip capsule `fixed_jaw_box3`) and `moving_jaw_box1..3` pos/size in the padded copy equal the Menagerie values (verifier compared line by line). Joint ranges match (Menagerie carries the same ranges rounded: wrist_flex ±1.658063, wrist_roll −2.7438473…2.8412063, gripper −0.174533…1.7453292 vs the vendored/padded full-precision equivalents). The narrowed claim — jaw collision geoms + joint ranges match; frames follow the vendored model — is accurate. The original "element-for-element" sentence is properly superseded, not silently edited.

### 4. Nothing else changed — CONFIRMED

- Vendored `so101_new_calib.xml` (`zk0-sim-v01-20261115/sim/model/`): md5 **`64092b46aa7050ba44bcdb65dc0c3c62`**, mtime 2026-11-15 16:10 — unchanged.
- Menagerie source: md5 `1470b615…`, mtime 2026-11-15 — unchanged.
- v1.1 workdir (`zk0-study001-twin-20261122`): the only file newer than 2026-11-22 20:00 remains the v1.1 completion-report doc itself — same spot-check result as the original verification. No v1.1 file was touched by the fix.

**Delta verdict: CERTIFIED-WITH-FINDINGS.** The F1 resolution (disclose, trial, revert under the pre-stated rule, shipped model byte-identical to the verified one) and the F2 correction both check out against the raw artifacts; the post-revert battery reproduces exactly in this verifier's own re-run. Findings D1 (with-spheres run outputs not retained on the DGX) and D2 (stall-shift range off by ~0.002 rad at −0.005 m) are record-keeping defects only and do not affect the certified battery or the lineage disclosure.
