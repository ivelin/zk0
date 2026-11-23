# Independent Verification — Twin-Suite v2 Solvability Build (Reference Controllers + Register)

**Verifier:** independent subagent, separate context, raw artifacts only.
**Builder's report:** `solvability-v2.md` (same folder).
**Governing:** `twin-suite-v2-preregistration-20261122.md` §A2.2 (in
`../twin-suite-v2-preregistration-20261122.md` relative to the notes root:
`~/workspace/research_notes/zk0-bench003-design-20261119/`).
**Artifacts:** DGX `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference/`,
batch rundir `../runs/solvability-batch1/`; verifier scratch copy + re-runs at
DGX `/home/openclaw/muse-work/verify-solvability-20261123/` (rundir
`runs/verify-batch1/`, guard exit_code 0).
**Date:** 2026-11-23.

## VERDICT: CERTIFIED-WITH-FINDINGS

The register's counts, verdicts, provenance, joint-limit integrity, and the
ring_insert no-gate consequence are confirmed at the source, and all five
decisive re-runs reproduced the batch records exactly. Four findings correct
stated reasons/measurements without changing any verdict or count: the
block ep498 exclusion reason, the cable pad-gap figures, the ring §7
mechanism narrative's provenance, and two grasp-record counts.

---

## 1. Frozen-code single-batch provenance — CONFIRMED

- Code md5s on the DGX match the report exactly: `refcore.py`
  `96aa928072e5487b1fe1f077be293132`, `run_reference.py`
  `a9d29bdb8f07c8b55181d10295f6d457`. Code mtimes (01:35 / 01:37 CT) predate
  the batch; all 20 `results/*.json` mtimes fall inside the single batch
  window 01:41–01:57 CT, in `batch_summary.txt` order.
- Guard log (`runs/solvability-batch1/guard/guard.log`): one launch at
  2026-11-23T01:40:13-06:00, cmd = `bash …/reference/batch_dev.sh`, logonly
  mode, CPU-only. `batch_dev.sh` is a single loop invoking the same
  `run_reference.py <task> <ep> results/<task>__ep<ep>.json` (timeout 900,
  `nice -n 10`) for all 20 scenes — no per-scene scripts, flags, or parameter
  changes.
- Code read: dispatch is by **task only** (`TASKS` dict in
  `run_reference.py`); per-task constants (e.g. grasp dz, candidate sets) are
  uniform across that task's scenes. Episode numbers appear in
  `refcore.py`/`run_reference.py` **only in comments** (measured-on notes) —
  no episode-conditional logic anywhere. Scene data (object poses, fixtures
  via `twin_params`/`build_scenes`) enters only as input to the global
  controller, as claimed.

## 2. Verdict re-runs (decisive sample) — ALL CONFIRMED, records identical

Re-run in a scratch copy (md5-verified identical code) under
`launch_dgx_job.sh` + `job_guard.sh`, logonly, `JOB_LOCK=0`, CPU-only,
`nice -n 10`. **Mirror: `JOB_GUARD_MIRROR_TARGET` was set to the precedent
off-box target; the guard logged MIRROR FAIL rc=255 (ssh connection refused)
— telemetry is on-box only; no no-mirror override flag was used.** Guard
exit_code 0. For every re-run scene, the fields `task, episode, generation,
sim_success, verdict, checks, objects_final, phase_notes` are **identical**
to the original batch record:

| Scene | Batch | Re-run | Mechanism confirmed from the re-run record |
|---|---|---|---|
| block_stack 453 | PASS | PASS (identical) | cube_b stacked on cube_a, final (0.278, 0.050, 0.044); all block checks true |
| block_stack 411 | FAIL | FAIL (identical) | grasp:above unsolved at +80 mm and +50 mm, solved at +30 mm via direct-pos-only; both lift and lift:pos-only IK-unsolved; cube_b stays z 0.017, dragged to (0.065, −0.057) — near-base lift IK limit as stated |
| ring_insert 7 | FAIL | FAIL (identical) | see §3 below |
| tool_insert 683 | PASS | PASS (identical) | tool in slot (0.234, 0.059, 0.046), centre_in_slot true, axis 1.1° |
| cable_clip 2125 | FAIL | FAIL (identical) | weld-carried target segment `cable_seg_15` settled at (0.213, 0.177, 0.069) vs channel (0.206, 0.195, 0.061): Δ (+7.3, −18.2, +8.0) mm — settled out of the channel volume under chain tension, as stated (z +8 / y −19 class); `cable_seated_in_clip_closed` false, `solver_failure` false |

## 3. Ring finding — CONFIRMED IN SUBSTANCE, narrative provenance corrected (Finding F3)

From the batch records (and the identical ep7 re-run):

- **Tilt at staging:** initial `ring_tilt_deg` entries are **39.6° (ep7),
  39.0° (ep167), 39.8° (ep195)** — the rim-held ring does arrive at staging
  tilted ≈39–40°. The report's "39–50°" upper end is **not in any batch
  record**; ~50° appears only in development-era code comments.
- **Wrist limit / tilt-levelling on ep7:** the `place_scan` solution for the
  over-peg pose records `wrist_flex_deg: 95.0` — exactly the model limit
  (padded arm `wrist_flex` range ±1.65806 rad = ±95.000°, identical to the
  v1 arm). The levelling loop records one tilt entry (39.6°) and then exits:
  in the frozen code that exit happens only when the 10° levelling step's IK
  returns out-of-limits or position error > 4 mm. Tilt-levelling is
  therefore confirmed IK-infeasible **at ep7's staging pose**.
- **But the §7 narrative over-generalises from ep7 / development runs:**
  - On ep7's batch run the staging servo did **not** converge — recorded
    `ring_staging_err_mm` runs 14.5 → 15.3 mm over 10 steps. The report's
    "servo does converge planar error 12.3 → 1.9 mm (measured on ep7)"
    matches **no batch record** (12.3 appears in batch only as a tilt value
    on ep167). Convergence does occur on other batch scenes (ep167:
    5.5 → 0.3 mm; ep195: 6.5 → 1.2 mm).
  - Tilt-levelling is not universally IK-infeasible: on ep167 it succeeded
    in steps (39.0° → 29.5° → 20.5° → 12.3° → 3.6° → 1.1°) and the seat
    **still** failed (final planar error 11.3 mm, ring z 0.119 — descend
    blocked short of the seat). On ep195 levelling progressed 39.8° → 13.9°
    before an IK step failed.
- Net: the register conclusion — scripted seat infeasible for this
  controller on all 4 scenes, 0/4 validated — is fully supported. The
  single-mechanism story in §7 (tilt + levelling infeasibility as *the*
  blocker) is ep7's story; ep167 shows the seat fails even with the ring
  levelled, so at least one further blocker (descend alignment) exists.
  §7 should be corrected to attribute its measurements per scene.

Also noted: `do_ring` appends the phase note "seated pose reached"
unconditionally, even when the log records `ring_seated: false` (all four
ring records). Misleading note text; verdicts are unaffected because they
come from `evaluate()`.

## 4. Cable weld justification — CONCLUSION CONFIRMED, figures corrected (Finding F2)

Re-measured on the padded arm model (verifier probes
`probe_padgap2.py` / `probe_padgap3.py` in the scratch dir), gripper joint
range −0.17453…1.74533 rad, cable Ø = 5 mm (`twin_params.CABLE_R = 0.0025`):

- Pad-centre separation along the face normal: **77.3 mm at +1.45 rad**
  (matches the report) and **13.5 mm at the joint limit** (report: 13.7 mm
  at −0.17 rad — consistent).
- **Face gap through the pinch centre** (line through the pinch centre
  along the face normal, OBB slab intersection): **6.95 mm at the
  controllers' commanded close (−0.16 rad) and 6.15 mm at the joint limit**
  — not the report's ≈8 mm (which subtracts only the nominal half-thicknesses,
  2.5 + 3.0 mm, assuming parallel faces; the moving pad is angled, so its
  effective half-extent along that line is larger).
- **True minimum pad-to-pad distance** (`mj_geomDistance`): **0.21 mm at
  −0.16 rad and −0.86 mm at the limit** — the pads effectively **meet at one
  edge** (closest point z ≈ 0.098). The report's phrase "minimum achievable
  face gap ≈ 8 mm" is therefore wrong as a minimum over the faces.
- **Judgement:** at the pinch centre — the only place this controller (or a
  policy imitating it) pinches — the gap at full close is 6.15 mm > Ø5 mm,
  so a centred pad pinch of the cable cannot form; the weld-device route
  remains justified. The margin is **1.15 mm, not ~3 mm**, and an
  edge-nip at the touching corner is not excluded by geometry — the
  justification should be restated on the pinch-centre gap (6.15 mm > 5 mm),
  not on a fictitious 8 mm minimum.
- **Criterion:** `refcore.Ref.evaluate()` calls v1.1 `replay.evaluate()`
  verbatim (no local copy, no modification). Its cable branch is exactly
  the unmodified v1.1 geometric criterion: a segment centre inside the
  channel volume (clip-frame x ±12 mm, y within channel half-width + 2 mm,
  z within ±12 mm of channel z) **and** in contact with `clip_block`, with
  solver-failure excluded from success. Confirmed in the v1.1 source and in
  the recorded checks (`cable_seated_in_clip_closed`, `solver_failure`).

## 5. Place-reachability resolution — CONFIRMED

- **Limits never relaxed:** the only `jnt_range` use in the code is reading
  it into `Ref.lim`; the IK clips every iterate to those limits
  (`refcore.py` line 184) and rejects solutions outside them (`within`
  check, line 192). No assignment to `model.jnt_range` exists. The padded
  arm's joint ranges are **identical** to the v1 arm's (diff of all
  `range="…"` attributes: no differences), including `wrist_flex`
  ±1.65806 rad and `gripper` −0.17453 rad.
- **Role inversion as described:** `do_block` selects the cube closer to
  the arm base (xy norm) as the moving cube and places it onto the far
  cube. **`place_scan`** tries the caller's orientation candidates in
  preference order (yaws → tilts), each gated on within-limits, ≤4 mm /
  ≤6°, and the torque check, then falls back to the free-orientation
  branch (position-only IK at the place point, iterated object offset,
  full polish) — exactly §2's mechanism. Batch place solutions record
  `wrist_flex_deg` 71.6–77.3°, inside limits.

## 6. Register accuracy (§8) — COUNTS CONFIRMED; one reason wrong (Finding F1), two counts off (Finding F4)

Recomputed from all 20 result files:

| Task | Attempted | Validated (files) | Register |
|---|---|---|---|
| block_stack | 4 (411, 453, 489, 498) | **2** (453, 489) | matches |
| eye_drops_to_basket | 4 (1630, 1746, 1783, 1798) | **3** (1630, 1746, 1798) | matches |
| ring_insert | 4 (7, 33, 167, 195) | **0** | matches |
| tool_insert | 4 (683, 697, 852, 883) | **1** (683) | matches |
| cable_clip | 4 (1969, 2020, 2125, 2130) | **2** (1969, 2020) | matches |
| **Total** | **20** | **8/20** | matches |

- **GENERATION marking:** every one of the 20 records carries
  `"generation": "ORACLE-GUIDED"` (written by `Ref.record()`), and the
  report/register mark every row ORACLE-GUIDED per A2.2 route (b). ✓
- **Denominator consequences:** validated denominators stated per task
  (2/4, 3/4, 0/4, 1/4, 2/4; total 8/20 vs nominal 20 dev / 96 full), and
  ring_insert's consequence is stated correctly per A2.2(iii): **no
  validated scenes → no gate on this evidence**. ✓
- **F1 — block ep498 exclusion reason is wrong in §6/§7/§8.** The report
  says the stack "settled ~8.1 mm off … outside the 25%-inset footprint
  criterion (allowance ≈7.5 mm)". The record says otherwise:
  `within_inset_footprint: true`, `contact_with_lower: true`,
  `no_other_support_contact: false`, `upper_contacts: [cube_b,
  moving_jaw_so101_v1]`. The v1.1 criterion is per-axis
  (|Δx|, |Δy| ≤ (0.030/2)·0.75 = **11.25 mm**, not 7.5 mm); the final
  offset is (7, 4) mm per axis (8.1 mm radial), i.e. **inside** the
  footprint. The actual failure is the **moving jaw still in contact with
  the upper cube at settle** (the release-retreat move is logged
  `solved: false`). Verdict FAIL and the exclusion itself are correct;
  the logged reason must be corrected, because §8 is the register of
  record for the full 96-scene run.
- **F4 — grasp-record counts:** §7's "16 grasp records" for eye ep1783
  and "17 grasp records" for tool ep883 do not match the batch records,
  which contain **13** and **16** grasp entries respectively (all with
  stall −0.159 rad, as stated). Presumably development-run counts;
  immaterial to the diagnoses (displacement finals match the records:
  carton (0.432, −0.145); tool (0.315, 0.000)).
- All other §7 diagnoses spot-checked against records match: block 411
  (above), eye 1783 final, tool 852 lift z 0.002→0.047 then final on mat
  (0.065, 0.041), tool 697 final (−0.012, 0.130), cable 2130 target
  segment Δx −17 mm / Δz +5 mm vs channel, ring 33 lift failure
  (z 0.037, board contact).

## 7. Execution / guard notes (verifier's own runs)

- Builder batch and verifier batch both ran logonly under the guard on
  the shared DGX, CPU-only; the builder's guard log shows preflight
  load1 7.65 and log-only WOULD-KILL temperature lines from other
  tenants' heat — consistent with §9, no action taken.
- **Mirror failure is real and shared:** both batches set
  `JOB_GUARD_MIRROR_TARGET` and both logged MIRROR FAIL rc=255 (ssh to
  the Muse VM refused — no sshd). Telemetry exists on-box only
  (`runs/solvability-batch1/guard/`, `verify-solvability-20261123/runs/
  verify-batch1/guard/`). No override flag was used in either batch.
  This remains an open infrastructure gap, flagged for the parent.

## Required corrections before the register is cited as settled

1. §6/§7/§8: replace ep498's reason with the recorded one (jaw contact at
  settle; footprint satisfied, per-axis allowance 11.25 mm).
2. §1/§5: restate the cable gap as pinch-centre gap 6.15 mm at the joint
  limit (6.95 mm at commanded close) vs Ø5 mm; drop "minimum face gap
  ≈8 mm" (pads meet at an edge; true minimum ≈0 mm).
3. §7 ring: attribute measurements per scene — ep7 staging servo did not
  converge (14.5→15.3 mm) and levelling was IK-infeasible there
  (wrist_flex 95.0°); ep167 levelled to 1.1° yet still failed the seat;
  batch tilts are 39.0–39.8°.
4. §7: correct grasp-record counts (eye 1783: 13; tool 883: 16) or mark
  them as development-run counts.

None of these change a verdict, a count, the validated denominators, or
the ring_insert no-gate consequence.

---

# Independent Verification — Revision 2 / rev2.1 (§11 of solvability-v2.md)

**Verifier:** independent subagent, separate context, raw artifacts only
(second verification round; the sections above are the original
verification of §1–§10 and are unaltered).
**Scope:** §11 only — the ONE global controller revision round, the
rev2.1 register (§11.5), the ring pendulum finding (§11.4.1), the
tool ep683 regression (§11.4.2), guard discipline, and the register
question under Amendment 2 §A2.2.
**Artifacts:** DGX `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference/`
(`results/`, `results2-canonical/`, `results3-canonical/`; guard logs in
`../runs/solvability-batch2/`, `../runs/solvability-batch3/`); verifier
re-runs at DGX `/home/openclaw/muse-work/verify-rev21-20261123/`
(rundir `runs/verify-rev21/`, guard exit_code 0).
**Date:** 2026-11-23.

## VERDICT: CERTIFIED-WITH-FINDINGS

The rev2.1 register counts (10/20 per-version), provenance, the
one-call bug fix, the ep683 regression, and the ring pendulum mechanism
are confirmed at the source, and all five decisive re-runs reproduced
the canonical records **byte-identically**. Two findings correct prose
without changing any rev2.1 verdict or count (the §11.2 cross-batch
variance note, and the repeated 39–50° one-point ring tilt range). On
the register question, this verifier **rules the UNION register
permitted under A2.2** — see §R5; the validated denominator is
therefore **11/20**, not 10/20.

## R1. Provenance and code integrity — CONFIRMED

- Frozen code md5s re-verified on the DGX, before and after this
  verifier's own runs: `refcore.py`
  `96aa928072e5487b1fe1f077be293132`, `run_reference.py`
  `a9d29bdb8f07c8b55181d10295f6d457` — unchanged.
- New code md5s exactly as reported in §11: `refcore2.py`
  `ee9ea862d6f4b3f8b133da0f0ad43c33`, `run_reference2.py`
  `24c8f7636da79e644adb9be89bda65d1`, `run_reference3.py`
  `22b4b758d67933ac05fc8ce892262485`.
- **No joint-limit relaxation, no criteria change:** in
  `refcore.py` → `refcore2.py` the limit machinery is identical —
  `self.lim` is read from the model's own `jnt_range` in both, with no
  assignment to `jnt_range` anywhere; `evaluate()` is byte-identical
  in both files and still delegates verbatim to v1.1
  `replay.evaluate()`. The full `refcore` diff is the G1–G5 changes
  (staged close timing, hold −0.04, slower execution, stepped lift,
  slower release) plus headers — no limit, torque-gate, tolerance, or
  criterion line is touched. The `run_reference` diff is G6–G9 plus
  headers; episode numbers appear only in comments (measured-on
  notes), no episode-conditional logic in any of the three new files.
- **The rev2.1 bug fix is exactly one call — CONFIRMED by diff:**
  `run_reference2.py` → `run_reference3.py` differs only in the header
  comment, the usage string, and the single `final_servo` rescan call:
  `place_scan(obj_tgt, o_body, f"servo-rescan {name}")` →
  `place_scan(obj_tgt, o_body, [Rt] + yaw_candidates() +
  tilt_candidates(), label=f"servo-rescan {name}")`. No parameter,
  strategy, limit, or criterion changed.

## R2. Re-runs under rev2.1 (decisive sample) — ALL IDENTICAL

Re-run in verifier scratch under `launch_dgx_job.sh` + `job_guard.sh`,
logonly, `JOB_LOCK=0`, CPU-only, `nice -n 10`, running the DGX's own
`run_reference3.py` with output to scratch. Guard exit_code 0; mirror
set to the precedent target and logged MIRROR FAIL rc=255 (see R6).
Every re-run record is **byte-identical (md5-equal) to its
results3-canonical record** — verdicts and all key fields:

| Scene | Canonical (rev2.1) | Re-run md5 = canonical md5 |
|---|---|---|
| block_stack 411 | PASS (stepped lift, stacked) | `ea436e54…` ✓ |
| block_stack 498 | PASS (stack inside footprint criterion) | `f5b61894…` ✓ |
| cable_clip 2130 | PASS (`cable_seated_in_clip_closed` true, solver_failure false) | `06f372c8…` ✓ |
| tool_insert 683 | FAIL (grasp infeasible in all yaw branches; tool ends (0.0733, −0.0265)) | `18ba848b…` ✓ |
| ring_insert 33 | FAIL (re-grasp tilts 93.6° / 87.3° / 95.9°; final planar err 29.1 mm) | `840fa80b…` ✓ |

## R3. The regression is real and honestly reported — CONFIRMED, one prose correction (Finding RF1)

- **tool 683, frozen set:** `results/tool_insert__ep683.json` is PASS —
  centre_in_slot true, axis 1.1°, tray contact true, tool in slot at
  (0.234, 0.059, 0.046) (also re-run PASS, identical, in the original
  verification §2 above).
- **tool 683, rev2.1:** FAIL in both canonical batches and in this
  verifier's re-run — phase note "grasp of tool infeasible in all yaw
  branches", grasp stall entries at −0.16 rad (full close, no pinch),
  tool ends displaced at (0.0733, −0.0265, 0.0048), all slot checks
  false. The regression is in the records of both sets, exactly as
  §11.3/§11.4.2 report it, including the plain statement that the
  global changes "traded the tool task backwards".
- **ep697 / ep852 knife-edge characterization — fair in substance:**
  ep697's forming pinch sits at stall −0.039 rad against the −0.16 rad
  full-miss signature, and its rev2.1 failure is genuinely the new mode
  (placed in slot: centre_in_slot ✓, tray contact ✓, axis 82.4°);
  ep852 reached the (buggy) servo in rev2 and lost the tool in carry
  in rev2.1, with near-identical final axis (74.6° in both). These are
  marginal pinches by the records.
- **RF1 — §11.2's cross-batch variance note is inaccurate as written.**
  It claims the two canonical batches "agree on 18/20 scenes" with
  ep697 an exception whose "grasp formed only in rev2.1". The
  canonical record files in fact agree on **19/20**: ep697's record is
  byte-identical across `results2-canonical/` and
  `results3-canonical/` (its grasp formed in **both**, stall −0.039).
  The only differing record is ep852 (rev2 ERROR via the latent
  `final_servo` bug → rev2.1 FAIL, with the fixed servo-rescan logged
  `all_candidates_infeasible`). The note should be corrected to 19/20
  with ep852 the sole exception; any 697 divergence belongs to
  development iterations, not the canonical batches. (rev2's "10/20"
  is correct as a pass count, with ERROR 852 counted as not-a-pass.)

## R4. Ring pendulum mechanism — IN THE RECORDS (one range corrected, Finding RF2)

- **Two-point equator re-grasp tilts are recorded, not prose:** the
  `ring_regrasp` test-lift entries in `results3-canonical/` read
  92.5° (ep7), **93.6° / 87.3° / 95.9°** (ep33), 92.4° (ep167), 92.0°
  (ep195) — the claimed 87–96° pendulum range is fully supported, and
  the ≤12° acceptance threshold in G8 was never met by a formed
  re-grasp on ep7/33/195 (ep167's later 0.4° note follows failed
  re-grasp attempts and a fallback grasp, not an accepted re-grasp).
- **RF2 — the one-point comparison range repeats an earlier
  overstatement:** §11.4.1's "one-point rim pinch's 39–50°" — the
  frozen batch records' initial `ring_tilt_deg` values are 39.6°
  (ep7), 39.0° (ep167), 39.8° (ep195), as already corrected in the
  original verification (Finding F3 above); the ~50° upper end is in
  no batch record. The mechanism conclusion is unaffected: 39–40°
  one-point vs 87–96° two-point still falsifies the symmetric-grip
  hypothesis, and ring_insert stays 0/4, excluded, no gate (A2.2).

## R5. RULING ON THE REGISTER QUESTION — UNION PERMITTED under A2.2

**Question:** may the validated register take the union across
controller versions (frozen: tool 683 ✓; rev2.1: block 411/498 ✓,
cable 2130 ✓) = block 4/4, eye 3/4, ring 0/4, tool 1/4, cable 3/4 =
**11/20**, with the validating controller version recorded per scene?
Or does the pre-registration require a single controller set
(rev2.1 canonical = 10/20, tool 0/4)?

**Ruling: the UNION is permitted. The validated register is 11/20.**

Grounds, from the A2.2 text itself:

1. **The proof standard is per-scene and existential.** "A scene
   counts toward a checkpoint's verdict only if it is validated
   solvable: at least one executed pass trajectory in the twin that
   satisfies the mechanical criteria at settle." The unit of proof is
   the scene and the single executed pass — not a batch, not a
   controller version.
2. **The executor class is open, the executor identity is not the
   claim.** Route (b) admits "a reference/scripted controller" without
   version qualification, and A2.2 states why: "solvability is a
   property of the scene, not a policy measurement, and no gated
   claim rests on the reference executor's skill." A scene proven
   solvable by one admissible reference controller is solvable; a
   later controller's failure on it is evidence about that
   controller, exactly as §7 of the builder's report already reasons
   for ep411 under the frozen executor ("not evidence the scene is
   unsolvable").
3. **The only executor restriction in A2.2 does not bite here.**
   Denominator independence bars the gated checkpoint, candidates,
   and the §6 controls from serving as executors. Both the frozen and
   rev2.1 reference controllers are admissible route-(b) executors;
   neither is anything the gate judges or calibrates against.
4. **The exclusion clause confirms it.** "Scenes **without an executed
   pass** … are excluded from denominators." Tool 683 has an executed
   pass — frozen batch, independently re-run PASS-identical in the
   original verification. It is not a scene without an executed pass,
   and excluding it would contradict the clause's own test.
5. **Nothing elsewhere in the pre-registration imposes a single
   controller set.** §11 change control versions the *suite* (scenes,
   criteria, envelope, bars); neither controller revision changed any
   of those (verified in R1: criteria and limits identical). §11.5's
   sentence "registers are per controller version and do not carry
   scenes across versions" is a builder assertion in the report under
   verification; it appears nowhere in the pre-registration and
   cannot override A2.2's existential standard.

**Conditions of the ruling** (all satisfiable from existing records):
the union register must record the validating controller version per
scene (tool 683: frozen `refcore.py`/`run_reference.py`; block
411/498 and cable 2130: rev2.1; all other validated scenes validate
under both versions); both per-version registers (§8: 8/20 frozen;
§11.5: 10/20 rev2.1) remain on record as executor-performance
records. One caution, stated for the record and not as a bar: the
union is sound because solvability is existential, but it must not
become executor-shopping — each additional controller version added
to a union should be a registered global revision round like this
one (no per-scene patches, independently verified), so the union
stays a proof set, not a search over executors. For the full
96-scene validation, whichever controller is carried forward still
runs unchanged; scenes it cannot validate may additionally be
validated by any other admissible executor under this same rule,
logged per scene.

**Union validated register (dev subsets): block_stack 4/4,
eye_drops_to_basket 3/4, ring_insert 0/4, tool_insert 1/4,
cable_clip 3/4 — total 11/20.** ring_insert's consequence is
unchanged: no validated scenes under any version → no gate on this
evidence (A2.2).

## R6. Guard discipline — CONFIRMED

- Both canonical batches launched via `launch_dgx_job.sh` +
  `job_guard.sh` (batch scripts `batch_canonical2.sh` /
  `batch_canonical3.sh` invoke one frozen runner for all 20 scenes;
  guard logs show the single launch commands, logonly, CPU-only,
  `JOB_LOCK=0`), guard exit_code 0 for both.
- **Mirror FAIL logged, no override:** each guard log records
  **9 × MIRROR FAIL rc=255** (ssh connection refused — the Muse VM
  exposes no sshd; same precedent target as batch1). No
  `JOB_GUARD_ALLOW_NO_MIRROR` override appears in either guard log,
  and §11 states none was used. Batch3's guard log contains **zero
  WOULD-KILL lines**, as §11 reports.
- This verifier's own re-run batch followed the same discipline
  (mirror target set, MIRROR FAIL rc=255 logged 4×, no override,
  exit_code 0). The off-box mirror path remains broken in this
  environment — an open infrastructure gap, unchanged since the
  original verification.

## Required corrections (prose only; no verdict or count changes)

1. §11.2: correct the cross-batch note to 19/20 agreement, ep852 the
   sole differing record; ep697's grasp formed in both canonical
   batches (RF1).
2. §11.4.1: correct the one-point ring tilt range to the recorded
   39.0–39.8° (RF2; same correction as original Finding F3).

Neither correction touches the rev2.1 per-version register (10/20),
any scene verdict, or the ring_insert no-gate consequence. The
validated denominator, per the R5 ruling, is the union: **11/20**.

---

# Delta Verification — §12 Errata, §13 Micro-Corrections, and the Finger-Pad D2 Erratum

**Verifier:** independent subagent, separate context, raw artifacts only
(delta pass; all sections above are the original verification and the
Revision 2 verification, and are unaltered).
**Scope:** only the appended material — §12 (errata F1–F4) and §13
(RF1/RF2 + register ruling) of `solvability-v2.md`, and the D2 erratum
paragraph appended to `fingerpad-fix-v2.md`.
**Artifacts:** local docs in this folder; DGX raw records at
`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference/`
(`results/`, `results2-canonical/`, `results3-canonical/`), read-only
via `dgx.sh`.
**Date:** 2026-11-23.

## VERDICT: CERTIFIED

Every delta claim checks out exactly against the raw records and the
files' own prefix hashes. No discrepancy was found in any of the eight
items. No verdict, count, or register entry in the earlier sections
changes as a result of this pass.

## 1. Append integrity — CONFIRMED (both passes)

- §12 pass: `head -n 560 solvability-v2.md | md5sum` =
  `ac895a50777c0c290d2064aaee3335bf` — matches the stated pre-§12 md5
  (`ac895a50…`). Line 560 is the last line of §11; §12 begins at
  line 564. §§1–11 are byte-identical to the pre-errata file.
- §13 pass: `head -c 42041 solvability-v2.md | sha256sum` =
  `af4cb7f7c982e5a2f4f6cc45eabda0e0651436232ccd1292a87772a6b1c5b81b` —
  matches the stated prefix hash (`af4cb7f7…`). The file is now
  **688 lines**, as stated; §13 begins at line 674. §§1–12 are
  unaltered.

## 2. F1 (ep498) — CONFIRMED

Batch-1 record `results/block_stack__ep498.json`: checks are
`within_inset_footprint: true`, `contact_with_lower: true`,
`no_other_support_contact: false`, `upper_contacts: [cube_b,
moving_jaw_so101_v1]`; the log's `release:retreat` move records
`solved: false`; verdict FAIL. The erratum's corrected reason (jaw
still in contact at settle after an unsolved retreat; footprint
satisfied, per-axis allowance (0.030/2)·0.75 = 11.25 mm) matches the
record exactly. The exclusion stands, on the corrected reason.

## 3. F2 (cable gap) — CONFIRMED

The erratum's figures — pinch-centre face gap **6.15 mm at the joint
limit** (−0.1745 rad), **6.95 mm at commanded close** (−0.16 rad),
vs cable **Ø5 mm** (`CABLE_R = 0.0025`, re-confirmed in the DGX
source), margin **1.15 mm**, and true minimum pad-to-pad distance
≈0 mm (0.21 mm / −0.86 mm) — are identical to this verifier's own
probe values recorded in §4 of the original verification above
(`probe_padgap2.py` / `probe_padgap3.py`). No inconsistency.

## 4. F3 (ring provenance split) — CONFIRMED

Batch-1 ring records, read directly:

- ep7: staging errors **14.5 → 15.3 mm** over 10 steps (no
  convergence); single tilt entry **39.6°**; `place_scan` over-peg
  `wrist_flex_deg: 95.0` (the model limit); `ring_seated: false`.
- ep167: staging **5.5 → 0.3 mm**; tilts **39.0° → 29.5° → 20.5° →
  12.3° → 3.6° → 1.1°**; final planar error **11.3 mm**, ring z
  0.119 — seat still failed with the ring levelled.
- ep195: staging **6.5 → 2.7 → 1.2 mm**; tilts **39.8° → … → 13.9°**
  before the seat sequence ends unseated.
- ep33: no staging tilt (lift failed), as the erratum states.

Batch initial-tilt range is therefore **39.0–39.8°**, as corrected.
The "12.3 → 1.9 mm" servo claim is indeed absent from the batch
records: a scan of all 20 batch-1 records finds **no**
`ring_staging_err_mm` of 1.9 anywhere, and 12.3 occurs **only** as a
`ring_tilt_deg` value on ep167. The development/batch attribution in
the erratum is accurate.

## 5. F4 (grasp-record counts) — CONFIRMED, counted by this verifier

Counting log entries carrying a `grasp` record in the batch-1 files:
**eye_drops_to_basket ep1783 = 13** grasp records (all stall
−0.159 rad); **tool_insert ep883 = 16** grasp records. Both match
the erratum (and correct §7's 16 / 17).

## 6. RF1 / RF2 (§13) — CONFIRMED

- **RF1:** md5-compared all 20 record pairs across
  `results2-canonical/` and `results3-canonical/`: **19/20
  byte-identical**; the **only** differing file is
  `tool_insert__ep852.json` (rev2 md5 `92407962…` → rev2.1
  `a4673805…`; rev2 verdict ERROR → rev2.1 FAIL). ep697 is
  byte-identical in both batches (md5 `8ac2d77ae9e0a914b905c7c0d4889e69`),
  its grasp records including the forming stall −0.039 rad in both —
  exactly as §13 states. Cross-batch agreement is 19/20.
- **RF2:** batch ring tilts are **39.0–39.8°** (§4 above), so §13's
  superseding of §11.4.1's "39–50°" with 39.0–39.8° is correct; the
  pendulum conclusion (39.0–39.8° one-point vs 87–96° two-point) is
  unaffected, as §13 states.

## 7. §13 register statement — MATCHES the R5 ruling

§13 records: union register permitted under A2.2 — block_stack 4/4,
eye_drops_to_basket 3/4, ring_insert 0/4, tool_insert 1/4,
cable_clip 3/4, **total 11/20**; validating version per scene
(tool 683: frozen set; block 411/498 and cable 2130: rev2.1; all
other validated scenes under both versions); both per-version
registers remain on record; and the caution that future union
additions must be registered global revision rounds (no per-scene
patches, independently verified). This matches §R5 above in counts,
per-scene attribution, and caution, verbatim in substance.

## 8. Finger-pad D2 erratum — CONFIRMED

- Arithmetic against `fingerpad-fix-v2.md`'s own Fix-note table:
  with-spheres minus verified grip stalls = 0.116 − 0.096 =
  **+0.020** (+0.005 m), 0.115 − 0.095 = **+0.020** (0.000 m),
  0.114 − 0.098 = **+0.016** (−0.005 m). The erratum's corrected
  range **+0.016…+0.020 rad** is right; the original
  "+0.018…+0.021 rad" was wrong at the low end, exactly as the
  delta finding D2 in `verification-v2-fingerpad.md` stated.
- Append-only: the file is now **331 lines**; line 328 is the last
  line of the original text ("…battery as the acceptance
  evidence."), lines 329–330 are blank, and the erratum is the
  single appended paragraph at line 331. Nothing earlier was edited
  in this pass.

**Delta verdict: CERTIFIED.** The §12 errata, the §13
micro-corrections and register statement, and the finger-pad D2
erratum are all accurate against the raw records; the appends are
append-only. The solvability register may be cited as settled in its
corrected form: per-version 8/20 (frozen) and 10/20 (rev2.1), union
**11/20** per the R5 ruling, ring_insert 0/4 with no gate on this
evidence.
