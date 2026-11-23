# Twin-Suite v2 — Independent Verification: Drake Port, Stage 1.2 Part 1

**Verdict: CERTIFIED-WITH-FINDINGS**

**Date:** 2026-11-22 · **Verifier context:** independent, separate from the
builder; verified from raw artifacts only (code, per-episode result JSONs,
v1.1 frozen validation records, guard/launch logs) plus two bounded
re-derivations run by the verifier in a DGX scratch directory (FK
cross-check; no-grasp tool-migration trace). CPU-only, `nice -n 19`, short
commands; no DGX training disturbed; no sustained verifier run was needed,
so no guard launch was required on the verifier side.
**Artifact class:** pre-registration §9(b) as mapped by Amendment 1 A1.4 —
environments/poses + code.
**Scope certified:** the Drake port's sanity claims and the
development-subset kinematic-replay equivalence diagnostics in
`drake-port-v2.md`. This is **not** the §3.2 certification standard
(policy-outcome agreement on the control checkpoints, bounds per §7) —
that is part 2, not started, and nothing here certifies it.

Governing inputs: `twin-suite-v2-preregistration-20261122.md` §3.2 +
Appendix A (development subset: block 411/453/489/498, eye
1630/1746/1783/1798, ring 7/33/167/195, tool 683/697/852/883).
Artifacts: DGX `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake/`
(`drake_port.py`, `drake_replay.py`, `mujoco_baseline.py`, `compare.py`,
`results/`, `results_hydro/`, `mujoco_results/`); frozen v1.1 tree
`/home/openclaw/muse-work/zk0-study001-twin-20261122/` (`param_hash`
`a4eb3f8c3611894d`, confirmed in `validation/validation_results_v11.json`).

---

## 1. Single programmatic builder, same engine-independent inputs — CONFIRMED

- `drake_port.py` is one builder class (`DrakeScene`) for all
  episodes/variants; task-specific geometry branches exist per *task*
  (block/eye/ring/tool), never per episode. A grep of
  `drake_port.py` + `drake_replay.py` for every development-subset
  episode number finds **no episode constants** (the only hit is a
  comment in `drake_replay.py` citing ring 195 / carton 1630 traces).
- Inputs are consumed exactly as claimed: poses via v1.1
  `build_scenes.load_episode_poses`, fixtures via `load_fixtures`,
  variant draws via `make_draws`, all geometry/mass/friction from
  `twin_params.py` (imports at the head of `drake_port.py`; jitter /
  mass / friction-multiplier application mirrors the v1.1 builder).
- **No v1.1 file was modified:** no file under the v1.1 `code/`,
  `scenes/`, or `validation/` trees has an mtime after 2026-11-22 21:30
  (the Drake work began ~21:44); the newest v1.1 files predate it.
  `mujoco_baseline.py` monkeypatches `replay.evaluate` in-process only,
  as described.

## 2. Port sanity: FK exactness + bit-exact MuJoCo baseline — CONFIRMED (stronger than claimed)

- **FK, re-derived by the verifier** (episode 411, recorded frame-0
  pose, both engines built/set independently): Drake gripper body
  (0.11825833, 0.00559181, 0.08902272), MuJoCo (0.11825837, 0.00559149,
  0.08902302) — agreement ≤ 5×10⁻⁷ m; gripperframe site Drake
  (0.14721041, 0.00852311, −0.00502317) vs MuJoCo (0.14720989,
  0.00852196, −0.00502307) — ≤ 1.2×10⁻⁶ m; quaternions agree to
  ~5×10⁻⁴ per component. The builder's "identical to 4 decimal places"
  claim holds, with margin.
- **MuJoCo baseline bit-exactness, checked on all 16 episodes** (the
  builder claimed full precision on block 411 + spot checks): for every
  development-subset episode, `mujoco_results/` final positions are
  **exactly equal** (float equality) to the stored v1.1 validation
  records in `validation/episodes/<task>__ep<e>_v0.json`, and grasp
  event lists (event/object/frame) and `sim_success` also match on all
  16. Claim confirmed and extended.

## 3. Diagnostics numbers — CONFIRMED by independent recomputation

Recomputed from the per-episode JSONs in `results/`, `results_hydro/`,
and `mujoco_results/` (not from the builder's `equiv_summary.json`,
which also matches):

| Config | Outcome agreement | Secondary agreement |
|---|---|---|
| Drake point | **15/16** (only block 498 disagrees) | **15/16** (only block 411 disagrees) |
| Drake hydroelastic | **16/16** | **15/16** (only block 411 disagrees) |

Per-task splits, MuJoCo successes (block 0/4, eye 1/4 = ep1798, ring 0/4,
tool 0/4), final-position medians/maxima, and the solver-failure lists
(point 5/16: block 489, eye 1630, eye 1783, ring 195, tool 883; hydro
4/16: block 489, eye 1630, ring 195, tool 852; MuJoCo 0/16) all
reproduce exactly.

**Block 498 flip — CONFIRMED, point-only, mechanism supported with one
numeric imprecision (Finding F1).** Grasp events are identical in both
engines (attach cube_b frame 59, release frame 205). Drake-point finals:
cube_b stacked on cube_a (offset Δx 2.9 mm, Δy 3.5 mm — inside the
11.25 mm inset half-width = 0.75 × CUBE_EDGE/2 with CUBE_EDGE = 0.030 m;
contact with lower cube only → SUCCESS). MuJoCo finals: both cubes on
the mat at equal z, inter-cube Δy 33.7 mm — outside the inset → FAIL.
Hydroelastic: both FAIL. The placement genuinely straddles the inset
criterion and post-release settle dynamics decide it. However, the
builder's "final positions differ by 2.4–3.0 cm" does not reproduce as
stated from the recorded finals: the per-object cross-engine deltas are
cube_a 3.44 cm and cube_b 0.57 cm (the ~3.0 cm figure matches the
*difference in inter-cube y-separation*, 33.7 mm vs 3.5 mm). Mechanism
stands; the quoted range is imprecise.
Additional detail the builder did not report: in the hydro run of 498
the outcome agrees (FAIL) but the trajectory does not — Drake-hydro
re-grasps at frame 232 and cube_b ends at (0.408, −0.008), a 0.555 m
final-position delta. Class agreement there is not trajectory agreement.

**Block 411 secondary flip — CONFIRMED.** MuJoCo cube z-values are
equal to ~3×10⁻¹⁷ (tie-break picks cube_b; secondary PASS, planar err
8.6 mm); Drake z-values differ by ~17 µm (builder said 20 µm), picking
cube_a (secondary FAIL). A criterion tie-break artifact, as diagnosed.

## 4. Weld-pin defect — CONFIRMED (and broader than §4.3 names)

Final states in both Drake configs vs MuJoCo, all still-attached at end:

| Episode | Drake point z | Drake hydro z | MuJoCo z |
|---|---|---|---|
| block 489 cube_a | −0.169 | −0.170 | +0.195 |
| eye 1630 carton | −0.109 | −0.134 | +0.204 |
| ring 195 ring | −0.221 | −0.220 | +0.162 |

MuJoCo's soft weld keeps every object at/above its support; the Drake
frame-start pose pin drives the object to the pinned pose regardless of
support. **Every** Drake solver-failure flag is this same class: the
two flags §4.3 does not name there — eye 1783 carton (z −0.071, point)
and tool 883 (z −0.057, point) / tool 852 (z −0.176, hydro) — are also
still-attached episodes ending below the −0.05 bound. So the defect
touches 5/16 (point) and 4/16 (hydro) episodes' final states, all
caught by the flag. The Drake harness's flag rule (non-finite, radial
> 1.0 m, or z outside [−0.05, 0.50] in `drake_replay.evaluate.store`)
implements the pre-registration §4 standard; note the v1.1
`replay.evaluate` itself only ever sets the flag on the cable path —
the Drake implementation is the §4 text applied to precision tasks,
which is the right reading, but the equivalence "v1.1 rule" label is
loose.
Two evidence caveats: (i) the builder's quoted final-z range
"−0.13…−0.22" is imprecise — the named three span −0.109…−0.221 and
all flagged episodes span −0.057…−0.221; (ii) the recorded `z_trace`
is sampled every 5 frames during the trajectory only (not during the
1.0 s settle, where the pin is maintained and part of the plunge
occurs): block 489 and ring 195 traces do show sub-zero descent, but
eye 1630's trace never goes negative (min +0.0488) — for eye 1630 the
decisive evidence is the final state, not the trace.
Effect on the headline numbers: all flagged episodes are FAIL in both
engines, so outcome agreement is not inflated in direction, but those
episodes' agreement is weak evidence (an unphysical Drake final state
agreeing in class). Excluding flagged episodes per the §4 rule:
point outcome agreement 10/11, hydro 12/12; secondary 10/11 and 11/12
respectively. The builder's open item 1 (contact-respecting Drake
grasp before Stage 2) is endorsed as blocking for Stage 2 use.

## 5. Loose-capsule migration (tool 683/852) — CONFIRMED by verifier re-run

The stored Drake result JSONs contain only a z-trace, so the migration
magnitudes are **not** recoverable from the recorded artifacts alone —
the builder's frame-by-frame trace was not persisted. The verifier
therefore re-ran the loose phase in a scratch copy (DrakeScene +
simulator, arm following the recorded gt, no grasp logic, tool 683 and
852, both configs):

- Point, ep683: tool x 0.1335 → 0.0478 by frame 92 (**−8.6 cm**),
  z sinks from 0.005 to ~0.0012 (≈3–4 mm sink). Builder: ≈8 cm, sinks
  3 mm — confirmed.
- Hydro, ep683: tool x 0.1335 → 0.0774 (**−5.61 cm**, plateaued by
  ~frame 50), z holds at 0.00495 ≈ 0.005, no sink. Builder: ≈5.6 cm by
  frame 95, rests at exactly z = 0.005 — confirmed.
- Stored-result behaviour is consistent: point ep683 records **no**
  attach (MuJoCo attaches at 92); hydro ep683 attaches at 92 in both
  engines; hydro ep852 attaches in Drake (77/161/200) where MuJoCo
  never does. Outcome classes agree on all four tool episodes in both
  configs (all FAIL), as claimed. The loose-object phase for slender
  round objects is not engine-equivalent — builder's conclusion stands.

## 6. The two pre-run build defects — FIXED in the reported runs' code path

- **(a) AssignRole proximity role:** current `drake_port.py` uses
  `SceneGraph.RemoveRole(..., Role.kProximity)` per arm collision
  geometry, with the AssignRole failure documented in a code comment;
  the verifier's own scene build reports 17 arm geometries processed.
  Behavioural proof in the reported runs: eye 1746 records
  `no_attach_in_excursion` at frame 93 in **both** Drake configs,
  identical to MuJoCo, with the carton ending at its recorded pose —
  the pre-fix signature (gripper shoving the carton into gate range,
  spurious attach) is absent; block 498's cube_a is likewise not
  ejected (finals are the sane values in §3). The fix was in effect
  for the reported numbers, not merely described.
- **(b) Post-step weld override:** `drake_replay.py` applies the pin
  at frame start before `AdvanceTo` (code + comment present), and its
  file mtime (21:56:48) precedes every reported result file (point
  21:59, hydro 22:02–22:05), so the reported runs executed this code
  path.
- **Provenance caveat (Finding F2):** `drake_port.py` was edited at
  22:05:14 *during* batch5 (hydro), and its regenerated bytecode is
  timestamped 22:05:17; the hydro ring results were written seconds
  later (22:05:22–38) on the edited code, and the point ep411 result
  was rewritten at 22:02:50 outside the batch scripts' pattern. There
  is no version control on the drake/ directory, so the content of
  the 22:05 edit cannot be diffed. The current file passes every
  behavioural check above (FK, eye 1746, inputs discipline), and ring
  hydro outcome/secondary classes match the point config and MuJoCo,
  so no result is contradicted — but strictly, the hydro ring subset
  cannot be proven to have run byte-identical builder code to the
  other batches. Severity: minor for part 1 diagnostics; the suite
  manifest for Stage 1 close-out should hash the builder code (per
  Amendment 1 A1.3) so this gap cannot recur.

## 7. Compute discipline — guard/mirror gap CONFIRMED as reported

All six batch launch logs (`runs/batch1…batch5`, incl. batch4b) carry
`launch_dgx_job: WARNING: launching WITHOUT off-box mirror by explicit
override (JOB_GUARD_ALLOW_NO_MIRROR=1)`, and the guard logs state
`JOB_GUARD_MIRROR_TARGET unset — telemetry stays ON-BOX`; telemetry is
present on-box under `runs/*/guard/` (guard.log, telemetry.csv). The
guard itself ran (preflight, log-only circuit breaker armed) and the
builder's reported preflight refusal is consistent with the guard's
design. CPU-only / nice'd execution is visible in the batch scripts.
**Severity:** *low for Stage 1 part 1* — the deliverable is local
result JSONs whose numbers this verification recomputed independently
from the artifacts themselves, so the missing off-box telemetry mirror
does not weaken any part-1 conclusion; *blocking for Stage 2* —
pre-registration §12 and the standing job-guard rule make the off-box
mirror a hard launch gate, and Stage 2 runs must not launch until the
VM→DGX mirror target is provisioned. The builder flagged this gap
itself, accurately.

---

## Findings summary

- **F1 (minor, numeric):** block-498 "2.4–3.0 cm" and weld
  "z −0.13…−0.22" ranges do not reproduce exactly from the recorded
  finals (correct quantities in §3/§4). Mechanisms and conclusions
  unaffected.
- **F2 (minor, provenance):** undocumented mid-batch edit to
  `drake_port.py` at 22:05:14 during the hydro batch; no VCS, content
  undiffable; hydro ring subset ran on the edited code. No result
  contradicted; hash the code in the Stage 1 manifest (A1.3).
- **F3 (main open defect, builder-disclosed, endorsed):** the Drake
  pose-pin grasp drives still-attached objects below support in
  5/16 point and 4/16 hydro dev episodes (all solver-flagged, all
  class-FAIL in both engines). A contact-respecting grasp realisation
  is required before any Stage 2 Drake result is trusted.
- **F4 (discipline, builder-disclosed):** guard mirror override on
  all batches — low severity for part 1, blocking for Stage 2 (§7).
- **F5 (completeness, minor):** hydro block-498 agrees in outcome
  class only; its trajectory diverges (re-grasp at 232, cube_b final
  delta 0.555 m). Not mentioned in the builder report.

## Verdict

**CERTIFIED-WITH-FINDINGS.** Items 1, 2, 3, 5, and 6 are confirmed —
several more strongly than claimed (bit-exactness on all 16 episodes,
not spot checks; FK agreement at the micrometre level; migration
magnitudes re-derived to within 0.1 cm). Item 4's defect is confirmed
and is broader than the builder's §4.3 enumeration, but it is flagged,
counted, and does not change any reported agreement class. Item 7's
discipline gap is confirmed exactly as the builder reported it. None
of the findings overturns a part-1 conclusion; F3 and F4 must be
closed before Stage 2, and F2 should be closed by hashing the port
code into the Stage 1 suite manifest.
