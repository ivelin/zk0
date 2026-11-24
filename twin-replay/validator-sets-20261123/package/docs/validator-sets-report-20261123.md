# Validator Sets — Morning Report (overnight run 2026-11-23 → 2026-11-24)

**Founder target (2026-11-23 22:58 CT):** per task, 5 validators × 10
sim PASS episodes = 50 certified pass episodes per task (250 total).
Validator A = twins of real-world held-out eval episodes; validators
B–E = task-description-only (spec-only) scenes.

**Headline: 114 of 250 set episodes are filled with certified pass
episodes this morning (36/50 twin episodes; 78/200 spec-only
episodes), plus a cable variant harvest still landing (see cable).**
Every filled episode is a scene + an executed solution trajectory
that passed the frozen task criteria and was **certified by
independent re-execution** (fresh process, raw scene files, PASS
reproduced, final positions within 2 mm — actual deltas 0.00 mm in
nearly all cases). Full per-episode composition: `sets.json` in this
folder.

## Scoreboard (episodes per validator set, target 10 each)

| Task | A (twins) | B | C | D | E | Task total / 50 |
|---|---|---|---|---|---|---|
| block_stack | **10** | 8 | 8 | 7 | 7 | 40 |
| eye_drops_to_basket | **10** | 8 | 7 | 7 | 7 | 39 |
| cable_clip | **10** | 2* | 1* | 1* | 1* | 15* (+ harvest) |
| tool_insert | 5 | 4 | 4 | 3 | 3 | 19 |
| ring_insert | 1 | 0 | 0 | 0 | 0 | 1 |
| **Total** | **36/50** | | | | | **114/250** |

\* cable spec-only counts are pre-harvest; a dedicated harvest of
distinct cable variant episodes is in flight (see Cable).

## How certification and validation stand (two different layers)

- **Certification (re-execution): COMPLETE for every episode in the
  sets.** This is the program's register standard (PROVEN → CERTIFIED).
- **Cross-family GOOD validation (Solution Standard: ≥2 frontier
  models from families different than the generator, Muse (Muse
  Wonder)): INCOMPLETE — 0 solutions are fully validated this
  morning.** 27 solutions carry one leg (26 GPT-OSS 120B reads from
  the validator pass, incl. anchors; 1 new Gemini leg). The agy quota
  — shared with Cos and tonight's data-certification program — stayed
  exhausted all night; the quota-aware orchestrator on the DGX
  (pid 558276) is still running its backoff schedule and rebuilding
  `legs.json` as quota frees. Legs owed at cutoff: ~130 across the
  pre-existing pool, plus all of tonight's new solutions. **The sets
  are certified; they are not yet dual-family validated, and this
  report does not claim otherwise.**
- One leg worth flagging: **block_stack spec01 (suite-1 variant
  solution)** received a strict Gemini 4 Carbon read — GOOD=NO,
  GRADE=INCIDENT, PASS_CONCUR=YES (the release-retreat dragged
  cube_b 8.5 mm). If its GPT-OSS leg returns GOOD=YES it becomes a
  formal dispute and drops out of the validated count; it remains in
  the set as a certified pass episode with the dissent recorded.

## Per-task notes

**block_stack — 40/50.** A-set: 10 of 12 certified twin scenes
(476/479/514 new tonight, all CLEAN nominal). Spec pool: 30 distinct
certified episodes (suite-1: 13 = 5 nominal/8 variant; suite-2: 17 =
6 nominal/11 variant). Cleanliness across both suites' block pools:
predominantly CLEAN (suite-2: 16 CLEAN/1 RECOVERED of 17).

**eye_drops_to_basket — 39/50.** A-set: 10 of 11 certified twin
scenes (7 new tonight, incl. one Route-C pose-corner pass, ep1821).
Spec pool: 29 (suite-1: 14 = 1 nominal/13 variant; suite-2: 15 =
1 nominal/14 variant). Eye remains the retry-heavy task: many
RECOVERED grades (suite-1 eye: 13 of 19 incl. twins graded RECOVERED
in the harvest). **eye spec12 is excluded everywhere**
(failed_verification in the spec program's V3 stage; see
inventory-update).

**cable_clip — A full; spec-only structurally short (see finding).**
A-set: 10 certified twin scenes (2125/1896/1903 new tonight).
Spec-only: the cable generator emits ONE nominal coil geometry —
**all 20 nominal scene files are byte-identical within each suite**
(md5-verified for both seeds: suite-1 `e6d0a75e…`, suite-2
`8562a1cd…`). The 40 "nominal certified solutions" are therefore
executions in 2 distinct geometries, not 40 episodes. Distinct
certified cable episodes in hand: 5 (suite-1 nominal + 3
validator-pass distinct variant draws + suite-2 nominal). A harvest
certifying distinct variant files (all hash-distinct) is running to
close this gap; final cable numbers land in an addendum to this
report. Cable solutions use the documented v1.1 weld device
(gripper cannot pinch a 5 mm cable) — a solvability proof under that
registered device, stated in every cable bundle.

**tool_insert — 19/50.** The grasp-infeasibility wall is the binding
constraint everywhere: A-set 5 distinct episodes across only 4 scenes
(683 + tonight's 697/885/622 + 697's v4 variant); 9 further tool
twins exhausted 10 attempts each ("grasp of tool infeasible in all
yaw branches"). Spec: 14 (7+7 across suites; only 3 nominal).
13 scenes per suite exhausted their budgets unproven.

**ring_insert — 1/50. The gap, plainly.** One certified episode
exists: **twin ep48, rev3 controller on variant v2, CLEAN** — the
first certified ring solution in the program, obtained inside the
registered budget. Everything else failed: 12 ring twins × 11
attempts, 40 spec-only scenes × 11 attempts across two suites (440
attempts tonight's programs alone, plus the earlier 220). The
measured mechanism stands (wrist-limit translation-authority
collapse over the peg; the level-hang orientation family ends
14–21 mm short of the peg axis). **The frozen criterion also makes
a perfectly flat engaged ring unscorable** (parametric ring bottom
12.0 mm vs modelled tube Ø 11.5 mm → bottom_above ≈ −0.0003 = FAIL);
ep48's pass is a legitimate peg-propped rest inside the pass band.
Two founder decisions remain open and unchanged: (1) the ring
criterion parameter; (2) human-in-sim for ring (NOT authorized, not
started). Ring stays NO GATE.

## Nominal vs variant (reported both ways, never blended)

Episodes in the sets that rest on NOMINAL solutions, per task —
spec-only: block 11 of 30, eye 2 of 29, tool 3 of 14, cable 2 of 5
(+0 harvest, all variant), ring 0. Twin A-sets are nominal-first:
block 10/10 nominal, eye 6/10, cable 7/10, tool 1/5, ring 0/1
(several of tonight's new twin canonicals are themselves variant
passes — e.g. cable 2125 v1, 1896 v2, 1903 v3; eye 1639/1643 v1,
1724 v4, 1778 v3, 1821 pose-corner).
If the founder's open operationalization question resolves to
nominal-only denominators, the spec-only sets shrink to those
nominal counts and most validators' spec sets are near-empty for
eye/tool/cable — that is the honest consequence, stated now rather
than after the fact.

## Provenance (every episode)

- Generator (scenes + reference controllers): **Muse (Muse Wonder)**
  subagents. (One child report tonight wrote "Muse Spark" — a naming
  slip in that report only; the generator family is Muse (Muse
  Wonder), corrected here per the standing taxonomy.)
- Controllers: frozen (refcore.py/run_reference.py, md5-verified),
  rev2.1 (refcore2.py/run_reference3.py), ring rev3
  (run_reference4.py) — global versions only, no per-scene patches,
  no criterion or threshold changes anywhere.
- Certifiers: independent re-execution in separate contexts
  (validator-pass program, spec-only program V1/V2/V2b, tonight's
  twin/suite-2/harvest certifiers).
- Validation legs so far: GPT-OSS 120B (via agy) and Gemini 4 Carbon
  (via agy) as recorded in `legs.json`; the spec program's V3 agy
  audit leg (54 confirmed) is **not** counted as a GOOD leg — its
  model is unrecorded and it audited records rather than grading
  solution quality (see inventory-update-20261123.md).
- **Grok: zero credit.** Its V3 leg (33/22) is struck under the
  founder's 2026-11-23 21:53 CT suspension; the 22 disputes were a
  fixtures artifact, preserved on record, blocking nothing.
- **Muse probe: 0 scenes affected** — attempted for ring only,
  failed on missing credentials every time, never produced a verdict,
  never credited (full check: muse-probe-acceptance-check.md). Muse
  is same-family as the generator and can never count as a leg here.

## Unproven log (budget-exhausted, full censuses in the registers)

- Twins: 26 scenes — eye 1760/1783; tool 684/830/852/883/888/629/
  630/779/784/824/859; cable 1939; ring 7/30/33/50/167/179/195/198/
  11/31/160/165.
- Spec suite-1: block 7, eye 5, tool 13, ring 20 unproven (+ spec12
  excluded on verification, not budget).
- Spec suite-2: block 3 (spec00/11/13), eye 5, tool 13, ring 20.
- Attempt volume tonight: suite-2 712 run records; twins per
  register census; ring alone ≈ 440 attempts across suites 1–2
  budgets in the wider program.

## Object parameters

Per the founder's 2026-11-23 23:01 CT guidance: arm = the project's
exact tested model (MEASURED). Twin scene objects: FROM-EPISODE
(converted-scene manifests). Spec-only objects: ESTIMATED (anchored
on arm scale, task description, typical real-world dimensions),
labeled per object in each suite's scene manifests — no estimate is
presented as measured, and the monthly realism re-audit can revisit
every one. OI-1/OI-2 did not stop any work tonight.

## Backup

Per the updated founder rule, the backup package for these sets is
**PREPARED and HELD — not uploaded** (same status as the spec
program's package for the 54). Package contents: `sets.json`,
per-episode scene files + canonical solution records for the set
episodes, and a SHA-256 manifest (`package/` in this folder). It
uploads to the HF `zk0bot/sim-scene-solutions` pattern on the
founder's go.

## Still owed / next

1. Cross-family GOOD legs for the full pool (orchestrator running;
   completes as agy quota frees — no action needed unless it stalls).
2. A certification round for recorded-but-uncertified duplicate
   variant passes in both spec registers (the remaining route to
   grow block/eye/tool spec sets without new scene generation).
3. Founder decisions: ring criterion parameter; ring human-in-sim;
   nominal-vs-variant operationalization; HF backup go.
4. Cable harvest final numbers (addendum to this report).
