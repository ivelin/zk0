# Overnight Sim Validator-Set Program — Program Record (2026-11-23, started ~23:05 CT)

**Founder target (2026-11-23 22:58 CT):** per task (block_stack,
eye_drops_to_basket, ring_insert, tool_insert, cable_clip), validator
scene sets for **5 validators × 10 sim PASS episodes = 50 certified
pass episodes per task (250 total)**. Validator A per task: digital
twins of real-world held-out eval episodes (may see the real scene via
the eval episode). Validators B–E per task: scenes generated from the
task description ONLY (spec-only, blind to any real scene). A pass
episode = a scene + an executed solution trajectory passing the task
in that scene. Morning deliverable: per-task accounting vs target,
provenance, unproven log, ring gap stated plainly; all scenes +
solutions archived/versioned and backed up remotely with hashes.

## Governing rules (founder, standing — binding on all workers)

- **Scene Solution Rule:** a scene counts only with a proven,
  INDEPENDENTLY CERTIFIED solution trajectory (independent
  re-execution in a separate context reproduces the pass). One
  solution per (scene, task) pair, stored once, backed up to the SSOT
  remote (HF `zk0bot/sim-scene-solutions` pattern).
- **Solution Standard (2026-11-23 20:11 CT):** solutions smooth,
  failure-free, relatively optimal/efficient. A solution counts only
  after validation by **≥2 frontier models from DIFFERENT families
  than the generating model**. Generator family for all
  scenes/controllers tonight: **Muse (Muse Wonder)**. Validation
  legs therefore come from: **Gemini (Google) \<model\> (via agy)**
  and **GPT-OSS 120B (OpenAI, via agy)** and/or **Claude (Anthropic)
  \<model\> (via agy)**. Exact model recorded per role per scene.
- **Grok is SUSPENDED** as judge/verifier — no Grok read certifies
  anything tonight.
- **Judge naming taxonomy:** "Muse (Muse Wonder)" = Meta subagents;
  "Gemini (Google) \<model\> (via agy)"; "Claude (Anthropic) \<model\>
  (via agy)" — Anthropic only where an Anthropic model actually acted.
- **Trajectory cleanliness:** every solution graded CLEAN / RECOVERED /
  INCIDENT from its execution record (failed grasps, retries,
  displacement events, IK fallbacks after task completion disclosed).
- **Frozen criteria:** NO criterion or threshold changes. ring_insert's
  criterion-parameter finding (parametric ring bottom 12.0 mm vs
  modelled tube Ø 11.5 mm; a perfectly flat engaged ring reads
  bottom_above ≈ −0.0003 → FAIL) is a FOUNDER decision, still open.
  Peg-propped rests from entry tilts ≥20° land in the pass band
  legitimately and may be used; the gap is reported, never papered
  over. Human-in-sim for ring is NOT authorized.
- **Object-parameter guidance (founder, 2026-11-23 23:01 CT):** the
  arm model is exact (the project's own tested assets, reused to the
  max). For OBJECTS, do not stall on unrecorded parameters (OI-1/OI-2):
  build realistic approximations anchored on (a) known arm
  geometry/scale, (b) the task description, (c) for validator-A twins,
  the eval episode's own visual/recorded evidence, (d) typical
  real-world dimensions for the object class in spec-only sets.
  RECORD every object's parameters per scene in the scene manifest,
  each marked **MEASURED / FROM-EPISODE / ESTIMATED**. An estimated
  parameter is never presented as measured. Missing object data is
  not a stop reason tonight.

## Fair-shot attempt budget (stated BEFORE tonight's attempts)

Adopts the desktop spec-only program's registered budget verbatim
(`attempt-budget-and-rules-20261123.md`, 2026-11-23 13:15 CT) for all
NEW scenes and for prior scenes with remaining budget. Prior canonical
attempts count against a scene's budget.

- **Route A (primary):** every admitted global controller version runs
  the scene's NOMINAL variant (k=0) once. block/eye/cable/tool:
  frozen + rev2.1 (2 attempts). ring: frozen + rev2.1 + rev3
  (3 attempts). Controllers are GLOBAL versions only — no per-scene
  patches, no conditionals on scene identity, no relaxed limits.
- **Route B (scenes unproven after A):** the task's best-performing
  controller runs sampled variants k=1..4 (4 attempts).
- **Route C (still unproven):** same controller runs pose-corner
  variants k=5..8 (4 attempts; precision tasks only — cable has no
  frozen corners).
- **Maximum per scene: 10 attempts (ring 11).** Every attempt produces
  a run record. Budget-exhausted scenes are logged **UNPROVEN** with
  the full census — never silently dropped, never retried indefinitely.
- A pass on a variant counts as a solution for that scene (variant
  named in the record); validator-set assignment prefers nominal
  passes, then variant passes of distinct scenes, then — only if a set
  is still short — additional DISTINCT variant pass episodes of an
  already-used scene, labeled as variant episodes. Set composition is
  reported transparently per validator.

## Starting inventory (verified at source, 2026-11-23 ~23:00 CT)

**Validator-A material — twin scenes (DGX
`zk0-study001-twin-v2-20261122/scenes-v2/`):** 40 nominal twin scenes,
8 per task, twins of held-out eval episodes. Certified solutions on
record (union register, per-version controllers; HF
`zk0bot/sim-scene-solutions` @ `e57de85a…`; video pack reproduced
20/20 exactly): **block 8/8** (411, 453, 489, 498, 499, 527, 546,
549), **eye 4/8** (1609, 1630, 1746, 1798), **tool 1/8** (683),
**cable 7/8** (1905, 1969, 2020, 2046, 2068, 2085, 2130), **ring 0/8**.
Expansion pool: v1 twin suite on the DGX
(`zk0-study001-twin-20261122/scenes/`, ~20 episodes/task) — unconverted
episodes can be brought to v2 format by the established arm-include
swap (no v1.1 file modified; v2 scenes are generated copies).

**Validators B–E material — spec-only FULL suite (desktop
`~/muse-work/zk0-spec-only-20261123/work/spec_only_full/`, master
seed 20261127, 20 scenes/task, K variants each), register state
verified from `register.json` + `certification-merged.json`:**
**block 13 CERTIFIED / 7 UNPROVEN; cable 20 CERTIFIED / 0; eye 15
CERTIFIED / 5 UNPROVEN; tool 7 CERTIFIED / 13 UNPROVEN; ring 0
CERTIFIED / 20 UNPROVEN — total 55 certified scenes.** Certification =
independent re-execution reproducing the pass (separate verify
context, desktop). Of these, the 2026-11-23 validator pass graded 15
CLEAN (block spec02/08/10/13; eye spec16/spec00/spec01 clean +
spec17 RECOVERED-solvable; tool spec00/06/09/19; cable spec00 k0 /
spec01 k1 / spec02 k2 / spec03 k3) with ONE cross-family leg
(GPT-OSS 120B via agy, concurring); the second leg is OWED (Gemini
was quota-blocked during that pass; the founder's Google AI Pro
upgrade landed 2026-11-23 ~20:48 CT). The remaining 40 certified
scenes still need cleanliness grades + both validation legs.
Expansion: the suite generator (`generate_spec_only_full.py`) can
produce a further suite under a new master seed (tonight: 20261128).

**Ring:** rev3 (run_reference4.py, frozen) went 0/20 variants on the
Tier C ring scenes and 0/4 on the FULL-suite scenes' budgets per the
register; the measured mechanism (wrist-limit translation-authority
collapse; level-hang family ends 14–21 mm short of the peg axis)
stands. Ring attempts tonight stay inside the registered budget on
NEW scenes only; expectation is honestly low.

## Machine plan

- **Desktop (Pop!_OS, 100.73.62.79):** spec-only harvest + new-suite
  generation/attempts in the NEW workdir
  `~/muse-work/zk0-validator-sets-20261123/` (the prior program's
  workdir is read-only for us — never overwrite). No agent processes
  of the prior program were running at 23:00 CT (checked).
- **DGX Spark:** twin expansion + agy validation legs, workdir
  `/home/openclaw/muse-work/zk0-validator-sets-20261123/`.
- All jobs via the documented launch wrapper + job guard, CPU-only
  MuJoCo at `nice -n 10`, whole-box ≤70% across tenants, no boot
  services. The guard telemetry mirror target is a standing known
  failure (Muse VM exposes no sshd, rc=255) — telemetry stays on-box
  in each rundir's `guard/` folder and the failure is logged, per
  precedent; no ceiling/breaker setting is altered.

## Set assignment rule (per task)

- **Validator A:** 10 twin pass episodes — certified twin scenes
  first (nominal), then newly certified twins, then distinct variant
  passes of certified twin scenes, labeled.
- **Validators B, C, D, E:** 10 spec-only pass episodes each, filled
  in register order from the certified pool (nominal passes first,
  then variant solutions of distinct scenes), then tonight's new-suite
  certifications, then distinct variant episodes, labeled. No scene's
  canonical solution is counted twice across validators except as
  explicitly labeled variant episodes of that scene.

## Deliverables (morning)

Per task: validator sets A–E with scene counts, certified pass
episodes per set vs the 10 target, solution provenance (generator
model, validator models, cleanliness), unproven-scene log with
budgets, ring gap stated plainly. Archive: scene XMLs + canonical
solution records + manifests (object parameters labeled
MEASURED/FROM-EPISODE/ESTIMATED; hashes) pushed to the HF
`zk0bot/sim-scene-solutions` pattern with a validator-sets manifest;
report in this folder; commit via the output-minimized git route
(exit codes/hashes only; on an approval notice, STOP and report).
