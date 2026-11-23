# Twin-Suite v2 — Stage 1 (Amendment 2 §A2.1): Spec-Only Scene Generator + Demonstration Set

**Date:** 2026-11-22 · **Tier:** C (spec only) · **Workdir (DGX):**
`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/spec_only/`
**Governing text:** twin-suite v2 pre-registration, Amendment 2 §A2.1.
**No v1.1 file was modified. No held-out episode pose, split entry, or
recording-derived per-episode data was used anywhere in generation** —
the generator's only inputs are `specs/specs.json` and the padded v2 arm
include. (Task identities, objects, fixtures and the mechanical criteria
are public spec text from the BENCH-003 / v1 pre-registration record.)

## 1. Deliverables

| Item | Location (under `spec_only/` on the DGX) |
|---|---|
| Five task specifications | `specs/block_stack.md`, `specs/eye_drops_to_basket.md`, `specs/ring_insert.md`, `specs/tool_insert.md`, `specs/cable_clip.md` |
| Machine-readable specs (the generator's actual input) | `specs/specs.json` (SHA-256 `09d89574…`) |
| Generator | `generate_spec_only.py` (SHA-256 `d5b73567…`), self-contained, CPU-only |
| Demonstration scenes | `scenes/` — 100 XMLs = 20 scenes × K=5 variants, v2 actuated format (include → `scenes-v2/arm/so101_new_calib_padded.xml`) |
| Suite manifest | `suite_manifest.json` (SHA-256 `6e9178d1…`): per scene — seed, sampled placement, all variant draws (jitter / mass / friction multiplier), per-file SHA-256, ground-truth initial state, target specification, parameter-label pointer |
| Settle check | `validate_spec_only.py` + `settle_check.json`; guard run record in `runs/settle1/` (on-box telemetry) |

**Seed:** master seed **20261126**, registered in the manifest (new seed,
distinct from the episode-suite seed 20261122; draw order documented in
the generator docstring). **Regeneration determinism verified:**
re-running the generator into a fresh directory produced byte-identical
scenes (`diff -rq` clean).

## 2. Demonstration set status (per task)

4 scenes per task × K=5 variants, as registered in §A2.1 (the pre-reg
names the four precision tasks; cable_clip's 4 are included per the task
brief, as a Tier C demonstration — its gate remains the v1.1 suite).

| Task | Scenes | Variants | Load + 1.0 s settle | Solver failures | Validated denominator |
|---|---|---|---|---|---|
| block_stack | 4 | 20 | 20/20 clean | 0 | **0 / 4 — solvability not yet executed** |
| eye_drops_to_basket | 4 | 20 | 20/20 clean | 0 | **0 / 4** |
| ring_insert | 4 | 20 | 20/20 clean | 0 | **0 / 4** |
| tool_insert | 4 | 20 | 20/20 clean | 0 | **0 / 4** |
| cable_clip | 4 | 20 | 20/20 clean | 0 | **0 / 4** |

Settle check = load in MuJoCo 4.0.1, 1.0 s settle, finite state, v1.1
solver-failure bounds. Max manipulated-object drift at settle: cubes
0.08 mm, carton 0.02 mm, ring 0.26 mm, cable seg_0 4.9 mm — **except the
loose tool, which rolls**: spec00_v0 2.1 cm; spec03 variants 1.6–6.3 cm.
The tool is a round capsule lying on the mat (the audited/v1.1 geometry);
rolling at rest is inherent to that geometry, not a generator defect.
Flagged for the solvability executor: a pass trajectory must start from
the settled tool pose or account for roll; the manifest's ground-truth
initial state is the pre-settle sampled state.

**Solvability is explicitly NOT claimed.** Per the brief, A2.2 executed
passes belong to the reference-controller workstream. Every scene ships
with what that executor needs: scene file (v0–v4), ground-truth initial
state, and target specification (goal object, fixture coordinates,
slot/peg/channel centres) in `suite_manifest.json`. Until passes exist,
validated denominators are 0 and no scene counts toward any verdict.

## 3. Equivalence self-check (§A2.1 components)

| §A2.1 component | Episode suite (Tier A) | This spec-only suite |
|---|---|---|
| Geometry / fixtures / objects | Asset library + task spec | Same audited values (§3.1 parameter table; cable from v1 table §6 lineage), same geometry construction as the v1.1 scenes (ring = 12 capsule segments, tray = slab + rim recess, cable = 24-capsule chain, basket/platform as built), v2 padded arm. **Equivalent, except gaps G-T1/G-C2 below.** |
| Initial-state distribution | Pose estimates ± measured σ from recordings | Declared uniform placement distributions over the workspace from the specs (ranges in `specs.json`), variants jittered by the audited per-task σ (§3.3) with mass/friction drawn from the audited ranges — exactly the §A2.1 Tier C substitute. **Not the same distribution as the episode suite; that is the registered Tier C difference, not a defect.** |
| Denominators | Frozen held-out set, 96 | Generated set, size registered at generation: nominal 4/task (demo set), **validated 0/task** until A2.2 passes land. Reported per task, never pooled with episode-suite verdicts. |
| Solvability route | Recorded demo replay (retired for precision tasks) | Reference/scripted controller or human-in-sim (A2.2 b/c) — executor workstream, not this build. Deliverable supplies scene + initial state + target per scene. |
| Controls | §3.4/§6 public checkpoints | Unchanged — no checkpoint is gated by this build; controls apply when a validator runs the suite. |
| Criteria / semantics | Mechanical `contacts_with()` criteria, three-valued verdicts, envelope, blinding, judging layer | Criteria carried **verbatim** into each spec (see §4); verdict semantics, envelope machinery and Wilson bar unchanged and unmodified. Spec-only verdicts, when they exist, are cited as spec-only-suite verdicts and never pooled (A2.1). |

## 4. Success criteria (verbatim, as carried into the specs)

- **block_stack** — "SUCCESS iff one cube rests stably on the other at settle: upper cube in contact with lower cube's top, upper cube's center within the lower cube's top footprint (inset by 25%), no other support contact."
- **eye_drops_to_basket** — "SUCCESS iff the box's final resting support is the basket (box center inside the basket's interior footprint and box supported by basket floor/walls), not the platform or the mat."
- **ring_insert** — "SUCCESS iff the ring is seated on the peg: peg passes through the ring hole, ring's lowest point within 1 cm of the board top surface around the peg."
- **tool_insert** — "SUCCESS iff the tool's final pose lies in its target tray slot: tool center within the slot's footprint and tool axis within 20° of the slot axis, resting in the tray."
- **cable_clip** — "SUCCESS iff the cable is seated in the clip (cable segment inside the clip channel with the clip closed over it)."

The secondary position criterion (3.0 cm; ring 2.0 cm; cable exempt) is
episode-suite-only — it compares against a recorded final position,
which a Tier C validator does not have. Noted in each spec.

## 5. Spec gaps — where episode data would have been needed, and what was done instead

Every gap was handled by a declared distribution choice or a labeled
[ESTIMATE]/[DEFAULT] carried from code/parameter sources — **never**
filled from episode data.

| ID | Gap | Handling |
|---|---|---|
| G-B1 / G-R1 | Per-episode object colours exist only in pose records | v1 fixed colours (cube_a purple, cube_b green, ring yellow), declared in specs. Visual only. |
| G-B2 / G-E3 / G-R3 / G-T3 | **No placement distribution exists** in the task description, asset library, or audited parameter table — for any task | Declared uniform distributions over mat regions derived from fixture layout + reach + non-overlap rules, printed with ranges in `specs.json` and each spec doc, labeled [DEFAULT] declared choice. |
| G-B3 | Criterion is symmetric ("one cube on the other") — moved cube unnamed | Declared: cube_a moved, cube_b base, so the solvability executor has an unambiguous target. |
| G-E1 / G-R2 / G-T2 / G-C3 | Fixture positions (platform, basket, board, tray, slot, clip block) are **not in the audited parameter table** | `twin_params.DEFAULT_FIXTURES` values carried as spec fixtures, labeled [DEFAULT] code placeholders in every spec. |
| G-E2 | Basket at the default position overhangs the mat edge (centre y=0.15 + 90 mm half-depth = 0.24 > mat edge 0.20) | Kept verbatim, flagged in the spec — not silently moved. A validator inheriting the code fixture inherits this; fixing it is a suite-version decision, not a builder edit. |
| G-E4 | Carton starting orientation on the platform is recording data | Declared: long axis along x, upright (the v1 code default pose), yaw drawn ±30°. |
| G-T1 | The tray's ≈11 seated tools (static in v1 scenes) have layouts only in frames/episodes | **Omitted**, declared simplification carried and flagged. The target slot recess is built exactly as v1.1. |
| G-C1 | No cable layout distribution or audited cable σ exists | v1 code `default_cable_points` coil (2 turns, r=55 mm, centre (0.27, −0.07)) used as the declared layout; variant jitter = declared 0.020 m whole-coil translation, labeled [ESTIMATE]. |
| G-C2 | cable_clip has **no v2 parameter audit** (audit scope = four precision tasks) | Cable parameters taken from the v1 table §6 / `twin_params` lineage and labeled as such in the spec; no v2 label claimed. |
| OI-1 / OI-2 (inherited) | Mat dimensions and cube material remain open items in the audit | Carried at audited values with labels; cube variant mass draws use the frozen-code sampling range 10–35 g — the audited 5 g foam corner is an envelope corner, not a variant draw, and is flagged in `specs.json`. |

**Places episode data would have been needed and was refused:** initial
poses of every object (→ declared distributions, G-*2/G-*3 above);
per-episode colours (→ fixed colours); carton orientation (→ declared);
seated-tool layout (→ omitted); fixture positions (→ labeled code
defaults, the one place a non-episode code source existed).

## 6. Guard / compute note

The settle-check job ran on the DGX via `launch_dgx_job.sh` (JOB_LOCK=0,
CPU-only, mode logonly), run dir `spec_only/runs/settle1/`. The first
launch hit the guard's mirror hard gate: `JOB_GUARD_MIRROR_TARGET` is
unset on the DGX and the mirror target is unreachable from it, so the
launcher refused. Per the brief's instruction (log if hit, proceed with
on-box telemetry), the job was relaunched with the guard's documented
per-launch no-mirror override; the override is recorded in the guard log
(`runs/settle1/guard/`) and telemetry stayed on-box in that run dir.
No ceiling or breaker settings were altered.

## 7. What remains (not this build)

1. **A2.2 solvability execution** (reference-controller workstream):
   executed pass per scene; scenes without a pass are excluded from
   denominators and logged. Tool-roll behaviour (§2) needs a stated
   convention there (settle-then-grasp vs sampled initial state).
2. **Realism / task-match screen** (frontier + human) before any scene
   is used in a gate.
3. Independent verification of this build (builder ≠ verifier, §9)
   before any certification claim.

---

## 8. Hygiene pass — findings F1–F3 closed (2026-11-23)

Minor findings from `verification-v2-spec-only.md` (CERTIFIED-WITH-FINDINGS),
closed in a small pass on the DGX workdir above. No v1.1 file was touched,
and no generated scene value was changed.

- **F1 — Per-scene seed.** Each of the 20 scene entries in
  `suite_manifest.json` now carries a `seed` field, derived deterministically
  as `per_scene_seed = master_seed (20261126) + global_scene_index`, where
  `global_scene_index = task_order_index × 4 + scene_index` (task order
  alphabetical: block_stack=0, cable_clip=1, eye_drops_to_basket=2,
  ring_insert=3, tool_insert=4; scene_index 0–3). Seeds therefore run
  20261126–20261145 in manifest order. The derivation is documented in the
  generator docstring and in the manifest header
  (`per_scene_seed_derivation`). The per-scene seed is a recorded identifier
  only — generation still draws from the single `Random(20261126)` stream —
  and a structural diff confirms the regenerated manifest is otherwise
  identical to the pre-fix manifest.
- **F2 — Platform web/base sub-geometry.** Assigned gap **G-E5** (the next
  ID in this suite's existing G-<task><n> gap scheme). The dimensions are
  now recorded in `specs.json` under
  `tasks.eye_drops_to_basket.fixtures.platform.sub_geometry`: web
  half-extents 0.01 × td/4 × h/2 (full 0.02 × 0.08 × 0.09 m), base
  half-extents tw/2 × td/2 × 0.002 (full 0.20 × 0.16 × 0.004 m) at z=0.002,
  labeled [DEFAULT] inherited v1.1 construction, untraced in the audited
  parameter table (audit E4 declares web/base "not separately
  parameterized"). Source traced to the frozen v1.1 builder
  (`zk0-study001-twin-20261122/code/build_scenes.py`, `platform_web` /
  `platform_base` geoms). Also registered in the top-level
  `spec_gaps_register_additions_2026-11-23` in `specs.json`, in the
  eye_drops spec doc's gap list, and as a code comment in the generator.
- **F3 — Cable seg_0 free-joint damping 0.05.** Assigned gap **G-C4**.
  Traced to the v1/v1.1 lineage: `zk0-study001-twin-20261122/code/build_scenes.py`,
  `cable_xml_build`, hardcodes `damping="0.05"` for `cable_seg_0_free`,
  while the ball joints use `twin_params.CABLE_JOINT_DAMPING = 0.2`
  (`zk0-study001-twin-20261122/code/twin_params.py`). `specs.json` now
  records `seg0_free_joint_damping: 0.05` separately from the ball-joint
  0.2, with the discrepancy explicitly registered as gap G-C4 (also in the
  cable_clip spec doc and a generator code comment). **The value was NOT
  changed** — whether 0.05 vs 0.2 is physically right for the free joint
  remains an open item for a future suite-version decision; this pass only
  makes the discrepancy visible.

**Byte-identity proof:** SHA-256 of all 100 scene XMLs was captured before
the fixes and after regeneration (`nice -n 10 python3 generate_spec_only.py`,
CPU-only). The two sorted hash lists are identical — `diff` clean, 100/100 —
and each list's own SHA-256 is
`5e3452e53883e0af54b09d3b738d3ccee2208617299d150a8c966fab0e3a240b`.
Lists are filed on the DGX at `spec_only/runs/hygiene1/scene_hashes_pre.txt`
and `scene_hashes_post.txt`. Only `suite_manifest.json` (seed fields +
derivation note), `specs/specs.json` (G-E5/G-C4 registers), the two spec
docs' gap lists, and generator comments/docstring changed.

New file identities after the pass: generator SHA-256 `c96fcec4…`,
`specs.json` `d1045b54…`, `suite_manifest.json` `dfd67858…`
(pre-pass: `d5b73567…`, `09d89574…`, `6e9178d1…`).
