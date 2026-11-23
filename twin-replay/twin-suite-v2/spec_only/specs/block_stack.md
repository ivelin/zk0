# Task Spec — block_stack (Tier C / spec-only)

Validator-facing specification. Sources: twin-suite v2 parameter audit §2 (B1–B5) and §1 globals; registered mechanical criterion (v1 pre-registration, carried unchanged by v2 §4). No episode, pose, split, or recording data is used anywhere in this spec.

## Task
Pick up one 30 mm cube and stack it stably on the other cube on the WOWROBO mat.

## Objects
- cube_a, cube_b: box, edge 30 mm [MEASURED-EST, B1; envelope 20–40 mm], mass 20 g nominal [MEASURED-EST, B2; full uncertainty 5–35 g — material unresolved, OI-2: painted wood or dense foam], friction (0.6, 0.005, 0.0001) [DEFAULT, B4; sliding envelope 0.48–0.72]. Colours: cube_a purple, cube_b green (v1 fixed colours; per-episode colours are episode data — gap G-B1).

## Workspace / fixtures
Common workspace (see specs.json `workspace`): mat 550×400×3 mm [MEASURED-EST, G8, OI-1 open], walls 60 mm, arm base at origin. No task fixtures.

## Placement distribution (declared)
Uniform x∈[0.15,0.45] m, y∈[−0.12,0.12] m on the mat, yaw uniform ±45°, minimum pairwise separation 0.08 m, resting on mat (z = edge/2). Label: [DEFAULT] declared distribution choice — gap G-B2 (no placement distribution exists in the spec sources).

## Target specification (for the solvability executor)
Moved cube: cube_a. Base cube: cube_b. Goal state: cube_a resting on cube_b.

## Success criterion (verbatim, registered mechanical criterion)
"SUCCESS iff one cube rests stably on the other at settle: upper cube in contact with lower cube's top, upper cube's center within the lower cube's top footprint (inset by 25%), no other support contact." Secondary (reported, not gated): final planar position within 3.0 cm (episode suites only; not applicable spec-only — no recorded final position exists).

## Spec gaps
G-B1 colours per episode (episode data, not used; fixed colours declared) · G-B2 placement distribution (declared uniform region) · G-B3 moved-cube identity (criterion is symmetric; cube_a declared as moved).
