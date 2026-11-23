# Task Spec — cable_clip (Tier C / spec-only)

Sources: v1 parameter table §6 + twin_params cable constants. NOTE: cable_clip is OUTSIDE the v2 parameter audit's scope (audit covers the four precision tasks; cable_clip's instrument is twin-suite v1.1, certified) — gap G-C2. Its v2 spec-only scenes are a demonstration of the Tier C path; cable_clip gate verdicts remain under the v1.1 episode suite per pre-registration §1/A2.1.

## Task
Seat a segment of the coiled cable into the clip on top of the clip block.

## Objects & fixtures
- clip block: box 110×70×55 mm, friction (0.6, 0.005, 0.0001); two jaw boxes (24×8×12 mm) forming an 8 mm channel on the block top, friction (0.5, 0.005, 0.0001). Channel width is a modelling choice; the real clip's sprung-jaw closing force is NOT modelled (v1 table §6). Block at (0.30, 0.05) m, yaw 0° [DEFAULT code fixture — gap G-C3].
- cable: composite capsule chain (APPROXIMATION, flagged in v1 table §6): 24 capsules × 30 mm, Ø5 mm, ball joints (damping 0.2 [DEFAULT]); seg_0 free joint damping 0.05 [DEFAULT]/untraced, gap G-C4, total length ≈72 cm, total mass 45 g [MEASURED-EST]/[PROP 40–60 g], friction (0.8, 0.005, 0.001). USB plug body 25×12×8 mm, 6 g at the chain end.

## Placement distribution (declared)
Cable as a ~2-turn coil of radius 55 mm centred at (0.27, −0.07) m, z = cable radius — the v1 code `default_cable_points` layout (a code-declared default, NOT episode data). [DEFAULT — gap G-C1: no cable layout distribution exists in the spec sources; variant jitter uses a declared 0.020 m whole-coil translation, labelled ESTIMATE.]

## Target specification
Goal state: a cable segment inside the clip channel (channel centre at block top + 6 mm, printed per scene in the manifest).

## Success criterion (verbatim)
"SUCCESS iff the cable is seated in the clip (cable segment inside the clip channel with the clip closed over it)." Per A2.2(ii): the recorded cable_clip demos are failed executions — solvability is proven by a reference-executor pass, independent of any demo outcome. Secondary position criterion: exempt (deformable).

## Spec gaps
G-C1 cable layout distribution + jitter scale declared · G-C2 cable parameters have no v2 audit (v1 lineage only) · G-C3 clip-block position from code default. · G-C4 seg_0 free-joint damping 0.05 vs declared ball-joint damping 0.2 — traced to v1/v1.1 build_scenes.py cable_xml_build (hardcoded 0.05; ball joints use twin_params CABLE_JOINT_DAMPING=0.2), discrepancy declared, value unchanged (hygiene pass 2026-11-23).
