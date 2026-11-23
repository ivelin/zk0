# Task Spec — tool_insert (Tier C / spec-only)

Sources: parameter audit §5 (T1–T8) + §1 globals; registered mechanical criterion. No episode data used.

## Task
Pick up the loose tool from the mat and insert it into its target slot in the tool tray.

## Objects & fixtures
- tray: 385×270×45 mm [PROP, T1; point-only], friction (0.6, 0.005, 0.0001) [DEFAULT, T2]. Position (0.40, 0.0) m [DEFAULT — gap G-T2]. The ≈11 seated tools of the real tray (static, merged in v1 scenes) are NOT modelled — their layout is frame/episode data (gap G-T1); declared simplification carried.
- target slot: footprint 110×32 mm [DEFAULT, T3; point-only], recess depth 5 mm [MEASURED-EST — depth: ESTIMATE, never measured, T4], at (0.35, 0.06) m, yaw 0° [DEFAULT — gap G-T2], built as slab + rim recess exactly as the v1.1 scenes.
- loose tool: capsule, length 100 mm, shaft Ø10 mm [MEASURED-EST, T6; length 75–125 mm], mass 70 g [MEASURED-EST, T7; 50–100 g], friction (0.4, 0.005, 0.0001) [DEFAULT, T8; sliding lit. 0.30–0.50].

## Placement distribution (declared)
Tool lying on the mat (axis horizontal, z = tool radius), uniform x∈[0.15,0.33] m, y∈[−0.14,0.02] m, yaw uniform 0–180°, with an explicit tray-footprint rejection check in the generator. [DEFAULT — gap G-T3.]

## Target specification
Goal state: tool centre within slot footprint, tool axis within 20° of slot axis (slot axis = x at yaw 0°), resting in the tray. Slot centre/axis printed per scene in the manifest.

## Success criterion (verbatim)
"SUCCESS iff the tool's final pose lies in its target tray slot: tool center within the slot's footprint and tool axis within 20° of the slot axis, resting in the tray." Secondary 3.0 cm: episode suites only, n/a spec-only.

## Spec gaps
G-T1 seated-tools layout omitted · G-T2 tray/slot positions from code defaults · G-T3 placement distribution declared.
