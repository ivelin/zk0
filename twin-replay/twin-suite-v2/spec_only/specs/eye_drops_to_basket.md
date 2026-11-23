# Task Spec — eye_drops_to_basket (Tier C / spec-only)

Sources: parameter audit §3 (E1–E7) + §1 globals; registered mechanical criterion. No episode data used.

## Task
Move the eye-drops carton from the raised platform into the basket.

## Objects
- carton: box 90×35×35 mm [MEASURED-EST, E1; ±20%/axis], mass 30 g [PROP, E2; range 20–45 g], friction (0.5, 0.005, 0.0001) [DEFAULT, E3; sliding 0.40–0.60].

## Fixtures
- platform: top plate 200×160×4 mm, top surface 90 mm above mat [MEASURED-EST, E4; height 75–105 mm], friction (0.4, 0.005, 0.0001) [DEFAULT, E5]. Position (0.32, −0.06) m — [DEFAULT] code fixture placeholder (gap G-E1).
- basket: footprint 260×180 mm, wall height 70 mm, wall ≈3 mm [MEASURED-EST, E6; walls 60–80 mm], friction (0.4, 0.005, 0.0001) [DEFAULT, E7]. Position (0.30, 0.15) m — [DEFAULT] (gap G-E1). Note: with 180 mm depth centred at y=0.15 the basket reaches y=0.24, beyond the mat edge (0.20) — inherited verbatim from the v1 default fixture and flagged (gap G-E2), not silently moved.

## Placement distribution (declared)
Carton starts resting on the platform top: platform centre + uniform offset x∈[−0.03,0.03] m, y∈[−0.04,0.04] m, yaw ±30°, long axis along x, upright [DEFAULT — gaps G-E3, G-E4].

## Target specification
Goal state: carton resting supported by the basket (inside basket interior footprint).

## Success criterion (verbatim)
"SUCCESS iff the box's final resting support is the basket (box center inside the basket's interior footprint and box supported by basket floor/walls), not the platform or the mat." Secondary 3.0 cm position check: episode suites only, n/a spec-only.

## Spec gaps
G-E1 fixture positions absent from audited table (code defaults used, labeled) · G-E2 basket overhangs mat edge (kept, flagged) · G-E3 placement distribution declared · G-E4 carton starting orientation declared (recorded orientation is episode data). · G-E5 platform web/base sub-geometry [DEFAULT] inherited v1.1 construction, dimensions recorded in specs.json platform.sub_geometry (hygiene pass 2026-11-23).
