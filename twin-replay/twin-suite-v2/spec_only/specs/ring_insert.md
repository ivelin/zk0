# Task Spec — ring_insert (Tier C / spec-only)

Sources: parameter audit §4 (R1–R9) + §1 globals; registered mechanical criterion. No episode data used.

## Task
Pick up the ring lying flat on the mat and seat it on the peg on the board.

## Objects & fixtures
- board: cylinder Ø170 mm × 15 mm [MEASURED-EST, R1; Ø 150–190 mm], 6 recesses NOT cut (flat top) [MEASURED-EST, R2], friction (0.5, 0.005, 0.0001) [DEFAULT, R3]. Position (0.30, 0.0) m [DEFAULT — gap G-R2].
- peg: Ø16 mm, 70 mm above board [MEASURED-EST, R4; Ø 12–20 mm, height 55–85 mm], friction (0.4, 0.005, 0.0001) [DEFAULT, R5], at board centre.
- ring: outer Ø45 mm, inner Ø22 mm, thickness 12 mm [MEASURED-EST, R6], mass 10 g [MEASURED-EST, R7; 4–15 g], friction (0.5, 0.005, 0.0001) [DEFAULT, R8], built as the v1 12-segment capsule ring. Colour yellow (v1 fixed; per-episode colours are episode data — gap G-R1).

## Placement distribution (declared)
Ring flat on the mat, uniform x∈[0.18,0.38] m, y∈[−0.15,0.15] m, rejecting the disc of radius board_r+0.03 m around the board centre (ring starts off the board). [DEFAULT — gap G-R3.]

## Target specification
Goal state: ring seated on peg (peg through ring hole, ring lowest point within 1 cm of board top around the peg). Peg xy and board top z are printed per scene in the manifest.

## Success criterion (verbatim)
"SUCCESS iff the ring is seated on the peg: peg passes through the ring hole, ring's lowest point within 1 cm of the board top surface around the peg." Secondary 2.0 cm when unengaged: episode suites only, n/a spec-only.

## Spec gaps
G-R1 per-episode ring colour · G-R2 board position · G-R3 placement distribution declared.
