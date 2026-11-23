# Study 001 Twin-Replay — Physics Parameter Table (Phase 2, Builder A, 2026-11-22)

Source tags (per pre-registration honesty rule 1):
- **[MEASURED-EST]** — estimated from the recordings; method stated.
  Dimensions come from `docs/scene-inventory.md` pixel-ratio estimates
  (uncertainties ±20% unless stated); mass = volume × material density.
- **[PROP]** — physical-prop knowledge; the prop is named.
- **[DEFAULT]** — MuJoCo / literature default or literature range midpoint,
  labelled as such. No value in this table is a silent invention.

MuJoCo conventions: friction vector = (sliding, torsional, rolling);
solref/solimp left at engine defaults unless noted. Gravity −9.81 m/s²
[DEFAULT]. Timestep for twin scenes: 0.002 s [DEFAULT, MuJoCo example
scenes] — Phase 4 may change it globally with a logged reason.

## 1. Common / workspace

| Item | Geom (planned) | Mass | Friction (slide, tors, roll) | Restitution | Material class | Source |
|---|---|---|---|---|---|---|
| Table / enclosure floor | plane (static) | — | (0.8, 0.005, 0.0001) | 0.0 | cardboard-on-table | friction [DEFAULT] (cardboard/paper on wood laminate, lit. 0.5–0.9) |
| WOWROBO mat | box, ≈550×400×3 mm, static | — | (0.9, 0.005, 0.0001) | 0.0 | rubberised fabric mat | dims [MEASURED-EST] (scene-inventory §1); friction [DEFAULT] (rubber-mat surfaces are high-grip, lit. 0.7–1.0+) |
| Cardboard enclosure walls | 4 thin boxes, static | — | (0.5, 0.005, 0.0001) | 0.0 | corrugated cardboard | dims [MEASURED-EST]; friction [DEFAULT] |
| White tape strips (eye/cable scenes) | visual only — no geom | — | — | — | paper tape | [MEASURED-EST] frames show them flush; modelling them as geometry would add false contact edges. Declared simplification. |
| SO-101 gripper fingers/pads | part of kinematic arm (Phase 4 model) | n/a (kinematic) | (1.0, 0.005, 0.0001) | 0.0 | hard plastic (printed/moulded) | friction [DEFAULT]; pads are smooth hard plastic in wrist close-ups, e.g. `frames/block_stack__ep411/wrist_first.png` [MEASURED-EST: appearance] |

## 2. block_stack

| Item | Geom | Mass | Friction | Restitution | Material | Source |
|---|---|---|---|---|---|---|
| Cube ×2 (purple; green or orange) | box 30×30×30 mm (±10 mm) | 20 g (range 10–35 g) | (0.6, 0.005, 0.0001) | 0.0 | painted wood or dense foam — appearance ambiguous in frames | dims [MEASURED-EST]; mass [MEASURED-EST] (27 cm³ × 0.7 g/cm³ painted-wood density; if foam, ≈5 g — flag: mass is the least certain value in this table); friction [DEFAULT] (wood-on-rubber-mat lit. range) |

## 3. eye_drops_to_basket

| Item | Geom | Mass | Friction | Restitution | Material | Source |
|---|---|---|---|---|---|---|
| Eye-drops carton (with bottle inside) | box 90×35×35 mm (±20%) | 30 g (20–45 g) | (0.5, 0.005, 0.0001) | 0.0 | printed cardboard + plastic bottle | dims [MEASURED-EST]; mass [PROP] (retail eye-drops carton + 10–15 ml bottle ≈ 25–40 g typical) + [MEASURED-EST] volume check; friction [DEFAULT] |
| Platform (white stand) | 3 boxes (top plate 200×160×4 mm, web, base plate), static | — | (0.4, 0.005, 0.0001) | 0.0 | smooth white plastic / painted sheet | dims [MEASURED-EST] (scene-inventory §4); friction [DEFAULT] |
| Basket | 5 thin boxes (floor + 4 walls), 260×180×70 mm, static, wall ≈3 mm | — | (0.4, 0.005, 0.0001) | 0.0 | clear ribbed plastic (PP/PET) | dims [MEASURED-EST]; friction [DEFAULT]. Ribs not modelled — declared simplification. |

## 4. ring_insert

| Item | Geom | Mass | Friction | Restitution | Material | Source |
|---|---|---|---|---|---|---|
| Board | cylinder Ø170 mm × 15 mm, static; 5 recesses NOT cut (flat top) | — | (0.5, 0.005, 0.0001) | 0.0 | pale wood / wood-effect | dims + recess count [MEASURED-EST] (scene-inventory §5). Declared simplification: recesses are ≈2–3 mm deep and only cradle the ring's start pose; Phase 3 initial poses must compensate, Phase 4 may add recess geoms globally if validation shows start-pose slip. |
| Peg | capsule/cylinder Ø16 mm, 70 mm above board, static, fixed to board | — | (0.4, 0.005, 0.0001) | 0.0 | light wood | dims [MEASURED-EST]; friction [DEFAULT] |
| Ring | torus approximated by MuJoCo torus geom (or 12-segment capsule ring if torus contacts misbehave — Phase 4 choice, logged), outer Ø45, inner Ø22, thickness 12 mm (±) | 10 g (4–15 g) | (0.5, 0.005, 0.0001) | 0.0 | painted wood/plastic | dims [MEASURED-EST]; mass [MEASURED-EST] (ring volume ≈ π(R²−r²)·t ≈ 5.5 cm³ × 0.7–1.2 g/cm³ wood/plastic density ≈ 4–7 g, upper range allows denser plastic); friction [DEFAULT] |

## 5. tool_insert

| Item | Geom | Mass | Friction | Restitution | Material | Source |
|---|---|---|---|---|---|---|
| Tray + case | box 385×270×45 mm, static; slot recesses NOT cut except the target slot (see below) | — | (0.6, 0.005, 0.0001) | 0.0 | EVA foam insert | dims [PROP] (Sealey Premier EVA tray class) cross-checked [MEASURED-EST]; friction [DEFAULT] (foam is high-grip) |
| Target slot | shallow box recess modelled as a 4-wall + floor inset in the tray top at the empty slot's location, sized to the loose tool +2 mm | — | as tray | 0.0 | EVA foam | location/shape [MEASURED-EST] (cross/T recess in first frames). Declared: only the target slot is modelled; the ≈11 seated tools are part of the static tray (they never move in any viewed episode). |
| Seated tools (≈11) | static, merged with tray geom (visual meshes optional) | — | — | — | chrome-vanadium steel | identity/labels [MEASURED-EST] (tray print, scene-inventory §6); material [PROP] (Sealey Premier extension bars are chrome-vanadium steel) |
| Loose tool (manipulated) | capsule + 2 end cylinders, or single box-with-rounded-ends composite; length 100 mm (±25), shaft Ø10 mm | 70 g (50–100 g) | (0.4, 0.005, 0.0001) | 0.0 | chrome-vanadium steel, knurled grip | dims [MEASURED-EST] (wrist close-up ep630); mass [MEASURED-EST] (steel 7.85 g/cm³ × ≈9 cm³ shaft+ends ≈ 70 g); friction [DEFAULT] (polished steel on foam/mat lit. 0.3–0.5) |

## 6. cable_clip

| Item | Geom | Mass | Friction | Restitution | Material | Source |
|---|---|---|---|---|---|---|
| Clip block | box 110×70×55 mm (±20%), dynamic-or-static: STATIC per episode at its measured initial pose (it is re-placed per episode; no viewed episode shows it moving) | — | (0.6, 0.005, 0.0001) | 0.0 | white plastic/foam block | dims + per-episode placement [MEASURED-EST] |
| Clip (on block top) | two small boxes forming the jaw/channel on the block top, static; channel width ≈8 mm (cable Ø + ≈3 mm) | — | (0.5, 0.005, 0.0001) | 0.0 | black plastic | appearance [MEASURED-EST]; channel width is a modelling choice — the real clip is a sprung jaw whose closing force is NOT modelled; outcome scoring uses the pre-registered geometric criterion (cable segment inside channel), not clip force. Declared approximation. |
| Cable | **composite capsule chain (APPROXIMATION, flagged)**: 24 capsules, each 30 mm long, Ø5 mm, joined by ball joints (damping 0.2 [DEFAULT], stiffness 0 — free hinges with joint damping only), total length ≈72 cm | total 45 g (≈1.9 g/segment) | (0.8, 0.005, 0.001) | 0.0 | PVC-jacketed copper cable | length/Ø [MEASURED-EST] (coiled layout in top views; Ø vs USB plug body); mass [MEASURED-EST] (typical USB cable 40–60 g [PROP]) segment count/length: chosen so segment length (30 mm) ≫ diameter (5 mm) for stable contacts while resolving the observed coil radius (≈40–60 mm); ball-joint chain cannot hold the cable's real bending stiffness/torsion — declared limitation, cable_clip is scored on outcome class only per the pre-registration. |
| USB connector | box 25×12×8 mm attached to chain end segment | 6 g | (0.5, 0.005, 0.0001) | 0.0 | moulded plastic + metal shell | dims [MEASURED-EST] (top views) |

## 7. Grasp model (declared modeling device, per pre-registration)

- Arm is **kinematic**: joint qpos set per recorded frame (30 fps playback
  of the 20 fps recording — interpolation is Phase 4's documented choice);
  arm geoms do not generate contact forces on objects except via the weld
  below and incidental gripper-finger contacts if Phase 4 enables them.
- **Attach**: a weld equality constraint between the manipulated object and
  the gripper frame is created when BOTH hold:
  (a) the recorded gripper trace crosses **50% closed** (normalised gripper
  joint range, closing direction), AND
  (b) the object's centre is within **3.0 cm** of the gripper centre point
  (midpoint between finger tips).
  Threshold rationale: 50% is the midpoint of the recorded close stroke —
  before it the real fingers have not yet clamped a centred object; 3.0 cm
  ≈ one cube-edge / ring-radius scale, i.e. the object must be where the
  fingers actually are. Both are GLOBAL constants for all tasks/episodes.
- **Release**: weld removed when the recorded gripper trace re-opens past
  **30% closed** (hysteresis vs the 50% attach threshold prevents
  flicker on noisy traces).
- Only the task's manipulated object is weldable (cubes: either cube, but
  one at a time — first to satisfy (a)+(b) after a close crossing).
- For cable_clip the weld attaches to the nearest cable segment instead of
  an object centre; condition (b) uses that segment's centre.
- Known artifacts (to be reported, not hidden, per honesty rule 4): weld
  grasp has no slip, no force limit, and can impart the gripper's exact
  pose to the object at attach (snap ≤3 cm possible); objects cannot be
  knocked by the arm except through the welded object.

## 8. Source-tag count (honesty summary)

Counting table cells above by primary tag: **[MEASURED-EST] 14**,
**[PROP] 3** (eye-drops carton mass, tray dimensions class, seated-tool
material — plus cable mass cross-check), **[DEFAULT] 13** (nearly all
friction/restitution values and gravity/timestep). Friction is the weakest
column: no friction was measured from the recordings (no sliding events
with measurable deceleration were extracted in Phase 1); all friction
values are literature/MuJoCo defaults and are the first candidates for the
pre-registration's global-parameter change log during validation.

## 9. Not determined in Phases 1–2

- Cube material (painted wood vs dense foam) and hence cube mass within
  5–35 g — frames cannot distinguish; affects stack stability margins.
- Exact mat dimensions (±20%) and exact camera poses (Phase 3 pose
  estimation may tighten these from lettering geometry).
- Ring recess depth; clip spring force (not modelled, see §6).
- Whether the loose tool's identity is the same SKU in every episode
  (labels suggest the tray set is fixed; the loose piece's home slot is
  the cross/T recess in all episodes viewed).
