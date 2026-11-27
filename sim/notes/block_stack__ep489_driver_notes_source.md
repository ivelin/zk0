# Driver notes — (arm_padded_v4, block_stack, 30 mm cubes)

Persistent notes per protocol §2B. Written by the solving driver; read at the
start of the next session on a similar scene. Notes inform, never bind.

## Session 2026-11-27 — block_stack ep453, arm v4, scene v4cam (folded home)

Stack facts learned before moving (from records, to be tested live):
- The arm in this harness is kinematically driven (qpos interpolation at
  2000 Hz, exactly as enforce_replay drives it). There is no servo yield to
  feel in the arm itself — what I feel is the cube's contact/friction answer.
  The cube is a free body: 30 mm, 20 g, friction 0.6 on a 0.9 mat.
- One moving jaw only. A pinch forms in two stages: moving fingertip touches
  its face, then walks the cube ~3 mm sideways into the fixed finger. Expect
  the cube to translate during the close; read its position AFTER the close,
  not before, when planning the place.
- 30 mm pinch at gripper joint ≈ 26.6–27.0° (0.464–0.471 rad). At the
  measured witness placement (REAL453_CLOSE orientation, V=(14,−10,14) mm,
  grip 32.8°) both tips sit ~3 mm off the side faces, root ~17 mm above the
  top face — closing ~3° from there should bring the moving tip to its face.
- Channel discipline for the deliverable: the replay normalises the gripper
  channel per trajectory (p5/p99). A trace that starts with a genuine
  full-close segment (folded home, joint 0 → chan 2) and a genuine full-open
  segment (joint 1.45 → chan 55) makes the normalisation the identity, so the
  replay closes exactly as far as I piloted. A partial-clamp plateau as the
  trace minimum would be re-normalised to full close in replay — do not rely
  on it.
- Jaw × cube_a (the BASE cube) is NOT in the declared contact set — only
  jaw × cube_b is. During the place, the fingers descend around cube_b while
  it sits on cube_a; a finger touching cube_a is an execution fault. Watch
  the gripper-to-cube_a distance on the way down.
- At the final settle read, the upper cube's contact set must be exactly
  {cube_a}: retreat the gripper fully clear before settling, or a lingering
  finger contact fails the criterion.

## Familiarization findings

- Approach in REAL453_CLOSE orientation is clean: unfold, hover, pre-shape
  grip to 32.8° in mid-air, vertical descent to the witness site — zero
  contacts, zero faults, IK rot err ~0.4°. The wrist view at the grasp site
  shows both fingertips alongside the cube at mid-height, matching the real
  ep453 wrist frames. The descent problem of the v2/v3 families does not
  exist in this orientation.
- FIRST CLOSE EJECTS (2026-11-27 fam): closing 0.545 → 0.524 rad in one
  15-frame ramp struck cube_b and slid it 121 mm in −y (to the front wall),
  still upright; no jaw contact ever registered at frame ends — the strike
  is mid-step, same signature as the ring seek diagnosis. The moving tip's
  closing arc meets the cube with a tangential (−y) component; a face
  touch at speed becomes a flick. Closing the last degrees as one ramp is
  wrong. Next tests: (a) micro-stepped close (~0.3° steps, state read
  between) to find the exact strike joint; (b) seek-by-translation: grip
  pre-set just above the pinch, translate the SITE along the closing axis
  in 0.5 mm steps so the tip arrives by translation, not by arc — the
  method that formed the clamp reliably on the ring.
- MICRO-CLOSE (fam run 2): cube unmoved at 0.560 / 0.550; EJECTED again on
  the 0.550 → 0.540 step, landing at the identical spot (0.2097, −0.1341).
  Speed is not the variable — the strike joint is ≈0.544.
- DIAGNOSTIC (diag_close.py, substep log of the same close): at grip 0.550
  the moving tip box (moving_jaw_so101_v1_col_b004) is already only
  0.46 mm from the cube — the witness's "3 mm" was the RENDERED mesh; the
  collision boxes meet far earlier. First touch at grip ≈0.5441, contact
  b004 at −1 µm, and within 27 substeps the cube is at 0.486 m/s in −y:
  the solver resolving the kinematic finger's micro-penetration ejects
  the 20 g cube before the fixed finger (still ~2 mm away) can arrest it.
  The pinch never gets the chance to form two-sided. Lesson: with a
  kinematic finger, NEVER close into a free cube across a gap — the cube
  must be CONFINED first (fixed finger already touching it) so the strike
  distance is ~0. New method (test 3): at the witness site, translate the
  site +y in 0.5 mm steps so the FIXED finger walks the cube into the
  moving tip, then close the last degrees with the cube confined.
- AXIS CORRECTION (fam run 4, per-body geomDistance added to the pilot):
  test 3's +y translations (2.7 mm total) produced NO contact — because
  the pocket's closing axis at the REAL453 pose is along WORLD X, not Y.
  At the witness site, grip 0.572: fixed finger (wrist_roll_follower box
  b899) is 0.52 mm from the cube on the −x side; moving tip (b004) is
  2.21 mm away on the +x side; both closest-pair directions are
  x-dominated. The tip's ejection push reads −y because the tip sweeps an
  arc (tangential velocity), not because the face normal is −y.
  Corrected confinement: translate the site +x in 0.25 mm steps — the
  fixed finger first touches (0.52 mm), then walks the cube +x across
  the 2.21 mm gap into the tip; close only once both fingers touch.
- TRANSLATION CONFINEMENT IS IMPOSSIBLE (fam run 4 + diag_fixed.py):
  translating the whole gripper moves BOTH fingers together — once the
  fixed finger touches the cube, the tip gap stays constant (2.21 mm);
  the cube can never be walked into the tip. Worse, the touch itself
  hops the cube: diag_fixed replaying the +x walk's frames 226->231
  shows first touch at frame 1/87 by b899 with a clean face-on normal
  (+1,0,0) — and the cube ALREADY at velocity (0.354, −0.378, +0.331)
  m/s: it hops up, sideways (−y, the tip-sweep direction) and +x within
  a substep or two of ANY kinematic-finger touch. In the pilot the ramp
  continued ~3 frames past the touch and knocked it to (−30, −73) mm.
  Confinement can only come from the CLOSE (pocket width shrinking).
  New harness commands walk / walkg: frame-by-frame linear moves that
  STOP on jaw contact, on cube motion > 1 mm, or on attach. Method:
  micro-close with walkg, stop at the tip's first touch, let the cube
  settle inside the pocket (one touch displaces it only ~3 mm when the
  finger stops immediately — diag_fixed final (+2.1, −2.2) mm), then
  keep micro-closing with holds between steps so the squeeze builds
  against the fixed finger instead of launching the cube.
- STOP-ON-CONTACT WORKS; SUSTAINED ADVANCE DOESN'T (fam run 5):
  walkg −0.025/50f then −0.006/30f stopped at the tip's first touch
  (grip 0.545, jaw_min_dist −0.01 mm) with cube displacement 0.0 mm —
  the first touch in this campaign that did not eject the cube. One
  extra walkg frame (−0.03 mm pen) also stable. But the next advance —
  a plain 5-frame grip ramp 0.544 -> 0.543 (0.001 rad) — launched the
  cube to (0.2035, −0.1843), Δ(−29, −171) mm. So: the touch is stable
  only while the finger does not keep advancing; ~0.05 mm of sustained
  advance into an UNCONFINED cube (fixed finger still 0.53 mm away)
  flicks it out along −y (the tip's sweep direction). The squeeze must
  therefore be built with the cube confined in x FIRST: after the tip
  touch, translate the site −x (new walkx command: no contact stop) so
  the fixed finger closes its 0.53 mm gap and touches; its touch hops
  the cube +x INTO the already-touching tip. Only then micro-squeeze.
- WALKX DIRECTION TRAP (fam run 6): after the tip touch, translating
  the site −x to bring the fixed finger in DRIVES THE ALREADY-TOUCHING
  TIP into the cube (tip is on the +x side; site −x = tip advance) —
  cube launched Δ(−11, −86) mm and tipped within 2 walk frames. The
  fixed finger is closed by translating +x (it sits on the −x side;
  +x moves it toward the cube while the tip retreats). Corrected
  sequence (run 7): tip touch via walkg -> open the grip a hair
  (walkg +0.003) so the tip is clear -> walkx +x ~0.6 mm, stop on cube
  motion: the fixed finger's touch hops the cube +x into the pocket
  (tip ~0.7 mm away) -> micro-squeeze with grip increments, cube now
  confined in x by both faces.
- FIXED-FINGER TRANSLATION TOUCH ALSO LAUNCHES (fam run 7): tip touch
  (stable) -> open +0.003 (tip gap 0.26 mm) -> walkx +x: at frame 27,
  exactly when the 0.53 mm fixed gap closed, the cube launched
  Δ(+30, −135) mm, tumbling (z 0.0368). The site path drifts −y as it
  translates (IK interpolation: site y −0.0057 -> −0.0062 over the
  walk), so the fixed finger strikes diagonally (+x, −y) and wipes the
  cube along −y. Translation touches are out. UNTried configuration
  (run 8): make BOTH gaps sub-0.1 mm WITHOUT any translation touch —
  walkx +0.45 mm stops 0.08 mm short of the fixed face (no touch, cube
  unmoved), then the grip channel alone re-touches the tip and presses
  across the 0.08 mm gap, so the squeeze builds against both faces
  almost immediately instead of accelerating the cube across 0.53 mm
  of free space (the fam-5 failure).
- RUN 8 RESULT: the sub-0.1 mm setup WORKED (walkx +0.45 mm completed,
  cube_disp 0.0, fixed gap 0.09 mm, tip 0.57 mm) but the final walkg
  touch LAUNCHED the cube Δ(+4, −142) mm. Root cause found in the
  pilot source: walkg re-solves IK for the current site and
  interpolates in JOINT space — mid-walk the site bulges −y by
  ~0.7–1.1 mm (endpoint err reads 0.01 mm), so a "grip walk" is also a
  −y finger sweep along the cube's faces. The stable fam-5/7 touches
  started from the descent-converged pose (bulge ~0). The plain `grip`
  command freezes q5 exactly (append_interp with cur_qdeg) — no IK, no
  bulge. RUN 9 RECIPE: descend at grip 0.572 -> walkx +0.45 mm FIRST
  (bulge harmless, tip 2.66 mm away) -> close with micro-`grip`
  increments (−0.0005 rad / 4 frames; peak per-frame tip advance
  ~0.017 mm, near touch −0.0002/3f ≈ 0.007 mm — below the stable-touch
  penetration) reading cube_b after every increment -> press past the
  touch the same way; the fixed face is 0.08 mm away, so the one-sided
  press distance is ~7x shorter than fam-5's fatal press. Attach =
  closedness >= 0.50 + jaw x cube_b contact x 3 frames (bookkeeping
  only; certification-mode weld is unreachable in pilot AND replay —
  enforce_replay.py lines 23-25 — so the lift must be a true friction
  carry; test it with a +z move and watch cube_b z follow).
- RUN 9 — THE GENTLE TOUCH + ATTACH WORK; THE PRESS STILL SQUIRTS.
  Recipe as planned (descend 0.572 -> walkx +0.45 -> micro-grip with
  q5 frozen: bulk −0.002/8f to 0.552, ramps 0.5475/50f, 0.5435/50f,
  0.5395/60f). Site stayed EXACTLY frozen through every grip command
  (0.2141, −0.0055, 0.0229) — the q5-freeze eliminated the bulge.
  The 0.5395 ramp touched at ~frame 490 (pen −0.02 mm) and the attach
  bookkeeping FIRED at frame 493 (grasp_contact_true, closedness
  0.674) with cube_b UNMOVED at spawn (0.2323, −0.0131, 0.0149) — the
  campaign's first contact-true attach with zero cube displacement.
  THEN the press (grip 0.5375/40f, peak ~0.006 mm/frame) squirted the
  cube at ~frame 509 (joint ≈0.5393, tip pen ≈ −0.036 mm): contact
  lost (event frame 519), cube Δ(+6.5, −152) mm, settled flat. The
  fixed face (0.07 mm away) never engaged: the escape is along −y,
  out the OPEN side of the pocket — x-confinement is irrelevant to a
  y-escape. Next: substep diagnosis of the press (diag_press.py) —
  cube velocity decomposition + tip contact normal/point in the first
  200 substeps, press variants: 10x slower press, +y site
  compensation, ±yaw, ±3 mm site height. Attempt accounting: runs
  1–9 are all familiarization (grip/physics tests); no solving
  session (grasp->lift->carry->place with completion intent) has
  been flown yet — 10 attempts remain.
- DIAG RESULTS (diag_press.py / diag_press2.py / diag_press3.py,
  offline, from the rebuilt run-9 gentle-touch state — the rebuild
  reproduced frame 501 exactly, attach event included):
  * THE TIP CONTACT IS THE CUBE'S (+x, +y, TOP) CORNER — contact pos
    (0.2473, 0.0019, 0.0299) = the corner point itself; normal
    (−0.832, −0.091, −0.547). The press wedges that corner; the cube
    pivots over its bottom edges and flips out −y/+z.
  * THE EJECTION IS A JOINT THRESHOLD, NOT A RATE EFFECT: press at
    run-9 rate and at 10x slower both detonated at joint ≈ 0.53918
    (cube dead still, disp 0.00, until that joint; then v jumps to
    (−0.29, −0.33, +0.23) m/s within a few substeps, max |vy| 1.79).
    ±y site chases only moved the threshold. Slower ≠ safer.
  * YAW IS A DEAD END: yaw −3° RETREATS the tip (gap 0.26 mm — the
    diag_press "surviving press" never touched the cube); when the
    staged press re-reached it, ejection was worse (425 mm, off the
    mat). yaw +3° drives the tip into the corner (ejects during the
    rotation itself).
  * diag_press3 (depth sweep) is mostly harness artifact: absolute
    IK gotos flip IK branches at lowered targets (gripper lands
    ~16 cm off); its close (0.0005 rad/frame) is too coarse to
    reproduce the gentle touch. One clean datum: the −4 mm descent
    itself was fault-free and the tip gap at grip 0.572 shrank
    2.16 -> 0.87 mm (fixed 0.54 -> 0.23) — the tip box sits closer
    to the cube lower down. Depth must be tested in the PILOT with
    relative moves (run 10), not by diag gotos.
- RUN 10: the −4 mm post-descent SLIDE struck the cube (Δ(+9.5,
  −147) mm, no fault — a declared jaw contact): at grip 0.572 with
  the fingers alongside (fixed gap 0.52 mm), sliding down 4 mm drove
  the fixed side's wider lower geometry (and/or the site's −0.4 mm
  y-drift) into the cube. The pocket exists only at the witness
  height; depth-via-slide is dead. (A direct deep descent with the
  grip open remains untested in the pilot.)
- UNTESTED DECISIVE QUESTION (diag_press4): what normal force does
  the stable gentle touch actually carry (mj_contactForce), and can
  a lift from the touch state — or from a press stopped just below
  the 0.53918 threshold — wedge/hold the cube? Every press so far
  aimed at a two-sided pinch; the single-sided corner contact plus
  lift dynamics has never been measured.

## Attempt findings

(to be appended per attempt)

## Diag 4–7 + session conclusion (2026-11-27, same session)

- DIAG 4 (force + lift, from the rebuilt run-9 frame-501 touch state;
  the rebuild reproduced the pilot state exactly, attach included):
  * The GENTLE TOUCH ALREADY CARRIES ~49.9 N of normal force
    (mj_contactForce; the cube weighs 0.196 N) at penetration
    −0.016 mm, on the single corner contact (b015 × cube_b, contact
    pos = the cube's (+x, +y, TOP) corner point, normal
    (−0.832, −0.091, −0.547)). Tangential components are ~75% of the
    normal. Force was never the problem — the contact is a cocked
    mousetrap: ~50 N of constraint force balanced on one corner.
  * Pressing in micro-stages: disp stays 0.000 mm; at joint 0.53930
    the contact splits (b004 joins; total 65.8 N) — still all corner
    contacts (z = 0.0299). Static stability, dynamic trap.
  * LIFT DETONATES from BOTH states: touch-state lift +5 mm → cube
    Δ171 mm; pressed-state lift → Δ236 mm. The lift's joint-space
    path drags the loaded tip across the corner and the stored force
    releases directionally. NO LIFT IS POSSIBLE from this pose.
- DIAG 5 (y-shift + micro-descend):
  * Site y-shifts at the witness height: −4/−6 mm strike the cube
    during the shift itself (the tip box's y-edge sweeps the cube's
    corner); −2 mm shifts clean but the close then flicks the cube
    via the box's SIDE edge (cube ends against the front wall);
    +2 mm makes the tip miss the cube in y entirely (0.040 rad of
    close, no contact). The y-window for a face contact, if any, is
    narrower than ±2 mm and bracketed by two failure modes.
  * Micro-descend from the touch state: detonates at the FIRST
    0.1 mm step (Δ171 mm). Any site motion in corner contact = end.
- DIAG 6 (static pitch scan, advisory): pitch −35°..+35° about world
  y at the witness site. Negative pitches open the pocket (no touch
  by joint 0.50); at every pitch ≥ −5° the moving jaw's closest
  point on the cube is the (+15, +15, +15) mm CORNER, and positive
  pitches drive the FIXED side into penetration (gap −0.68 mm at
  +15°, −3.46 mm at +35°). Pitch does not move the contact off the
  corner.
- DIAG 7 (static family scan, advisory): deep sites at REAL453
  (z −2/−4/−6 mm): fixed gap stays 0.53 mm at all depths (the fixed
  side does NOT engage deeper down — run 10's strike was the move's
  y-drift, not a wider fixed geometry); at −4 mm the moving side is
  already at −0.14 mm at grip 0.572 (a deep descent arrives
  intersecting the cube); witness is the same corner at every depth.
  REAL411 and RING at the ep453 witness site: IK rot err 4.7°/5.7°
  (their own witness sites were not hunted — see untested list).

## SESSION CONCLUSION — driver-authored failure map

The grasp never stabilised in ANY tested configuration, so no
solving attempt could start: every live sequence ended in
familiarization at the grasp stage. Solving attempts used: 0 of 10
— not for lack of trying the grasp (10 live runs + 7 diagnostics),
but because the failure is at the prerequisite every attempt shares.

Root mechanism (all measured at the engine's resolved state):
1. The padded moving-jaw paddle meets a 30 mm cube CORNER-FIRST in
   every configuration tested: REAL453 witness pose dynamically;
   pitch −35..+35°, depth 0..−6 mm, y-shift −6..+2 mm statically;
   yaw ±3° dynamically. The closest point is invariantly the cube's
   (+15, +15, +15) corner (relative), contact normal tilted ~33°
   from the face normal.
2. The kinematic grip drive cannot yield, so the corner contact
   stores ~50 N at first touch (250× the cube's weight). The cube
   sits still only while NOTHING moves.
3. Every subsequent action detonates it at a JOINT THRESHOLD
   (press: joint ≈ 0.53918, rate-independent over 10×) or on first
   motion (lift +5 mm, descend 0.1 mm, yaw, lateral shift): the
   cube flips out over its bottom edges through the pocket's open
   ±y side; x-confinement cannot arrest a rotational escape.
4. WHY THE REAL GRASP WORKED: calibration-20261127/real_grasp_fk
   records the real ep453 close completing with servo_yield_deg
   10.26 — the real STS3215 BACKED OFF ~10° under load and let the
   paddle settle. This harness (pilot AND enforce_replay) drives
   qpos kinematically; the v4 forcerange exists in the model but a
   kinematic drive bypasses it. The compliance the real grasp
   depended on is structurally absent here. This is an
   engine/admission fact about the harness, not a scene verdict:
   per §2B the scene is NOT declared infeasible — the method the
   real robot used cannot be expressed by the current drive.

What the next driver should try (in order):
- (a) A COMPLIANT grip drive in the harness (actuated gripper with
  the v4 forcerange actually in the loop, or a scripted yield:
  on jaw contact, let the grip joint retreat under load the way
  the real servo did). If the paddle can settle, the whole
  REAL453 recipe (run 9) likely completes: attach at zero
  displacement was already achieved; only the press/lift broke.
- (b) A contact-model review of the padded paddle vs the real jaw:
  the real jaw's contact patch may not be the padded boxes' corner.
  (Corrections-lane question, not a driver call.)
- (c) Untested, with predictions: REAL411/RING pocket hunts at
  their own witness sites (IK at the 453 site was poor; prediction:
  same corner — the paddle is an arm property); top-down / yaw-90
  grasp (prediction: corner on the ±y faces instead); wall-
  confinement (push cube_b to the −y wall, grasp in the corner;
  prediction: blocks the −y translation but not the +z flip, and
  the lift drag remains).
- (d) If (a) lands: re-run the run-9 recipe verbatim — descent at
  0.572, walkx +0.45 mm FIRST, q5-frozen grip ramps only (never
  walkg — its IK bulge sweeps −y), attach, then press WITH YIELD
  and lift. Channel discipline per the notes at the top of this
  file (folded start + full-open segment; verify with
  enforce_replay._closedness_trace before any replay).

---

## HARNESS CHANGE 2026-11-27 — compliant grip drive (grip-drive lane)

Option (a) above is IMPLEMENTED. `grip_drive.py`
(`gripdrive/1.0-20261127`, beside this folder) is now the single
drive configuration shared by `pilot453_v4.py` and
`enforce_replay.py`: the gripper joint is driven by the model's
position-servo actuator (`data.ctrl`), force-capped at the derived
±1.9123 N·m (STS3215 C001 stall 19.5 kg·cm); the arm joints stay
kinematic. Gain = the arm model's system-identified kp 18.45701 /
kv 2.73117 (so101_new_calib) — NOT fitted (see calibration below).
Pilot traces still store COMMANDED joint values; `state` now also
reports `joint_actual_rad` and `grip_yield_deg` (actual − commanded,
positive when the servo is held back). Seal implication: harness
element changed — (scene, arm, engine, harness) tuples citing the
old harness are superseded for future work; history stays history.

**Yield calibration — attempted as specified, produced NO gain.**
Static pinch at the run-9 witness (REAL453_CLOSE, V=(14,−10,14) mm,
ep453 v4cam scene), quasi-static ramp to command 16.71° (rig 2.36°
+ A3 offset 14.35°), target settled 26.97° (the recorded 10.26°
yield). Sweep kp 0.25…72 (`grip-drive-20261127/grip_calibration.
json`): kp ≤ 0.5 cannot close the jaw to the cube at all; kp ≥ 1
reaches the cube (~9.8 N peak) but the sustained capped press cams
it out of the pocket's open −y side (slid 12.3–13.1 mm, no tip) —
no contact at settle at ANY gain, so no pinch yield exists to fit.
First contact is the same corner as run 9 (pair b000 × cube_b at
(+15,+15,+15) mm, normal (−0.83,−0.09,−0.55), at grip ≈31.1°). The
fingertip face never reaches the cube's side face: the pinch
prerequisite fails geometrically at every gain. This is option
(b)'s territory (contact model), not a gain problem.

**Diagnostic re-fly of the run-9 recipe (2026-11-27, this drive).**
- Verbatim rebuild (frames 1..501): NO contact, NO attach. The
  servo's free-space steady-state error at this pose is ~0.105–
  0.13° (measured after a 200-frame hold — it does not converge
  away), so the jaw stops ~0.1–0.2 mm short of where the kinematic
  drive put it; the recipe's precision is absorbed. Touch force
  0.0 N (rigid drive: 49.9 N at its touch); cube unmoved.
- Press-through (0.0005 rad stages from the rebuilt state): the
  touch + attach DO reproduce at commanded 0.5370 (5 stages past
  the recipe's end): contact b015 × cube_b, −0.025 mm, 48.955 N,
  cube displacement 0.0, attach fired, observed yield 0.132°.
  Note stage 4's command (0.5375) was already past the rigid
  detonation joint 0.53918 with nothing happening — what matters
  is the ACTUAL joint, which lags the command by the yield.
- ONE more stage (commanded 0.5365): DETONATION, cube Δ173.6 mm —
  the same corner instability as the rigid drive. The torque cap
  does not defuse it; the instability is in the contact geometry.
- Lift from the stage-5 attach state: +5 mm — the cube does NOT
  rise; the single corner contact slides off (force → 0) and the
  cube stays at spawn, unmoved (benign — the rigid drive's lift
  detonated it Δ171 mm). +45 mm traverse knocks it 32.1 mm
  sideways, z never changes. THE LIFT DOES NOT HOLD.

**Guidance for the next driver.** The drive is no longer the
block; the corner-only contact patch is (option (b), corrections
lane). If you fly this recipe again before (b) lands: budget
~+0.0025 rad of press past the recipe's end to reach the touch
(servo error), expect attach at commanded ≈0.5370 with ~49 N on
the corner, and know that the press budget past the touch is
under 0.0005 rad before detonation and the corner contact cannot
friction-carry the lift. Records: `grip-drive-20261127/`
(calibration, pinch diagnostic, ep411 validation note, canary
regression — digit-identical to the v4 records — and the three
re-fly diagnostics).

---

## 2026-11-27 — SMOKE TEST (agy, gemini-4-neon-xhigh)

- **Prior notes confirmed readable:** Read `DRIVER_SEAT.md` and all prior driver notes above in full (familiarization runs 1–10, diag 4–7 corner-contact failure map, and the `gripdrive/1.0-20261127` compliant grip drive update and re-fly guidance).
- **Actions taken (smoke test only; no task attempt, gripper untouched at `joint_rad: 0.0`):** Drove the running seat server exclusively through `seat.py`, reading each JSON reply before the next command: `phase agy smoke - familiarization` → `init` → `move 0 0 0.05 20` → `goto 0.25 0 0.30 40` → inspected all three live-watch renders → `goto -0.15 0 0.29 40` → `init`.
- **What the renders showed (at hover `[0.2494, -0.0008, 0.2953]`):** `view_topcam.png` showed the yellow/black SO-101 arm extending from the left wall to hover directly above the green cube (`cube_b`) near the purple cube (`cube_a`) on the dark blue-gray walled mat; `view_frontcam.png` showed both cubes undisturbed on the mat with the arm rising upright on the left edge (gripper above frame); `view_wristcam.png` showed the closed yellow gripper jaws looking forward across the mat toward the far tan arena wall corner.
- **Feed & fault check:** All four files in `live-watch/` updated cleanly; final `status.json` reported `increment: 5`, `phase: "agy smoke - familiarization"`, `frames_in_trace: 1`, `last_fault: null`, cubes unmoved at spawn, no `"error"` in any reply, and `fault_rewound: null` (no refusable/rewound fault) on all motion replies.


---

## 2026-11-27 — AGY SOLVE SESSION 1 (gemini-4-neon-xhigh) — PLAN

- **Intent & Approach:**
  1. **Familiarization first** (`phase familiarization - ...`, unscored): probe the workspace, jaw geometry, and compliant servo (`gripdrive/1.0-20261127`) live using the three registered camera views (`topcam`, `frontcam`, `wristcam`) and JSON telemetry (`jaw_dist_fixed_*`, `jaw_dist_moving_*`, `joint_actual_rad`, `grip_yield_deg`).
  2. **Key hypothesis to test first (centered, wide-open, deeper side-face grasp):** In the prior driver's `real453` witness pose (`V=(14,-10,14) mm`, `site_y ≈ -0.0055`, `site_z ≈ 0.0229`), `site_y` was offset ~+7.6 mm from `cube_b` (`y = -0.0131`) and the ~20° finger tilt placed the moving jaw tip near `z = 0.030` (the cube's top edge), causing corner-first contact at `(+15,+15,+15) mm`. I will test descending with the gripper **wide open** (`joint = 1.45 rad`), **centered in Y** on `cube_b`, and **lower in Z** so both jaw faces engage `cube_b`'s vertical side faces rather than its top corner, then closing gently with `grip` (keeping arm `q5` frozen) to feel whether the compliant servo establishes a stable two-sided friction pinch that survives a vertical lift.
  3. **Fallback branches to explore if needed:** (a) small `yaw` adjustments to square the jaws to `cube_b`'s faces, or larger `yaw` rotations (±90°, 180°) to pinch across different faces / reverse moving-vs-fixed side; (b) `real411` and `ring` orientation families at their own centered pockets; (c) wall-assisted confinement if free-space pinching cams the cube out.
  4. **Solving (up to 10 attempts, `phase attempt N - ...`):** Once a repeatable friction grasp + lift is established in familiarization, fly a full closed-loop pick-lift-carry-place-retreat-settle sequence from fresh `init`, watching `grip_to_cube_a_min_mm > 0` throughout placement (considering a few-mm controlled drop above `cube_a` if jaw tips extend near `cube_b`'s bottom) and retreating fully clear before `settle`.
  5. **Export, Distillation & Certification Replay:** On live completion, export `agy1_live_trace.npz`, distill a smooth trajectory `agy1_cleaned.npz` preserving intent, attach plateau, and grip channel discipline (`p5 = 2.0` from ≥6% frames at `joint <= 0.10`, `p99 = 55.0` from ≥2% frames at `joint = 1.45`), and run `enforce_replay.py ... --mode=certification --object=cube_b` to `agy1_replay.json`.
- **What I want to observe:**
  - Actual jaw-to-`cube_b` closest vectors (`jaw_dist_fixed_dir`, `jaw_dist_moving_dir`) and closest geoms as a function of Y centering, Z depth, `yaw`, and orientation family (`real453`, `real411`, `ring`).
  - Whether centered/deeper side-face contact eliminates the corner wedging/detonation and allows the compliant servo (`±1.9123 N·m`) to yield smoothly (`grip_yield_deg > 0`) and friction-carry `cube_b` on lift without slip.
  - Jaw-to-mat clearance at deeper Z and `grip_to_cube_a_min_mm` clearance during placement onto `cube_a`.
- **Help wanted from the engine:**
  - Exact per-step JSON telemetry (`jaw_dist_fixed_*`, `jaw_dist_moving_*`, `joint_actual_rad`, `grip_yield_deg`, `grip_to_cube_a_min_mm`, `attached`, `events`, and `fault_rewound` rollback diagnostics) plus fresh `topcam`/`frontcam`/`wristcam` renders after every increment, and `settle` criterion previews.


---

## 2026-11-27 — EVIDENCE NOTE from the contact-model review (orchestrator relay, advisory only)

The contact-model review lane has landed its verdict (measured on the VM stack against
this scene's arm/scene build). Full record now in this seat tree at
ENF/contact-review-20261127/ (VERDICT.md, arc_truth.json, real_patch_evidence.json,
contact_params.json). Facts, for the driver to use or set aside as its own exploration
dictates:

1. **Verdict (B) — RENDERED TRUTH.** The moving jaw's corner-only contact on the cube is
   the rendered jaw's own geometry, not a box-decomposition artifact. Along the closing
   arc the touching boxes lead the rendered surface by only **0.02–0.10 mm of tip travel
   (0.02–0.08° of grip)**, with the same corner/edge character. No finer decomposition is
   coming — its ceiling would be that 0.02–0.10 mm change in contact timing.
2. **Fingertip face vs side face along the prior driver's arcs:** at first contact the
   rendered fingertip's face is still **3.45 mm** (run-9 witness pose) / **3.48 mm**
   (pure witness pose) away from the cube's side face below its top edge, and never
   comes within **2.7 mm** of that face anywhere along the arc. The fingertip arrives
   from above, at the cube's top edge.
3. **Real ep453 recording — what actually differs:** the recorded close is itself a
   near-pure gripper sweep at a frozen arm pose — frames 75→81 the arm joints move
   **≤ 0.35°** each while the gripper state sweeps **−19.99°**, and the cube does not
   move measurably. The difference from the sim tests is the pose the descent had
   already reached before the sweep: in the wrist frames the visible fingertip is
   **beside the cube's side face at cube height, tip pointing at the face with a small
   gap**, before the pinch forms; at pinch, both fingers are alongside and the cube's
   top is below the finger roots. First touch in the real close is fingertip-to-side-
   face. The yielded pinch (gripper state plateau **12.62°** against a **2.36°**
   command — yield **10.26°**) then holds steady through the lift (frames 81→90,
   gripper state steady within 0.07°).
4. **Contact parameters are unmeasured engine defaults.** Every finger geom in the v4
   chain carries MuJoCo's built-in defaults (friction 1.0 / 0.005 / 0.0001, solimp,
   solref, margin 0) — set nowhere in the scene/arm/build chain. The model's corner
   contact is therefore rigid: recorded dynamic events reach **~49.9 N of constraint
   force at 0.016 mm penetration**. The actuated grip drive models the servo's torque
   cap; it does not model fingertip–cube interface compliance, which the review names
   as the second physical ingredient the real grasp uses and the model lacks — and
   whose value cannot be derived from any recording on file (it would need
   instrumented press/slide tests on the physical rig). The review's bottom line:
   of the two missing ingredients, the one reachable from this seat is the approach
   pose — fingertips alongside the side faces before the squeeze.


---

## 2026-11-27 — ORCHESTRATOR ADDENDUM (founder strategy update 09:05 CT — tiered driving; added end-of-session duty)

This session is **TIER 1** of a tiered driver search: the top tier (you, Neon XHigh)
solves first and produces the knowledge the next tier down executes from. When a
lower tier later drives the next block scene, it will work from YOUR distilled
record instead of redoing your exploration. Your driving task, budget, milestones,
and stop conditions are unchanged. One duty is ADDED to your end-of-session work:

**Write a structured SCENE BRIEF for ep453 + a BLOCK-TASK PLAYBOOK**, as a clearly
headed section in your notes entry AND in your session summary file
(`agy1_session_summary.md`), containing:

1. **Plan** — the approach you chose and why, in the order you would execute it again.
2. **Measured failure map** — everything that failed, with the numbers you measured
   (poses, joint/grip values, forces, distances, displacements, frame counts), each
   tied to the configuration that produced it. Include dead ends and why they are dead.
3. **Working approach recipe** — if a grasp/lift/place sequence worked (even
   partially), the exact repeatable recipe: orientation family, site targets or
   offsets, grip commands and frame counts, the order of operations, and the
   telemetry signatures that told you it was working. If nothing worked, say so
   plainly and give the closest approach with its measured shortfall.
4. **Scene / arm / object facts** — anything a fresh driver would otherwise have to
   rediscover: cube positions and sizes as you found them, clearances, camera
   behaviour, seat/pilot quirks, channel discipline facts, fault/rollback behaviour.

Write it for a competent driver who has NOT seen this session: no unexplained
references to your own exploration order, every number grounded in something you
measured this session or read in these notes. This brief is a durable artifact —
it will be kept alongside the scene it describes and reused across tiers.


## 2026-11-27 — AGY SOLVE SESSION 1 (gemini-4-neon-xhigh) — OUTCOME & ENGINE DIAGNOSIS

### Stop Condition & Summary
- **Stop condition reached:** **(c) Blocked by an engine/harness fact** (NOT a scene infeasibility verdict; per protocol §2B, the physical scene and jaw geometry DO support clean friction grasps and lifts once the three measured engine/harness defects below are removed).
- **Solving attempts used:** **2 of 10** (following 7 unscored familiarization runs + read-only MuJoCo contact/solver diagnostics).
- **Milestones reached:**
  - **Attach:** YES — contact-true `grasp_contact_true` achieved with `0.0 mm` cube displacement in familiarization (top-down `yaw -96` at `z = 0.014` frame 146, and `z = 0.018` frame 169) and in **Attempt 1** (top-down `yaw -96`, `z = 0.018`, frame 148, `closedness = 0.599`).
  - **Lift / Place / Settle:** NO — blocked by three coupled engine/harness physics defects verified both live on the seat and via read-only counterfactual checks.
- **Candidate / Export:** None (no live completion; no trace exported; zero stack, scene, harness, or parameter files modified).

### Plan Outcome & Familiarization Findings
1. **Testing centered Y + lower Z in `real453` / `real453+yaw-5.7` / `real411`:**
   - `real453` centered Y (`site = [0.212, -0.0164, 0.015]`, wide open `1.45 -> 0.70`, `jaw_dist_fixed = 2.16 mm`, `jaw_dist_moving = 10.87 mm`, `dir = [-0.008, 0.0, -0.007]`): closing `0.70 -> 0.55` (20f) flung `cube_b` `-148 mm` in `-Y` (`[0.2389, -0.1615, 0.0149]`).
   - `real453+yaw-5.7` squared (`site = [0.212, -0.0138, 0.015]`): `grip 0.62 -> 0.60` (20f) flung `cube_b` `-72 mm` in `-Y` (`[0.2048, -0.0852, 0.0151]`).
   - `real453+yaw-5.7` low Z (`site = [0.2135, -0.0138, 0.010]`, right above mat floor — mat fault measured at `z = 0.008`, frame 82, `mat x wrist_roll_follower_so101_v1_col_b867`): `grip 0.62 -> 0.59` (20f) still flung `cube_b` `-108 mm` in `-Y` (`[0.2425, -0.1209, 0.0149]`).
   - `real411` (`site = [0.213, -0.0139, 0.015]`, `wrist_roll = -0.8°`, opposite roll `+6.6°` and opposite jaw edge `b029`): `grip 0.65 -> 0.58` (35f) **still** flung `cube_b` `-113 mm` in `-Y` (`[0.224, -0.1261, 0.0149]`). Reversing yaw, roll, and jaw side did not reverse the `-Y` launch direction — ruling out jaw geometry/roll asymmetry as the root cause.
2. **New orientation family discovered — top-down vertical pinch (`real453+yaw-96` / `yaw-105`):**
   - Rotating `yaw -96` puts the 5-DOF arm at `q_deg = [-0.8, 9.0, 13.4, 70.8, -89.3]` (`pitch = 93.2°`, nearly straight vertical; `wrist_roll = -89.3°`, pinching cleanly along world `Y` with the moving jaw on the `-Y` side away from `cube_a`, `grip_to_cube_a_min_mm = 9.8–16.9 mm`).
   - At `z = 0.018` (`yaw -105`, slightly rotated `-98.9°`), slow closing (`0.75 -> 0.68 -> 0.64 -> 0.58 -> 0.50`) walked `cube_b` calmly without detonation (`jaw_contact: true`, `jaw_min_dist_mm = -0.01 mm`), drifting `+17.6 mm` in `+X`.
   - At `z = 0.014` (`yaw -96`), `grip 0.72 -> 0.60` (30f) reached `grasp_contact_true` (frame 146, `closedness = 0.61`, `cube_b` unmoved, `jaw_min_dist_mm = -2.98 mm`, `grip_yield_deg = +0.775°`), then a `+20 mm` lift immediately flung `cube_b` `-166 mm` in `-Y` (`[0.2482, -0.1789, 0.0149]`, `jaw_contact_lost_10_frames` at frame 168).
   - At `z = 0.028` (`yaw -96`), `grip 0.58 -> 0.48` (30f) launched `cube_b` `+1.61 m` in `+Y` straight through the fixed jaw (`[-0.0006, 1.5965, 0.0198]`).
   - At `z = 0.020` (`yaw -96`, `jaw_dist_fixed = -0.61 mm`), `grip 0.68 -> 0.62` (20f) launched `cube_b` `-1.45 m` in `-Y` straight through the moving jaw (`[0.1272, -1.4617, 0.1031]`) while kicking the moving jaw **inward** (`joint_actual = 0.5892` vs `cmd = 0.62`, `grip_yield_deg = -1.764°`).
   - At `z = 0.018` (`yaw -96`, `jaw_dist_fixed = +0.92 mm`), `grip 0.65 -> 0.63` (15f) reached `grasp_contact_true` (frame 169, `cube_b` unmoved, **`jaw_min_dist_mm = -0.81 mm`**, **`grip_yield_deg = +0.101°`** — zero resisting force despite `0.81 mm` penetration with `+0.92 mm` air behind `cube_b`!), and the very next micro-step `grip 0.63 -> 0.62` (15f) launched `cube_b` `-467 mm` in `-Y` (`[0.2164, -0.4798, 0.0185]`, `grip_yield_deg = -0.228°`).

### Per-Attempt Log (2 of 10 used)
- **Attempt 1 (`attempt 1 - top-down yaw-96 vertical grasp and lift`):**
  - Fresh `init` -> hover `[0.2323, 0.011, 0.08]` wide open (`1.45`) -> `yaw -96 30` -> descend to `site = [0.2323, 0.0104, 0.018]` (`q_deg = [-0.8, 9.0, 13.4, 70.8, -89.3]`, `jaw_dist_fixed = 1.29 mm`, `grip_to_cube_a_min_mm = 9.79 mm`) -> `grip 0.68 25` -> `hold 10` -> `grip 0.642 20` -> `grip 0.638 12`.
  - **Attach milestone reached:** `attached: true`, event `grasp_contact_true` at frame 148 (`closedness = 0.599`), `cube_b_manip = [0.2323, -0.0131, 0.0149]` (0.0 mm displacement), `jaw_min_dist_mm = -0.75 mm`, `grip_yield_deg = +0.033°` (confirming ghost zero-force contact under Defect 1).
  - **Lift outcome:** `move 0 0 0.025 25` (+25 mm vertical lift) lost jaw contact on the first frame (`jaw_contact_lost_10_frames` at frame 168) and flung `cube_b` `-159 mm` in `-Y` to `[0.2469, -0.172, 0.0164]` (`joint_actual = 0.6335`, `grip_yield_deg = -0.259°`, `fault_rewound: null`). Abandoned.
- **Attempt 2 (`attempt 2 - centered real453+yaw-5.7 gentle close`):**
  - Fresh `init` -> hover `[0.205, -0.0131, 0.08]` wide open (`1.45`) -> `yaw -5.7 15` -> descend to `z = 0.015` -> slide `+8.5 mm` in X to `site = [0.2135, -0.0137, 0.015]` (`q_deg = [4.1, -24.7, 54.9, 39.8, 6.7]`, `jaw_dist_fixed = 0.79 mm`, dead-centered on `cube_b_y = -0.0131`) -> `grip 0.66 25` -> `hold 10` -> `grip 0.60 25` (`jaw_dist_moving = 1.19 mm`) -> `grip 0.57 25`.
  - **Outcome:** On first contact during `0.60 -> 0.57`, `cube_b` was flung `-96.5 mm` in `-Y` to `[0.2506, -0.1096, 0.0149]` (`attached: false`, `jaw_contact: false`, `fault_rewound: null`). Abandoned.

### Root Cause — Three Coupled Engine/Harness Defects (Measured & Causally Proven)
Read-only MuJoCo inspection (`mj_contactForce`, `d.efc_force`, `d.qacc`, `mj_jacSite`) of the exact unmodified scene (`scenes-v4/block_stack__ep453_v0__v4cam.xml` + `arm_padded_v4cam.xml` + `grip_drive.py` + `pilot453_v4.py`/`enforce_replay.py` stepping semantics) uncovered why **both** this session and the prior driver's session (including `grip_calibration.json`) failed — and disproved the prior conclusion that contact geometry blocked the pinch:

1. **Defect 1 (Primary grasp/contact corruption — unexcluded `base × shoulder` internal mesh interpenetration injecting ±100–408 g phantom acceleration):**
   - `enforce_replay.py` / `pilot453_v4.py` exempt chain-adjacent `base × shoulder` contacts from `enforcement_scan` because the pan motor mesh sits inside the base housing mesh (`-22.40` to `-27.87 mm`), noting they are "mechanism-internal, not task physics."
   - However, `arm_padded_v4cam.xml` has **no `<contact><exclude body1="base" body2="shoulder"/>`**, so MuJoCo's physics solver still resolves **12 `base_col_0/1/2 × geom19` contacts** carrying `16–23 kN` per constraint row (`69,979–88,341 N` normal force each, **~966 kN total** at the Run-9 witness!).
   - Because `set_arm()` / `set_arm_pilot()` only overwrite `qpos[0..4]` and `qvel[0..4] = 0` before `mj_step()`, all 5 arm joints remain dynamic hinges in the same solver island as `jaw × cube_b` during `mj_step()`. The ~966 kN `base × shoulder` interpenetration drives wild facet-dependent phantom angular acceleration `qacc_pan` between **`-20,387` and `+21,775 rad/s²`**, producing phantom gripper-site accelerations across the `pan = -12°..+6°` pick-to-place corridor of **`min 95.0 m/s² (9.7 g), median 1,593.5 m/s² (162 g), max 4,003.7 m/s² (408 g)`**:
     - Run-9 witness (`pan = +0.98°`): `qacc_pan = +3,569.9 rad/s²`, `a_site = [+24.3, -622.2, -5.2] m/s²` (**`63.0 g` in `-Y`**).
     - Centered `real453` (`pan = +4.2°`): `qacc_pan = +4,821.3 rad/s²`, `a_site = [-31.1, -842.4, -8.8] m/s²` (**`86 g` in `-Y`**).
     - `real411` (`pan = +4.5°`): `qacc_pan = +10,923.4 rad/s²`, `a_site = [-121.2, -1,903.1, -11.7] m/s²` (**`194 g` in `-Y`**).
     - Top-down `yaw -96` at `cube_b` (`pan = -0.7°`): `qacc_pan = +3,618.8 rad/s²`, `a_site = [+95.3, -698.5, +15.4] m/s²` (**`71 g` in `-Y`**).
     - Top-down `yaw -96` at `cube_a` (`pan = -11.0°`): `qacc_pan = -4,713.5 rad/s²`, `a_site = [-168.3, +957.6, +38.5] m/s²` (**`98 g` in `+Y`**).
   - **Direct consequences:**
     - In `real453`/`real411`/Run-9 witness, every jaw touch sees a `63–194 g` phantom `-Y` wipe, explaining why even a clean `(+1, 0, 0)` fixed-jaw touch launched `cube_b` in `-Y` and why `grip_calibration.py` cammed `cube_b` `-Y` at every `kp`.
     - In top-down `yaw -96`, the `-71 g` `-Y` phantom acceleration pulls the `-Y` moving jaw *away* from `cube_b` during `mj_step()`, so MuJoCo treats `moving_jaw × cube_b` as separating (`efc_force = [0, 0, 0, 0]`, `mj_contactForce = 0.0 N` across 9–20 contacts from `-0.036 mm` down to `-1.169 mm`!). The `1.0 mm`-thick voxel boxes march ghost-like into `cube_b` until `-1.17 mm` (`ncon` jumps `20 -> 486`), where their contact normal flips `180°` into the `-71 g` acceleration and blasts `cube_b` straight through the jaw (`-0.47` to `-1.46 m`).

2. **Defect 2 (Secondary phantom acceleration — unset `data.ctrl[0..4] = 0.0`):**
   - `set_arm()` (`enforce_replay.py:188–192`) and `set_arm_pilot()` (`pilot453_v4.py:161–165`) never write `data.ctrl[0..4]`, leaving all 5 active STS3215 arm actuators (`±1.9123 N·m`) pulling toward `q = 0` on every substep. Even with `base` collisions disabled in-memory, this injects `a_site = [+59.4, +0.5, +14.8] m/s²` (`~6 g`), walking `cube_b` `+49 mm` out of the top-down pocket (`0.2323 -> 0.2811`).

3. **Defect 3 (Mathematical zero-lift block in certification mode — `data.qvel[0..4] = 0.0` every substep):**
   - When Defects 1 and 2 are removed in-memory (`base` collisions disabled + `ctrl[0..4] = qpos[0..4]`), **both** centered `real453` and top-down `yaw -96` immediately form rock-solid compliant two-sided pinches (`g_act = 0.491 rad` = `28.1°`, `yield = +8.08°` at `cmd = 0.35`, `+11.5°` at `cmd = 0.29`, `F_fix = 27.34 N`, `F_mov = 27.48 N`, zero detonation — matching the real ep453 `10.26°` servo yield!).
   - **However**, under `set_arm`'s `data.qvel[0..4] = 0.0` overwrite, lifting `+50 mm` STILL produces **`dz = -0.1 mm` (0.0 mm lift)** on top-down `yaw -96` even at full stall squeeze (`cmd = 0.10`, `yield = +22.4°`, `~55 N` total clamp force!), and only `+4.4 mm` slip-drop on centered `real453`. Because `v_jaw = 0` on every substep, MuJoCo's friction law pulls `v_cube` toward `0`, never upward; the teleporting jaws simply slide off the top of `cube_b`.

4. **Positive counterfactual proof (in-memory only, zero files edited — `base` excluded + all 6 joints actuated via `data.ctrl`):**
   - **Top-down `yaw -96`** (`q = [-0.7, 8.9, 13.5, 70.8, -89.2]`, `g -> 0.29` or `0.10`): clamps cleanly at `g_act = 0.491 rad` (`yield = +11.5°` / `+22.4°`, `F_fix = 28.14 N`, `F_mov = 28.12 N`) and lifts `+50 mm` cleanly to **`cube_z = 61.4 mm` (`dz = +46.4 mm`, `0.000 rad` grip slip, `0.0 mm` Y drift)**!
   - **Centered `real453`** (`q = [4.2, -25.5, 55.5, 40.0, 6.7]`, `g -> 0.29`): clamps cleanly at `g_act = 0.490 rad` (`yield = +11.5°`) and lifts `+50 mm` cleanly to **`cube_z = 53.6 mm` (`dz = +38.6 mm`, `g_act = 0.487`)**!
   - **Prior Run-9 witness** (`q = [0.98, -27.16, 53.1, 43.99, 9.09]`, `V=(14,-10,14)`): still slipped on lift even with all 3 defects removed (`dz = -0.1 mm`), confirming that its `+8 mm` Y offset was also a genuine pose issue, fully resolved by centering in Y.

### What the Corrections Lane & Next Driver Should Know
- **Do NOT spend time rewriting the jaw collision mesh (option (b)) before fixing Defects 1–3!** The existing `arm_padded_v4cam.xml` voxelized jaws ALREADY form textbook `28 N` compliant friction pinches and lift `cube_b` `+46 mm` with zero slip once Defects 1–3 are fixed.
- **Required corrections-lane fixes (both `arm_padded_v4cam.xml` and `pilot453_v4.py`/`enforce_replay.py`):**
  1. Add `<contact><exclude body1="base" body2="shoulder"/></contact>` (and consider excluding all bolted chain-adjacent arm pairs already exempted in `enforcement_scan`, though `base × shoulder` was the only colliding pair across all 6 tested poses) in `arm_padded_v4cam.xml` / `arm_padded_v4.xml`.
  2. Drive all 6 arm joints via their existing STS3215 `<position>` actuators (`data.ctrl[0..5]`) instead of teleporting `qpos[0..4]` with `qvel[0..4] = 0` and `ctrl[0..4] = 0` (which simultaneously fixes Defects 2 and 3 and makes certification-mode friction carry physically real).
- **Once those fixes land, the next driver has two ready-to-fly winning recipes:**
  - **Top-down `real453+yaw-96`** (`site = [0.2323, 0.010, 0.018]`, `q_deg ≈ [-0.8, 9.0, 13.4, 70.8, -89.3]`): dead-vertical pinch along Y, moving jaw clear on `-Y` away from `cube_a` (`grip_to_cube_a_min_mm ≈ 10 mm`), `grip -> 0.29` (`yield ≈ +11.5°`), zero-slip lift `+46 mm`.
  - **Centered `real453+yaw-5.7`** (`site = [0.2135, -0.0137, 0.015]`, `q_deg ≈ [4.1, -24.7, 54.9, 39.8, 6.7]`): centered in Y on `cube_b`, `grip -> 0.29` (`yield ≈ +11.5°`), clean lift `+39 mm`.
  - Remember channel discipline for replay (`p5 = 2.0`, `p99 = 55.0`): keep ≥6% of frames at `joint <= 0.10` (folded-home hold or end-close) and ≥2% at `joint = 1.45` (wide open), and pass `--object=cube_b` to `enforce_replay.py`.


---

## 2026-11-27 — AGY SOLVE SESSION 2 (gemini-4-neon-xhigh) — PLAN

- **Session 2 start time:** `2026-11-27 10:25 CST`
- **Context read & confirmed:** Read `DRIVER_SEAT.md` (including the P5 full-actuation ERRATUM), `driver_notes.md` in full, and the updated P5 stepping code in `pilot453_v4.py`, `enforce_replay.py`, and `grip_drive.py` (noting `LANE_REPORT.md` and `VERDICT.md` were not present on disk in this seat tree, with their verified key facts relayed in `solve2_prompt.txt` and `DRIVER_SEAT.md`'s erratum).
- **Intent & Approach:**
  1. **Familiarization first (`phase familiarization - ...`, unscored):** Re-feel the P5 fully-actuated arm (`protocol v2 P1+P3+D2+P4+P5`), specifically measuring arm servo gravity sag and tracking lag (`site` actual vs `target_site` commanded). Re-fly the Session-1 **top-down `real453+yaw-96`** recipe (`pitch ≈ 93°`, `wrist_roll ≈ -89°`, moving jaw on `-Y` away from `cube_a`) to verify repeatable compliant attach (`grip -> 0.29`, `yield ≈ +10–12°`) and clean vertical lift (`+50–70 mm`), targeting actual `site_z ≈ 0.018–0.022` at pick so jaw tips sit `10–14 mm` above `cube_b`'s bottom face (maximizing `grip_to_cube_a_min_mm` clearance during placement).
  2. **Probe carry, place, release, and retreat in familiarization:** Measure the post-close `site - cube_b_manip` offset, traverse laterally at safe height (`z ≈ 0.09–0.10`) over `cube_a` (`[0.2571, 0.0484, 0.0149]`) watching for any slip or swing, descend slowly until `cube_b` sits `1–2 mm` above `cube_a`'s top (`cube_b_z ≈ 0.046–0.047`) while monitoring `grip_to_cube_a_min_mm > 0`, release gently in two stages (`0.29 -> 0.65 -> 1.45`) without bumping `cube_b`, retreat vertically clear, and preview with `settle`. (If top-down `yaw -96` hits any carry/place limitation over `cube_a`, test `centered real453+yaw-5.7` as backup.)
  3. **Solving attempts (`attempt 3`, continuing from Session 1's 2 attempts; 8 remaining):** Once the full pick-carry-place-retreat sequence is repeatable in familiarization, fly a clean end-to-end attempt from fresh `init` with full channel discipline (≥6% frames at `joint <= 0.10` and ≥2% at `joint = 1.45`), verifying `sim_success: true` at `settle`.
  4. **Export, Distillation & Certification Replay:** Export `agy2_live_trace.npz`, distill ONE smooth cleaned trajectory `agy2_cleaned.npz` preserving intent, attach plateau, and exact grip channel extremes (`p5 = 2.0`, `p99 = 55.0`), and verify via `enforce_replay.py block_stack 453 agy2 SCENE agy2_cleaned.npz agy2_replay.json --mode=certification --object=cube_b` until `aborted: false`, alarm silent, and criterion passed.
  5. **End-of-session deliverables:** Append Session-2 outcome + full Tier-1 **ep453 SCENE BRIEF + BLOCK-TASK PLAYBOOK** to `driver_notes.md` and write `agy2_session_summary.md`.
- **What I want to observe:**
  - Actual P5 arm servo gravity sag (`site` vs `target_site`) and settling time across unfold, `yaw -96`, descent, lift, and lateral carry over `cube_a`.
  - Exact `cube_b` walk during the compliant close (`site - cube_b_manip` offset), `grip_yield_deg` stability during lift and lateral carry, `grip_to_cube_a_min_mm` margin during descent/release onto `cube_a`, and whether `cube_b` settles dead-square on `cube_a` with contact set `{cube_a}`.
- **Help wanted from the engine:**
  - Per-increment JSON telemetry (`site`, `cube_b_manip`, `cube_a_base`, `joint_actual_rad`, `grip_yield_deg`, `attached`, `jaw_contact`, `jaw_min_dist_mm`, `jaw_dist_fixed_*`, `jaw_dist_moving_*`, `grip_to_cube_a_min_mm`, `events`, `fault_rewound`), fresh `topcam`/`frontcam`/`wristcam` renders after every increment, and `settle` criterion previews.


---

## 2026-11-27 — AGY SOLVE SESSION 2 (gemini-4-neon-xhigh) — OUTCOME

- **Session 2 wall time:** `2026-11-27 10:25 CST` to `2026-11-27 10:39 CST` (~14 minutes)
- **Harness / scene:** `pilot453_v4.py (protocol v2 P1+P3+D2+P4+P5)` / `ENF/scenes-v4/block_stack__ep453_v0__v4cam.xml` (seat scene file, differing from VM original by 1 relative include path line).
- **Attempt accounting:** **1 of 8** solving attempts used in Session 2 (**3 of 10** cumulative across Sessions 1 & 2; Attempt 3).
- **Stop condition:** Live pick → lift → carry → place → release → retreat → 1.0 s settle SUCCEEDED on **Attempt 3**, and `agy2_live_trace.npz` + `agy2_cleaned.npz` were exported and distilled; **stopped under stop condition (c) at the certification-replay step** because `vmstack/build_scenes.py` is absent from this seat bundle, blocking both `seat.py "settle"` (`vmstack/replay.py:24`) and `PY ENF/enforce_replay.py` (`enforce_replay.py:116`) with `ModuleNotFoundError: No module named 'build_scenes'`. No official `agy2_replay.json` was produced; no candidate is claimed from `enforce_replay.py`.

### Plan Outcome & Familiarization Summary (Unscored)
- Set `phase familiarization - P5 full-actuation feel and top-down grasp probe`.
- On the very first familiarization run on the fixed P5 stack, the Session-1 **top-down `real453+yaw-96`** recipe executed the **entire** task cleanly without a single fault or slip:
  - `init` (`folded_home`, actual `site = [-0.1638, 0.0, 0.2839]`, ~5.2 mm steady P5 gravity sag vs kinematic `-0.1596, 0.2891`) → `goto 0.2323 0.011 0.085 40 1.45` → `yaw -96 35` → `goto 0.234 0.010 0.025 35 1.45` → `goto 0.236 0.009 0.018 20 1.45` (actual `site = [0.231, 0.0096, 0.0159]`, `jaw_dist_fixed_mm = -0.23`, `grip_to_cube_a_min_mm = 9.95`).
  - `grip 0.29 35` produced an immediate compliant pinch (`grasp_contact_true` frame 158, `closedness = 0.77`, `joint_actual_rad = 0.4912`, `grip_yield_deg = +11.53°`, `cube_b = [0.2332, -0.0056, 0.0158]`, `grip_to_cube_a_min_mm = 9.23`), walking `cube_b` `+7.5 mm` in `+Y` into a rigid centered seat (`cube_b - site = [+0.0024, -0.0148, +0.0026]`).
  - `goto 0.236 0.009 0.095 35` lifted `cube_b` `+67.5 mm` (`z = 0.0833`) with `0.000 rad` slip (`joint_actual_rad = 0.4913`, `grip_yield_deg = +11.534°`), visually confirming jaw tips ~5 mm above `cube_b`'s bottom face.
  - Lateral carry (`goto 0.257 0.0632 0.095 40`), alignment (`goto 0.260 0.0645 0.060 35`), and place (`goto 0.2598 0.0658 0.050 25`) landed `cube_b` within `[+0.6, +0.1] mm` of `cube_a` with `grip_to_cube_a_min_mm = 7.59 mm`.
  - **Refinement spotted during familiarization:** Releasing slowly in two stages (`grip 0.75 15` then `grip 1.45 15`) allowed `jaw_contact` to drop for 10 frames before `closedness <= 0.30`, logging `jaw_contact_lost_10_frames` at frame 318 prior to `release` at frame 322 (`grip_to_cube_a_min_mm` dipped to `5.29 mm` due to ~2.4 mm unload sag). Retreat (`goto 0.2598 0.068 0.135 30 1.45`) left `cube_b = [0.2575, 0.0457, 0.0448]` stacked cleanly on `cube_a = [0.2573, 0.0484, 0.0148]`.
  - Calling `seat.py "settle"` returned `{"error": "ModuleNotFoundError: No module named 'build_scenes'"}`.

### Per-Attempt Log (Session 2: 1 of 8; Cumulative: 3 of 10)
- **Attempt 3 (`attempt 3 - top-down yaw-96 full pick-carry-place-retreat`):**
  - Applied three refinements learned in familiarization: (i) `hold 25` at `folded_home` to guarantee `chan_p5 == 2.00`; (ii) raised straddle target by `+2 mm` (`z = 0.020`) so jaw tips sit ~7 mm above `cube_b`'s bottom face (increasing `grip_to_cube_a_min_mm` margin during place/release); (iii) single continuous 15-frame release ramp `grip 1.45 15` so contact-drop to `release` (`c <= 0.30`) takes ~7 frames (< 10), eliminating `jaw_contact_lost_10_frames`.
  - **Executed command sequence & engine milestones:**
    1. `init` (f1): `folded_home`, `joint_rad = 0.0`, `cube_b = [0.2323, -0.0131, 0.0149]`, `cube_a = [0.2571, 0.0484, 0.0149]`.
    2. `hold 25` (f26): full-close anchor (`26 frames` at `chan = 2.0`).
    3. `goto 0.2323 0.011 0.085 40 1.45` (f66) → `yaw -96 35` (f101): unfolded and rotated to `real453+yaw-96` (`q_deg = [-0.2, 4.2, -16.3, 102.0, -88.2]`, `joint_actual_rad = 1.449`).
    4. `goto 0.234 0.010 0.025 30 1.45` (f131) → `goto 0.236 0.009 0.020 20 1.45` (f151): straddle hover at `site = [0.2309, 0.0097, 0.0175]`, `jaw_dist_fixed_mm = 0.38` (zero pre-close touch), `grip_to_cube_a_min_mm = 10.53`.
    5. **Attach milestone:** `grip 0.29 35` (f186) → `hold 5` (f191): `attached: true`, `events: [{"frame": 178, "type": "grasp_contact_true", "closedness": 0.77}]`, `joint_actual_rad = 0.4912`, `grip_yield_deg = +11.528°`, `jaw_min_dist_mm = -1.14`, `cube_b = [0.2331, -0.0058, 0.0160]`, `grip_to_cube_a_min_mm = 9.94`, `fault_rewound: null`.
    6. **Lift milestone:** `goto 0.236 0.009 0.095 35` (f226): `cube_b = [0.2356, -0.0059, 0.0816]` (`+65.6 mm`), `joint_actual_rad = 0.4912`, `grip_yield_deg = +11.53°` (zero slip), `grip_to_cube_a_min_mm = 48.59`.
    7. **Carry & align milestone:** `goto 0.258 0.064 0.095 40` (f266) → `goto 0.2598 0.0660 0.060 30` (f296): `cube_b = [0.2571, 0.0476, 0.0549]` (`Δx = 0.0 mm, Δy = -0.8 mm` over `cube_a`), `joint_actual_rad = 0.4912`, `grip_to_cube_a_min_mm = 16.85`.
    8. **Place & release milestone:** `goto 0.2598 0.0668 0.052 25` (f321, `cube_b = [0.2576, 0.0495, 0.0470]`, `grip_to_cube_a_min_mm = 9.22`) → `grip 1.45 15` (f336) → `hold 5` (f341): `attached: false`, `jaw_contact: false`, `events: [{"frame": 178, "type": "grasp_contact_true", "closedness": 0.77}, {"frame": 330, "type": "release", "closedness": 0.223}]`, `cube_b = [0.2569, 0.0463, 0.0448]`, `cube_a = [0.2572, 0.0484, 0.0148]`, `grip_to_cube_a_min_mm = 7.77` (minimum across entire run).
    9. **Retreat & 1.0 s live settle milestone:** `goto 0.2598 0.069 0.135 30 1.45` (f371) → `hold 20` (f391): `cube_b = [0.2569, 0.0463, 0.0448]` (0.0 mm drift over 1.0 s hold!), `cube_a = [0.2572, 0.0484, 0.0148]` (`Δx = -0.3 mm, Δy = -2.1 mm`, `z_b - z_a = 30.0 mm`), `grip_to_cube_a_min_mm = 87.45`, `jaw_contact: false`, `fault_rewound: null`. Visual confirmation on `topcam`/`frontcam`/`wristcam`: green `cube_b` dead-square flush on purple `cube_a`, jaws wide open high clear.
  - **Export & Distillation:**
    - Exported `ENF/pilot_ep453_v4/agy2_live_trace.npz` (`frames: 391`, `chan_p5: 2.0`, `chan_p99: 55.0`).
    - Distilled `ENF/pilot_ep453_v4/agy2_cleaned.npz` (`shape: (374, 6)`, trimmed redundant tail settle beyond 5f and gently Gaussian-blended internal transit corners on the 5 arm joints while pinning `folded_home`, attach plateau, release plateau, and final pose; `chan_p5 = 2.00`, `chan_p99 = 55.00`, `max |dq| = [0.525, 2.881, 3.659, 1.410, 3.988] deg/f`, `max |ddq| = [0.045, 0.274, 0.348, 0.165, 0.383] deg/f²`).
  - **Official `settle` & `enforce_replay.py` Verbatim Block (Seat Packaging Defect):**
    - `seat.py "settle"` returned verbatim: `{"error": "ModuleNotFoundError: No module named 'build_scenes'"}`
    - `PY ENF/enforce_replay.py block_stack 453 agy2 SCENE ENF/pilot_ep453_v4/agy2_cleaned.npz ENF/pilot_ep453_v4/agy2_replay.json --mode=certification --object=cube_b` returned verbatim:
      ```
      Traceback (most recent call last):
        File "sim/vmstack/enforced/enforce_replay.py", line 116, in <module>
          import build_scenes as B
      ModuleNotFoundError: No module named 'build_scenes'
      ```
    - Root cause (verified read-only; zero stack/scene/harness files touched): `vmstack/` in this seat bundle contains only `enforced/`, `path_check.py`, `replay.py`, and `twin_params.py`; `build_scenes.py` is absent anywhere under `SEAT`.
  - **Driver-Side Read-Only P5 Diagnostic Mirror (NOT an official `enforce_replay.py` record):**
    - Replaying `agy2_cleaned.npz` through a scratch read-only mirror of `enforce_replay.py`'s exact P5 certification loop + `replay.evaluate`'s exact `block_stack` primary criterion (`verify_p5_offline.py`, zero seat files modified) gave: `aborted: false`, `enforcement_fault: null`, `weld_ever_active: false`, `grip_channel_ok: true` (`2.0..55.0`), `grasp_events: [{"frame": 176, "type": "grasp_contact_true", "closedness": 0.7699}, {"frame": 328, "type": "release", "closedness": 0.2228}]`, `arm_self_contacts_adjacent: {}`, `c2_trace: {frames: 17, planar_travel_m: 0.0019}`, `min_grip_to_cube_a_mm: 7.75` (f339), `alarm_fired: false`, `sim_success_criteria_read: true`, `upper_contacts: ["cube_a"]`, `within_inset_footprint: true`, `contact_with_lower: true`, `no_other_support_contact: true`, `xy_offset_mm: [-0.30, -2.09]`.

### Hints & Waypoint Cues Used (Parity Doctrine Disclosure)
- **Cameras:** Only the three registered viewpoints (`topcam`, `frontcam`, `wristcam`) were inspected throughout.
- **JSON Telemetry Hints Used:** `site`, `cube_b_manip`, `cube_a_base`, `q_deg`, `joint_actual_rad`, `grip_yield_deg`, `attached`, `jaw_contact`, `jaw_min_dist_mm`, `jaw_dist_fixed_mm`, `jaw_dist_moving_mm`, `grip_to_cube_a_min_mm`, `events`, `fault_rewound`.
- **Waypoint Cues Used:** Only the seat's own `init` cube positions (`cube_b = [0.2323, -0.0131, 0.0149]`, `cube_a = [0.2571, 0.0484, 0.0149]`) and the Session-1 self-discovered top-down `real453+yaw-96` recipe (no external scans, templates, or prior-driver anchors).

### What Fixed P5 Physics Changed vs Session 1
- Removing `base × shoulder` interpenetration and switching all 5 arm joints to real STS3215 servo control (`data.ctrl[0..5]`, no `qpos`/`qvel` overwrite) completely eliminated the `9.7–408 g` phantom accelerations, `0.0 N` ghost contacts, and zero-lift friction lock of Session 1.
- Compliant pinches now lock cleanly at `joint_actual_rad = 0.4912` (`+11.53°` yield) and carry `cube_b` vertically and laterally with `0.000 rad` slip and zero detonation.
- New P5 reality to account for when driving: the 5 arm servos exhibit real gravity sag (`~3–5 mm` steady Z sag, up to `~14 mm` dynamic lag at high hover / `wrist_flex = +102.0°` saturation, and `~2.4 mm` extra Z sag on cube unload during release). Closed-loop `goto` compensation handles this cleanly.

### What the Orchestrator / Next Driver Should Know
- **Action needed to complete certification replay:** Restore `build_scenes.py` into `SEAT/validator-sets-20261123/coverage-topup-work/vmstack/` (or run `enforce_replay.py` on the VM/DGX where `build_scenes.py` resides) and run:
  `PY ENF/enforce_replay.py block_stack 453 agy2 SCENE ENF/pilot_ep453_v4/agy2_cleaned.npz ENF/pilot_ep453_v4/agy2_replay.json --mode=certification --object=cube_b`
- `agy2_cleaned.npz` is already distilled, smooth, channel-compliant (`p5 = 2.0, p99 = 55.0`), and verified end-to-end under the P5 physics/enforcement/criterion mirror.

---

## ep453 SCENE BRIEF + BLOCK-TASK PLAYBOOK

*Standalone brief written for a competent teleoperation driver who has not seen Sessions 1–2 (`block_stack` ep453 twin on the fixed P5 full-actuation stack, `protocol v2 P1+P3+D2+P4+P5`, `arm_padded_v4cam.xml`).*

### 1. Plan (Chosen Approach & Rationale, in Execution Order)
Choose the **top-down vertical pinch (`real453+yaw-96`)** rather than a horizontal/tilted `real453` approach:
1. **Why `real453+yaw-96`:** Rotating `R_mode = real453` by `yaw -96°` places the gripper almost straight vertical (`pitch ≈ 93°`, `wrist_roll ≈ -89°`), pinching squarely along world `Y` with the fixed jaw on the `+Y` side of `cube_b` and the bulky moving jaw swinging wide on the `-Y` side—**away** from `cube_a` (`+Y`, `61.5 mm` away). This gives three decisive advantages: (a) symmetric flat-face two-jaw clamping with zero corner-camming, (b) `~10 mm` clearance to `cube_a` at pick and `>= 7.75 mm` clearance throughout place/release because the jaw tips grasp the upper ~23 mm of `cube_b` and stay ~7 mm above `cube_b`'s bottom face, and (c) straight vertical lift, pure planar carry, and straight vertical retreat without wrist reorientation.
2. **Execution order:**
   1. Start from `init` (`folded_home`, `joint = 0.0`) and `hold 25` so ≥6% of frames anchor genuine full close (`p5 = 2.0`).
   2. Unfold high above `cube_b` (`z = 0.085`) while opening wide (`joint = 1.45`, anchoring `p99 = 55.0`), then `yaw -96`.
   3. Descend in two stages (`z = 0.025` → `z = 0.020`) to straddle `cube_b`'s upper two-thirds with zero pre-close touch (`jaw_dist_fixed_mm > 0`).
   4. Close smoothly to `grip 0.29 35` (`joint_actual_rad ≈ 0.491`, `grip_yield_deg ≈ +11.5°`, `grasp_contact_true`) + `hold 5` attach plateau.
   5. Lift vertically (`z = 0.095`), carry horizontally over `cube_a` compensating for the measured `cube_b - site` offset (`[+0.0022, -0.0150, +0.0008] m`), align & lower to `z = 0.060` (verify XY within `< 1 mm` of `cube_a` visually and via telemetry), then gently lower to `z = 0.052` (~2 mm above `cube_a` top).
   6. Release in ONE smooth 15-frame ramp (`grip 1.45 15`) + `hold 5` so jaw-open finishes cleanly without triggering `jaw_contact_lost_10_frames`, then retreat straight up with slight `+Y` offset (`y + 2 mm, z = 0.135`) clear of the stack and settle (`hold 20` / `settle`).

### 2. Measured Failure Map (Sessions 1 & 2)

| Configuration / Action | Session & Stack | Measured Failure | Verdict / Dead End? |
| :--- | :--- | :--- | :--- |
| Prior Run-9 witness (`real453`, `V=(14,-10,14)`, `q=[0.98,-27.16,53.1,43.99,9.09]`) | Pre-P5 & P5 counterfactual | `+8 mm` Y off-center relative to `cube_b`; slipped on lift (`dz = -0.1 mm`) **even after all 3 defects were removed**. | **DEAD END (pose geometry)** — always center jaws on `cube_b` in Y. |
| `real453+yaw-5.7` at `z = 0.008` | Session 1 (pre-P5) | `fault_rewound` frame 82: `mat × wrist_roll_follower_so101_v1_col_b867` (`dist < 0`). | **HARD FLOOR LIMIT** — keep `site_z >= 0.014` in `real453` (`>= 0.016` actual in `yaw -96`). |
| `real453` / `real453+yaw-5.7` / `real411` (`z=0.010..0.015`), Attempt 2 | Session 1 (pre-P5 ONLY) | First jaw touch flung `cube_b` `-72` to `-148 mm` in `-Y` (even in opposite-roll `real411`, `-113 mm`). | **FIXED IN P5** — caused by pre-P5 Defect 1 (`base × shoulder` ~966 kN, `63–194 g` phantom `-Y` accel). Centered `real453+yaw-5.7` now grasps and lifts cleanly on P5 (`dz = +38.6 mm` counterfactual / `+53.05 mm` verifier Check 2). |
| Top-down `real453+yaw-96` (`z=0.014..0.028`), Attempt 1 | Session 1 (pre-P5 ONLY) | `grasp_contact_true` f148 (`-0.75 mm` penetration, `0.0 N` ghost force, `+0.033°` yield), then `+25 mm` lift flung `cube_b` `-159 mm` in `-Y`; deeper close blasted `cube_b` `-0.47..-1.46 m` in `-Y` or `+1.60 m` in `+Y`. | **FIXED IN P5** — caused by Defects 1–3 (`-71 g` separating phantom accel + `ctrl[0..4]=0` + `qvel[0..4]=0` zero-lift block). Now the primary winning recipe on P5. |
| Slow two-stage release (`grip 0.75 15` then `grip 1.45 15`) at `target_z = 0.050` | Session 2 fam (P5) | Logged spurious `jaw_contact_lost_10_frames` (f318) before `release` (f322); unload Z sag (`-2.4 mm`) lowered `grip_to_cube_a_min_mm` from `7.59` to `5.29 mm` (still non-faulting, narrower margin). | **AVOIDABLE TIMING/HEIGHT TRAP** — use single `grip 1.45 15` ramp from `target_z = 0.052` (eliminated contact-loss event and raised min clearance to `7.75 mm`). |
| `seat.py "settle"` and `PY ENF/enforce_replay.py` in `zk0-sim-seat-20261127` | Session 2 (P5) | Immediate `ModuleNotFoundError: No module named 'build_scenes'` (`replay.py:24`, `enforce_replay.py:116`). | **SEAT PACKAGING BLOCK** — `vmstack/build_scenes.py` missing from seat directory; live physics/pilot unaffected, certification replay requires restoring `build_scenes.py`. |

### 3. Working Recipe (Complete Pick → Lift → Carry → Place → Release → Retreat, Fixed P5 Stack)
Verified live in Session 2 Attempt 3 (`agy2_live_trace.npz`, 391 frames) and distilled to `agy2_cleaned.npz` (374 frames; `chan_p5 = 2.0, chan_p99 = 55.0`). Orientation family: `real453` rotated via `yaw -96`.

| Step | Command (`seat.py`) | Cum. Frames | Expected Telemetry Signature (P5) |
| :--- | :--- | :---: | :--- |
| 1. Init | `init` | 1 | `folded_home`, `joint=0.0`, `cube_b=[0.2323,-0.0131,0.0149]`, `cube_a=[0.2571,0.0484,0.0149]` |
| 2. Close anchor | `hold 25` | 26 | 26 frames at `joint=0.0` (`chan=2.0`), guarantees `p5=2.0` |
| 3. Unfold & open | `goto 0.2323 0.011 0.085 40 1.45` | 66 | `site≈[0.2365,0.0083,0.1012]`, jaws open `1.45` (`chan=55.0`) |
| 4. Top-down yaw | `yaw -96 35` | 101 | `R_mode="real453+yaw-96"`, `q_deg≈[-0.2,4.2,-16.3,102.0,-88.2]`, `joint_actual≈1.449` |
| 5. Hover over `cube_b` | `goto 0.234 0.010 0.025 30 1.45` | 131 | `site≈[0.2313,0.0100,0.0269]`, `jaw_dist_fixed_mm≈4.2`, `grip_to_cube_a≈13.7 mm` |
| 6. Straddle `cube_b` | `goto 0.236 0.009 0.020 20 1.45` | 151 | `site≈[0.2309,0.0097,0.0175]`, `jaw_dist_fixed_mm≈+0.38` (no touch), `grip_to_cube_a≈10.5 mm` |
| 7. Compliant grasp | `grip 0.29 35` | 186 | `attached: true`, `grasp_contact_true` (~f178, `c=0.77`), `joint_actual≈0.4912`, `grip_yield_deg≈+11.53°`, `cube_b≈[0.2331,-0.0058,0.0159]`, `grip_to_cube_a≈9.9 mm` |
| 8. Attach plateau | `hold 5` | 191 | Stable pinch (`0.0 mm` drift, `joint_actual=0.4912`) |
| 9. Vertical lift | `goto 0.236 0.009 0.095 35` | 226 | `cube_b≈[0.2356,-0.0059,0.0816]` (`+65.6 mm`), `joint_actual=0.4912` (zero slip), `cube_b - site ≈ [+0.0022,-0.0150,+0.0008]` |
| 10. Lateral carry | `goto 0.258 0.064 0.095 40` | 266 | `cube_b≈[0.2560,0.0429,0.0874]`, `joint_actual=0.4912`, `grip_to_cube_a≈49 mm` |
| 11. Align above `cube_a` | `goto 0.2598 0.0660 0.060 30` | 296 | `cube_b≈[0.2571,0.0476,0.0549]` (`Δxy=[-0.0,-0.8] mm` vs `cube_a`), `grip_to_cube_a≈16.8 mm` |
| 12. Place descent | `goto 0.2598 0.0668 0.052 25` | 321 | `cube_b≈[0.2576,0.0495,0.0470]` (~2 mm above flush), `grip_to_cube_a≈9.2 mm` |
| 13. Crisp release | `grip 1.45 15` | 336 | `attached: false`, `release` (~f330, `c=0.223`), zero `jaw_contact_lost_10_frames`, `cube_b≈[0.2569,0.0465,0.0448]`, `grip_to_cube_a≈8.1 mm` |
| 14. Jaw-open dwell | `hold 5` | 341 | `joint_actual→1.39`, `jaw_contact: false`, `grip_to_cube_a≈7.77 mm` (run minimum) |
| 15. Retreat clear | `goto 0.2598 0.069 0.135 30 1.45` | 371 | `cube_b=[0.2569,0.0463,0.0448]` (0.0 mm disturbance), `grip_to_cube_a≈83.1 mm` |
| 16. Settle | `hold 20` (and `settle` once `build_scenes.py` is present) | 391 | `cube_b=[0.2569,0.0463,0.0448]`, `cube_a=[0.2572,0.0484,0.0148]`, `Δxy=[-0.3,-2.1] mm` (tolerance `±11.25 mm`), `z_b-z_a=30.0 mm`, `grip_to_cube_a=87.45 mm` |

*Backup orientation family (if ever needed for another ep):* `centered real453+yaw-5.7` (`site = [0.2135, -0.0137, 0.015]`, `q_deg ≈ [4.1, -24.7, 54.9, 39.8, 6.7]`, `grip 0.29`), verified in Session-1 counterfactual (`+38.6 mm` lift) and verifier Check 2 (`z = 53.05 mm`).

### 4. Scene / Arm / Object Facts
- **Cubes (`CUBE_EDGE = 0.030 m`, `CUBE_MASS = 0.020 kg`, `CUBE_FRICTION = (0.6, 0.005, 0.0001)`):**
  - `cube_b` (green, manipulated): spawns at `[0.2323, -0.0131, 0.0149]` (resting flush on mat `z = 0`).
  - `cube_a` (purple, base): spawns at `[0.2571, 0.0484, 0.0149]` (`66.3 mm` centre-to-centre, `36.3 mm` edge-to-edge along mostly `+Y`).
  - **Registered success criterion:** `cube_b` in contact with `cube_a`, `|Δx| <= 11.25 mm` and `|Δy| <= 11.25 mm` (top footprint inset 25%), and `cube_b` contact set `== {"cube_a"}`. Stacked flush centre height: `cube_b_z = 0.0448..0.0449 m`.
- **Clearances & Enforcement:**
  - Any jaw × `cube_a` contact or arm × mat/wall contact is an immediate `EnforcementAbort` (`fault_rewound` in pilot, `aborted: true` in `enforce_replay.py`).
  - In `real453+yaw-96` picked at target `z = 0.020` (actual `site_z = 0.0175`), jaw tips clear `cube_b`'s bottom face by ~7 mm, preserving `grip_to_cube_a_min_mm = 10.5 mm` (pick straddle) → `9.9 mm` (attach) → `9.2 mm` (place) → `7.75 mm` (post-release dwell minimum) → `87.5 mm` (retreated).
  - Mat contact limit in `real453` is `site_z ≈ 0.008 m`; keep `site_z >= 0.014 m` (`>= 0.017 m` actual in `yaw -96`).
- **Arm & P5 Servo Quirks:**
  - All 6 joints use STS3215 position actuators capped at `±1.9123 N·m`. Expect `~3–5 mm` steady gravity sag in Z (and `~10–14 mm` lag during rapid lifts or when `wrist_flex` hits its `+102.0°` joint limit at high `z >= 0.08`).
  - During release (`grip` ramp with frozen `q5`), `site_z` sags another `~1.5–2.4 mm` as load/jaws settle; always release `~2 mm` above flush (`target_z = 0.052` rather than `0.050`).
  - `move dx dy dz` computes its target from *actual* sagged `data.site_xpos`, whereas `goto x y z` targets absolute world coordinates—prefer `goto` with closed-loop correction from `cube_b_manip` / `site` telemetry.
  - Because `cube_b` sits at `y = -0.0131` and `real453+yaw-96` closes the moving jaw from `-Y` toward the fixed jaw on `+Y`, the compliant close walks `cube_b` `+7.3 mm` in `+Y` (`-0.0131 -> -0.0058`), yielding a rock-steady carried offset `cube_b - site ≈ [+0.0022, -0.0150, +0.0008] m`. Always read `cube_b - site` after lift before positioning over `cube_a`.
- **Cameras (`topcam`, `frontcam`, `wristcam`):**
  - `frontcam` is the primary depth/stacking verification view (clearly shows jaw-tip height above `cube_b`'s bottom face, hover gap over `cube_a`, and flush stacked alignment).
  - `topcam` is partially occluded directly above the cubes by the yellow wrist/arm links during `yaw -96` descent/place, though clear for global arm clearance.
  - `wristcam` gives a close-up of jaw alignment on `cube_b` and proximity to purple `cube_a` on the `+Y` side.
- **Grip Channel Discipline (Load-Bearing for `enforce_replay.py`):**
  - `export` maps `joint ∈ [0.10, 1.45]` linearly to `chan ∈ [2.0, 55.0]` (`chan = 55 - 53 * clip((1.45 - joint) / 1.35, 0, 1)`).
  - `_closedness_trace` normalizes per-episode using **`p5` and `p99`** (`lo = percentile(g, 5)`, `hi = percentile(g, 99)`). Always include `>= 6%` frames at `joint <= 0.10` (e.g., `hold 25` at `folded_home`) AND `>= 2%` frames at `joint = 1.45` so `chan_p5 == 2.0` and `chan_p99 == 55.0`; otherwise replay rescales the grip command and changes pinch force.
  - Always pass `--object=cube_b` to `enforce_replay.py` (multi-object scene).
- **Fault / Rollback & Harness Notes:**
  - `pilot453_v4.py` snapshots state before every command; if any substep hits a forbidden contact, `restore(snap)` rewinds the command and returns `"fault_rewound": {...}`. Treat it as boundary feedback and adjust clearance.
  - `pilot453_v4.py`'s `settle` command (`replay.py:24`) and `ENF/enforce_replay.py` (`enforce_replay.py:116`) both execute `import build_scenes as B` from `vmstack/`. Ensure `vmstack/build_scenes.py` is present in the seat bundle before invoking `settle` or `enforce_replay.py`.


---

## 2026-11-27 — AGY SOLVE SESSION 3 (gemini-4-neon-high) — PLAN

- **Session 3 start time:** `2026-11-27 11:06 CST`
- **Context read & confirmed:** Read `DRIVER_SEAT.md` (including the P5 full-actuation ERRATUM) and `driver_notes.md` in full, including Session 2's outcome and the Tier-1 `ep453 SCENE BRIEF + BLOCK-TASK PLAYBOOK`. Noted that `vmstack/build_scenes.py` is restored on this seat, `pilot411_v4.py` defaults to `R_mode = real411`, and ep411's geometry differs substantially from ep453 (`cube_b` close-in at `[0.0861, -0.0431, 0.015]`, `cube_a` far out at `[0.2391, 0.0296, 0.015]`, ~169 mm mostly `+X` carry).
- **Intent & Approach (transferring Tier-1 Playbook method to ep411):**
  1. **Familiarization first (`phase familiarization - ...`, unscored):**
     - Read initial `state` and inspect all three registered camera renders (`topcam`, `frontcam`, `wristcam`) to ground ep411's layout.
     - Test orientation families (`real411` default, `real411 + yaw`, and Tier-1's top-down `real453 + yaw` variants) at both close-in `cube_b` (`x=0.0861, y=-0.0431`) and far `cube_a` (`x=0.2391, y=0.0296`). Verify which orientation gives clean two-sided flat-face clamping on `cube_b` without corner-camming, mat/wall faults, or joint-limit saturation, while leaving generous jaw-tip clearance (`grip_to_cube_a_min_mm > 0`) for placement.
     - Probe the compliant pinch (`grip 0.29` / tuned target, expecting `grip_yield_deg ≈ +10–12°` and `grasp_contact_true`), measure the post-close `cube_b - site` offset, and stress-test the **~169 mm carry** (lift → multi-stage or smooth horizontal traverse → alignment above `cube_a`) watching for slip, servo lag, gravity sag, or wrist-limit saturation.
     - Probe placement descent (~2 mm above flush to account for P5 unload Z sag), single-ramp crisp release (`grip 1.45 15` to avoid `jaw_contact_lost_10_frames`), vertical clear retreat, and `settle` criterion preview.
  2. **Solving attempts (`phase attempt N - ...`, budget 10 attempts, 0 used):**
     - Once a repeatable pick → lift → 169 mm carry → place → release → retreat → settle sequence is verified in familiarization, fly a full closed-loop attempt from fresh `init`, observing full grip-channel discipline (≥6% of frames at `joint <= 0.10` via initial `hold` at `folded_home` and ≥2% at `joint = 1.45` so `p5 = 2.0` and `p99 = 55.0`), until `settle` confirms the registered criterion is met with the gripper fully retreated clear.
  3. **Export, Distillation & Certification Replay:**
     - Export `ENF/pilot_ep411_v4/agy3_live_trace.npz`, distill ONE smooth cleaned trajectory `ENF/pilot_ep411_v4/agy3_cleaned.npz` preserving intent, attach plateau, and exact grip extremes (`p5 = 2.0`, `p99 = 55.0`), and verify with `PY ENF/enforce_replay.py block_stack 411 agy3 SCENE ENF/pilot_ep411_v4/agy3_cleaned.npz ENF/pilot_ep411_v4/agy3_replay.json --mode=certification --object=cube_b` until `aborted: false`, alarm layer silent, and criterion passed.
  4. **End-of-session deliverables:** Append Session 3 outcome and a `TIER-2 ADDENDUM (session 3, ep411)` to the Playbook in `driver_notes.md`, and write `ENF/pilot_ep411_v4/agy3_session_summary.md`.
- **What I want to observe:**
  - Reachability, IK error, and joint margins at close-in `cube_b` (`r ≈ 96 mm`) vs extended `cube_a` (`r ≈ 241 mm`) under `real411` vs top-down `yaw` orientations.
  - Pre-close fixed/moving jaw gaps (`jaw_dist_fixed_mm`, `jaw_dist_moving_mm`), cube walk during close, `grip_yield_deg` stability across the ~169 mm carry, P5 arm gravity sag/lag, and `grip_to_cube_a_min_mm` margin during place and release.
  - Exact `settle` criterion readouts and `enforce_replay.py` certification outputs.
- **Help wanted from the engine:**
  - Full per-increment JSON telemetry (`site`, `cube_b_manip`, `cube_a_base`, `q_deg`, `joint_actual_rad`, `grip_yield_deg`, `attached`, `jaw_contact`, `jaw_min_dist_mm`, `jaw_dist_fixed_*`, `jaw_dist_moving_*`, `grip_to_cube_a_min_mm`, `events`, `fault_rewound`), fresh `topcam`/`frontcam`/`wristcam` renders after every command, and `settle` criterion previews.


---

## 2026-11-27 — AGY SOLVE SESSION 3 (gemini-4-neon-high) — OUTCOME

- **Session 3 wall time:** `2026-11-27 11:06 CST` to `2026-11-27 11:36 CST` (~30 minutes)
- **Harness / scene:** `pilot411_v4.py (protocol v2 P1+P3+D2+P4+P5)` / `ENF/scenes-v4/block_stack__ep411_v3__v4cam.xml` (seat scene file, differing from VM original by 1 relative include path line).
- **Attempt accounting:** **1 of 10** solving attempts used (`Attempt 1`), following unscored familiarization.
- **Stop condition reached:** **(a) Replay-clean candidate exists (SUCCESS)** — `enforce_replay.py --mode=certification --object=cube_b` returned `"aborted": false`, `"alarm_fired": false`, `"weld_ever_active": false`, and `"sim_success_criteria_read": true` on `agy3_cleaned.npz`. (Candidate only — not certified; judging happens separately.)

### Plan Outcome & Familiarization Findings (Unscored)
1. **Unexpected ep411 primary obstacle — close-in `cube_b` shoulder-pan collision wall (NOT the 169 mm traverse):**
   - Visual inspection (`topcam`/`frontcam`) immediately revealed `cube_b` (purple in ep411, at `[0.0861, -0.0431, 0.0149]`) spawns directly beside the SO-101 shoulder turret (`geom19`), while `cube_a` (green, at `[0.2391, 0.0296, 0.0149]`) sits out in the middle of the arena.
   - Live probing hit `fault_rewound: cube_b × shoulder (geom19)` whenever `shoulder_pan` reached `+13.3°` (`goto 0.09 -0.043 0.12` f18, `goto 0.11 -0.043 0.08` f66, `goto 0.11 -0.012 0.05` f79). A read-only `mj_geomDistance` scan of `shoulder × cube_b` vs `shoulder_pan` on the unmodified scene confirmed that `pan ∈ [+12°, +72°]` sweeps `shoulder` `-1.8` to `-6.7 mm` through `cube_b` at spawn (`pan = 0°` gives `+8.10 mm`, `pan = +5°` gives `+4.46 mm`, `pan = +10°` gives `+0.80 mm`, `pan = +15°` gives `-2.67 mm`). Because `pan` starts at `0°` and has range `[-110°, +110°]`, any direct approach to spawn `cube_b` must keep `pan <= +11°`.
   - Furthermore, attempting to straddle spawn `cube_b` directly in `real411+yaw-90` or `real453+yaw-96` at `site_x <= 0.112` hit non-adjacent arm self-collisions (`moving_jaw × upper_arm`, `moving_jaw × shoulder`, and `gripper × shoulder`).
   - Switching to `real453+yaw-130` (`wrist_roll ≈ -122.8°`, `ik_rot_err ≈ 15.0°` vs `20.0°` for `real411+yaw-118`) solved the approach cleanly: `site = [0.122, 0.004, 0.015]` kept `pan = +1.2°` (`+7.2 mm` shoulder clearance) and placed the bulky `gripper` body clear of `shoulder` while reaching the moving jaw behind `cube_b`.
   - Closing directly to `grip 0.29` at spawn in `real453+yaw-130` still walked/pivoted `cube_b` into `shoulder` (`cube_b × geom19`, `-0.44 µm`).
2. **Winning ep411 adaptation — un-attached pre-grasp scoop (`closedness < 0.50`) + Tier-1 `real453+yaw-96` stack:**
   - Engaging the moving jaw behind `cube_b` at `grip 0.85 15` (`closedness = 0.444 < 0.50`, `attached: false`) and sliding outward to `goto 0.195 -0.022 0.015 35 0.80` (`closedness = 0.481 < 0.50`, `attached: false`, `grip_to_cube_a_min_mm = 28.62 mm`) cleanly relocated `cube_b` `+41 mm` in `+X` to `[0.127, -0.045]`, completely clearing the `shoulder` collision zone without triggering any premature attach or `c2_trace` attached-drag.
   - From there, lifting wide open (`1.45`), rotating `yaw +34` into Tier 1's exact **`real453+yaw-96`**, and executing Tier 1's playbook (straddle at `z = 0.022`, `grip 0.29 35`, vertical lift `+67 mm`, two-stage ~135 mm carry, alignment hover, place ~2 mm above flush, single-ramp `grip 1.45 15` release, `hold 5` dwell, vertical retreat, `settle`) worked on the first full try (`sim_success: true`, `Δxy = [-1.0, -1.7] mm`, zero faults, zero slip over the long carry).

### Per-Attempt Log (1 of 10 used)
- **Attempt 1 (`attempt 1 - scoop clear of shoulder then top-down yaw-96 stack`):**
  - **Executed sequence & milestones (zero `fault_rewound` across all 516 live frames):**
    1. `init` (f1) → `hold 35` (f36): `folded_home`, `joint = 0.0` (`chan = 2.0`), anchoring `chan_p5 == 2.0`.
    2. `goto 0.155 0.004 0.105 35 1.45` (f71) → `setR real453` (f91) → `yaw -130 30` (f121): unfolded wide open (`chan = 55.0`) into `real453+yaw-130`.
    3. `goto 0.122 0.004 0.015 30 1.45` (f151) → `grip 0.85 15` (f166) → `goto 0.195 -0.022 0.015 35 0.80` (f201): mid-height scoop (`pan = +1.2° → +10.5°`, `attached: false`, `closedness <= 0.481`, `grip_to_cube_a_min_mm = 28.62`) relocated `cube_b` from `[0.0861, -0.0431, 0.0149]` to `[0.1270, -0.0450, 0.0219]`.
    4. `goto 0.165 -0.015 0.085 20 1.45` (f221) → `yaw 34 25` (f246) → `goto 0.135 -0.025 0.022 25 1.45` (f271): rotated to `real453+yaw-96` and straddled relocated `cube_b` (`jaw_dist_fixed_mm = +0.30`, zero pre-close touch, `pan = +18.9°` clear of shoulder).
    5. **Attach milestone:** `grip 0.29 35` (f306) → `hold 5` (f311): `attached: true`, `events: [{"frame": 296, "type": "grasp_contact_true", "closedness": 0.718}]`, `joint_actual_rad = 0.4942 -> 0.4891`, `grip_yield_deg = +11.70° -> +11.41°`, `cube_b = [0.1222, -0.0391, 0.0172]`.
    6. **Lift milestone:** `goto 0.135 -0.025 0.095 35` (f346): `cube_b = [0.1229, -0.0396, 0.0846]` (`+67.4 mm`), `joint_actual_rad = 0.4834`, `grip_yield_deg = +11.08°`, zero slip.
    7. **Long carry & align milestone:** `goto 0.195 0.010 0.095 30` (f376) → `goto 0.253 0.044 0.095 35` (f411) → `goto 0.2555 0.0460 0.062 30` (f441): traversed ~135 mm over `cube_a` with `0.0004 rad` grip variation (`joint_actual_rad = 0.4799 -> 0.4798 -> 0.4802`, `grip_yield_deg = +10.88°..+10.90°`), `grip_to_cube_a_min_mm = 19.56`.
    8. **Place & release milestone:** `goto 0.2593 0.0468 0.0525 25` (f466: `cube_b = [0.2394, 0.0309, 0.0470]`, `cube_a = [0.2394, 0.0298, 0.0148]`, `grip_to_cube_a_min_mm = 10.56`) → `grip 1.45 15` (f481) → `hold 5` (f486): `attached: false`, `events: [..., {"frame": 475, "type": "release", "closedness": 0.223}]`, zero `jaw_contact_lost_10_frames`, `grip_to_cube_a_min_mm = 8.87` (run minimum).
    9. **Retreat & live `settle` milestone:** `goto 0.2593 0.049 0.135 30 1.45` (f516) → `settle`: `settle_fault: null`, `sim_success: true`, `checks: {"upper": "cube_b", "face_gap_m": -0.0001, "within_inset_footprint": true, "contact_with_lower": true, "no_other_support_contact": true, "upper_contacts": ["cube_a"]}`, `objects_final: {"cube_a": [0.23932, 0.02936, 0.01484], "cube_b": [0.23834, 0.02761, 0.04479]}` (`Δxy = [-0.98, -1.75] mm`), `grip_to_cube_a_min_mm = 87.09`.
  - **Export & Distillation:**
    - Exported `ENF/pilot_ep411_v4/agy3_live_trace.npz` (`516` frames, `chan_p5 = 2.0`, `chan_p99 = 55.0`).
    - Distilled `ENF/pilot_ep411_v4/agy3_cleaned.npz` (`496` frames, `chan_p5 = 2.00`, `chan_p99 = 55.00`, `41` frames `8.3%` at `chan <= 2.0`, `148` frames `29.8%` at `chan >= 55.0`): replaced the intermediate `setR real453` wrist-roll wiggle (`f36..f120`) with one smooth 65-frame cosine-ramped unfold directly to the `real453+yaw-130` hover pose and gently Gaussian-blended internal arm transit corners while pinning `folded_home`, scoop-engage, straddle/pinch/attach, place/release/dwell, and final retreat poses (`max |dq| = [0.966, 1.838, 2.363, 1.990, 2.781] deg/f`, `max |ddq| = [0.161, 0.353, 0.596, 0.332, 0.229] deg/f²`).
  - **Official Certification Replay (`ENF/pilot_ep411_v4/agy3_replay.json`, verbatim key fields):**
    ```json
    {
      "mode": "certification",
      "harness": "enforce_replay.py (protocol v2 P1+P3+D2+P4+P5)",
      "n_frames": 496,
      "grip_channel_ok": true,
      "grip_norm": {"open_ref_deg": 2.0, "closed_ref_deg": 55.0},
      "weld_reachable": false,
      "weld_ever_active": false,
      "grasp_events": [
        {
          "frame": 276,
          "type": "grasp_contact_true",
          "closedness": 0.718,
          "jaw_object_contacts": [
            ["moving_jaw_so101_v1_col_b016", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b029", "cube_b_geom"],
            ["wrist_roll_follower_so101_v1_col_b892", "cube_b_geom"]
          ],
          "min_contact_dist_m": -0.000443
        },
        {"frame": 455, "type": "release", "closedness": 0.2228}
      ],
      "enforcement_fault": null,
      "aborted": false,
      "c2_trace": {
        "attached_phase_obj_support_contact_frames": 2,
        "attached_phase_in_contact_planar_travel_m": 0.00185
      },
      "arm_self_contacts_adjacent": {},
      "alarm": {
        "baseline_frame0": {},
        "alarm_fired": false,
        "alarm_pairs": {}
      },
      "final_eval_instrumentation": {
        "sim_success_criteria_read": true,
        "objects_final": {
          "cube_a": [0.23931636483193972, 0.0293654477122664, 0.014844515360125576],
          "cube_b": [0.23847058663779483, 0.027588569440902176, 0.044789668916572]
        },
        "checks": {
          "upper": "cube_b",
          "face_gap_m": -0.0001,
          "within_inset_footprint": true,
          "contact_with_lower": true,
          "no_other_support_contact": true,
          "upper_contacts": ["cube_a"]
        }
      },
      "final_obj_pos": [0.23847, 0.02759, 0.04479]
    }
    ```

### Hints & Waypoint Cues Used (Parity Doctrine Disclosure)
- **Cameras:** Only the three registered cameras (`topcam`, `frontcam`, `wristcam`) were viewed throughout.
- **JSON Telemetry Hints Used:** `site`, `cube_b_manip`, `cube_a_base`, `q_deg`, `joint_actual_rad`, `grip_yield_deg`, `attached`, `jaw_contact`, `jaw_min_dist_mm`, `jaw_dist_fixed_*`, `jaw_dist_moving_*`, `grip_to_cube_a_min_mm`, `events`, `fault_rewound`, `ik_pos_err_mm`, `ik_rot_err_deg`, and `settle` criterion readouts.
- **Read-Only Diagnostic Check:** One read-only `mj_geomDistance` scan of `shoulder × cube_b` vs `shoulder_pan ∈ [-90°, +90°]` on the unmodified ep411 scene XML to quantify the `pan >= +13.3°` collision wall observed live on the seat (zero files modified).
- **Waypoint Cues Used:** None beyond the seat's own `init` cube coordinates and Tier 1's playbook method.

### Tier-Down Measurement — Where Tier-1 Notes SUFFICED vs Where They DID NOT
- **Where Tier-1 notes SUFFICED directly (major acceleration — solved on Attempt 1 in ~30 min):**
  1. **Top-down `real453+yaw-96` pinch-lift-place-release recipe:** Transferred verbatim once `cube_b` was clear of the shoulder — same `grip 0.29 35` pinch (`+11.4°..+11.7°` yield, `grasp_contact_true`), same `z ≈ 0.020..0.022` straddle height preserving `~9–10.5 mm` `grip_to_cube_a_min_mm` clearance during placement, same `~2 mm` above-flush release height (`target_z = 0.0525`) compensating for P5 unload Z sag, and same single-ramp `grip 1.45 15` + `hold 5` crisp release avoiding `jaw_contact_lost_10_frames`.
  2. **Grip channel discipline (`p5 = 2.0`, `p99 = 55.0`, `--object=cube_b`):** Prevented any replay normalization mismatch on the very first `enforce_replay.py` certification run.
  3. **P5 arm servo behavior (`goto` closed-loop lag/sag compensation + reading `cube_b - site` after lift):** Immediately gave sub-millimeter placement alignment (`Δxy = [-0.85, -1.78] mm`) across the long carry with zero retries.
  4. **Distillation strategy (pinning contact/attach/release plateaus + gentle Gaussian corner blending on arm joints):** Passed certification replay on the first attempt (`aborted: false`, `alarm_fired: false`).
- **Where Tier-1 notes DID NOT suffice (new ep411 physics/geometry discovered by Tier 2):**
  1. **Close-in `shoulder × cube_b` collision wall (`r < ~115 mm`):** Tier 1's ep453 scene had `cube_b` at `x = 0.2323` (far from the base), so the playbook had no warning about `shoulder` (`geom19`) sweeping into close-in cubes for `pan ∈ [+12°, +72°]`, nor about `gripper/moving_jaw × shoulder/upper_arm` self-collisions when tucking `site_x <= 0.115`.
  2. **Pre-grasp un-trapping / scoop maneuver (`closedness < 0.50`):** Tier 1's direct vertical descent onto spawn `cube_b` is geometrically impossible on ep411; Tier 2 had to author the `real453+yaw-130` low-pan (`+1.2°`) mid-height scoop (`grip 0.85` → slide to `x = 0.195` at `grip 0.80`) to relocate `cube_b` into open space before entering Tier 1's recipe.
  3. **`real411` vs `real453` for yawed top-down poses:** Even though `real411` is ep411's default orientation family, `real453+yaw-130` had lower IK rotation error (`15.0°` vs `20.0°` for `real411+yaw-118`) and cleared `moving_jaw × shoulder` cleanly where `real411+yaw-118` grazed `-5.5 µm`.
  4. **Cube colors are scene-specific:** In ep411 `cube_b` (manipulated) is **purple** and `cube_a` (base) is **green** — the reverse of ep453. Always trust `cube_b_manip` / `cube_a_base` telemetry rather than cube color.

### What the Next Driver Should Know
- Whenever a block scene spawns the manipulated cube near the robot base (`x < 0.12 m`), check `shoulder × cube` clearance across `pan` first and keep `pan` small (`<= 10°`) until the cube is scooped out to `x >= 0.125 m` using `real453+yaw-130` with `closedness < 0.50` (`joint >= 0.80`).
- Once `cube_b` is in open space (`x >= 0.125 m`), the long carry (`~135 mm` here) is completely benign on the fixed P5 stack: two smooth horizontal waypoints (`30–35` frames each at `z = 0.095`) held `joint_actual_rad` steady within `0.0004 rad` (`grip_yield_deg ≈ +10.9°`) with zero slip.

---

## TIER-2 ADDENDUM (session 3, ep411) — BLOCK-TASK PLAYBOOK

*Appended to Tier 1's `ep453 SCENE BRIEF + BLOCK-TASK PLAYBOOK` without modifying Tier 1's text. Grounded in Session 3 live measurements on `block_stack ep411` (`pilot411_v4.py`, `protocol v2 P1+P3+D2+P4+P5`, `arm_padded_v4cam.xml`).*

### 1. What Transferred Unchanged from Tier 1
- **Top-down `real453+yaw-96` grasp/lift/place/release architecture:** `grip 0.29 35` compliant pinch (`joint_actual ≈ 0.48–0.49 rad`, `grip_yield_deg ≈ +11°`, `grasp_contact_true`), straddle `target_z = 0.020–0.022` keeping jaw tips ~6–7 mm above `cube_b`'s bottom face (`grip_to_cube_a_min_mm >= 8.87 mm` through place/release), ~2 mm above-flush release (`target_z = 0.0525`) to absorb P5 unload sag, single-ramp `grip 1.45 15` + `hold 5` crisp release (eliminating `jaw_contact_lost_10_frames`), vertical clear retreat, and `p5 = 2.0 / p99 = 55.0` grip-channel anchoring (`>= 6%` frames at `joint <= 0.10`, `>= 2%` at `joint = 1.45`, `--object=cube_b`).
- **Closed-loop `goto` + post-lift `cube_b - site` compensation:** Across the extended ep411 reach (`x = 0.253`, where P5 lag reached `[-5.7, -5.2, -6.9] mm`), measuring `cube_b - site ≈ [-0.0140, -0.0145, -0.0022] m` at hover and correcting at `z = 0.062` landed `cube_b` within `Δxy = [-0.85, -1.78] mm` of `cube_a` on Attempt 1.

### 2. What Had to Change for ep411 Geometry (Measured Numbers)
- **Scene geometry:** `cube_b` (purple, manipulated) spawns close to the base at `[0.0861, -0.0431, 0.0149]`; `cube_a` (green, base) spawns at `[0.2391, 0.0296, 0.0149]` (`169.4 mm` centre-to-centre, mostly `+X`). Note cube colors are swapped relative to ep453.
- **Shoulder-turret collision wall (`pan >= +12°`):** At spawn `[0.0861, -0.0431]`, `shoulder` (`geom19`) clears `cube_b` by `+8.10 mm` at `pan = 0°`, `+4.46 mm` at `+5°`, `+0.80 mm` at `+10°`, and collides (`-1.8` to `-6.7 mm`) across `pan ∈ [+12°, +72°]`. Live pilot faulted at `pan = +13.3°` (`dist_m = -1.56e-6`).
- **Inner-reach self-collision wall (`site_x <= 0.115 m`):** Tucking `real453+yaw-96` or `real411+yaw-90` inside `site_x <= 0.112 m` at low Z triggers `gripper × shoulder` (`b649 × geom17`, `-4.4 µm`) or `moving_jaw × upper_arm/shoulder`.
- **Two-stage pick (Pre-grasp scoop in `real453+yaw-130`, then Tier-1 `real453+yaw-96`):**
  - Approach at `real453+yaw-130` (`[0.122, 0.004, 0.015]`, `pan = +1.2°`, `wrist_flex = 77.0°`, `ik_rot_err = 14.98°`) keeps `shoulder` `+7.2 mm` clear of `cube_b` and `gripper` clear of `shoulder`, while placing the moving jaw behind `cube_b`.
  - Engage at `grip 0.85 15` (`closedness = 0.444 < 0.50`, `attached: false`) and pull outward to `[0.195, -0.022, 0.015]` at `joint = 0.80` (`closedness = 0.481 < 0.50`, `attached: false`, `grip_to_cube_a_min_mm = 28.62 mm`). This relocates `cube_b` `+41 mm` in `+X` to `[0.127, -0.045]`, where `pan = +18.9°` and `site_x = 0.135` are completely clear of `shoulder` for the standard `real453+yaw-96` grasp and lift.
- **Long carry (`~135 mm` post-scoop):** Splitting the carry into two smooth horizontal stages (`30` + `35` frames at `z = 0.095`, mid-waypoint `[0.195, 0.010, 0.095]`) produced virtually zero grip slip (`joint_actual_rad`: `0.4834 -> 0.4799 -> 0.4798 -> 0.4802`).

### 3. New Failure-Map Rows (ep411 / Close-In Cubes on P5)

| Configuration / Action | Scene & Stack | Measured Failure | Verdict / Mitigation |
| :--- | :--- | :--- | :--- |
| Direct approach to close-in `cube_b` (`[0.0861, -0.0431]`) requiring `pan >= +13.3°` | ep411 (P5) | `fault_rewound`: `cube_b_geom × geom19 (shoulder)`, `-1.0` to `-3.1 µm` live (`-1.8` to `-6.7 mm` across `pan ∈ [+15°, +70°]`). Also faults on immediate recovery if arm coasts past `+13.3°`. | **HARD GEOMETRIC WALL AT SPAWN** — keep `pan <= +10°` until `cube_b` is slid out to `x >= 0.12 m`. |
| `real411+yaw-90` / `real453+yaw-96` descent at `site_x <= 0.112 m`, `z <= 0.042 m` | ep411 (P5) | `fault_rewound`: `moving_jaw × upper_arm` (`-8.2 µm` at `1.45 rad`), `moving_jaw × shoulder` (`-0.06` to `-14.8 µm`), or `gripper × shoulder` (`b649/b120 × geom17`, `-0.5` to `-6.6 µm`). | **INNER-REACH SELF-COLLISION ZONE** — keep `site_x >= 0.122 m` (for `yaw-130`) and `>= 0.130 m` (for `yaw-96`) at low Z. |
| `real411+yaw-118` descent to `[0.122..0.128, 0.004, 0.018]` | ep411 (P5) | `fault_rewound`: `moving_jaw × shoulder` (`-5.5` to `-7.5 µm`, `ik_rot_err = 20.0°`). | **USE `real453+yaw-130` INSTEAD** (`ik_rot_err = 15.0°`, clears `shoulder` cleanly down to `z = 0.015`). |
| `real453+yaw-155` wide-open descent to `[0.130, -0.005, 0.025]` | ep411 (P5) | `fault_rewound`: `moving_jaw × upper_arm` (`-3.6 µm`, wide-open jaw points back into `upper_arm`). | **EXCESS YAW** — stay at `yaw-130` (`wrist_roll ≈ -122.8°`). |
| Direct `grip 0.29 40` close on spawn `cube_b` in `real453+yaw-130` (`pan = +1.2°..+7.8°`) | ep411 (P5) | `fault_rewound`: `cube_b_geom × geom19 (shoulder)` (`-0.44` to `-5.0 µm`) as closing jaw pivots/walks `cube_b` `+Y` along shoulder face. | **DO NOT PINCH AT SPAWN** — scoop `cube_b` outward in `+X` first (`joint = 0.85 -> 0.80`, `closedness < 0.50`). |
| Pinching relocated `cube_b` too close to shoulder (`cube_b_x = 0.101`, `pan = +17.1°`) then lifting | ep411 (P5) | `fault_rewound` on first lift frame: `cube_b_geom × geom19 (shoulder)` (`-8.4` to `-12.5 µm`). | **INSUFFICIENT SCOOP DISTANCE** — track `-Y` (`[0.195, -0.022, 0.015]`) during scoop so `cube_b` reaches `x >= 0.125 m` before pinching. |

### 4. ep411 Working Recipe (Verified Live Attempt 1 + Certification Replay Clean)
Verified live in Session 3 Attempt 1 (`agy3_live_trace.npz`, 516 frames) and distilled to `agy3_cleaned.npz` (496 frames; `chan_p5 = 2.0, chan_p99 = 55.0`; `agy3_replay.json`: `aborted: false`, `alarm_fired: false`, `sim_success_criteria_read: true`).

| Step | Command (`seat.py`) | Cum. Frames | Expected Telemetry Signature (P5) |
| :--- | :--- | :---: | :--- |
| 1. Init | `init` | 1 | `folded_home`, `joint=0.0`, `cube_b=[0.0861,-0.0431,0.0149]`, `cube_a=[0.2391,0.0296,0.0149]` |
| 2. Close anchor | `hold 35` | 36 | 36 frames at `joint=0.0` (`chan=2.0`), guarantees `p5=2.0` (`>=6%` of trace) |
| 3. Unfold & open | `goto 0.155 0.004 0.105 35 1.45` | 71 | `site≈[0.163,0.0026,0.1242]`, `pan=-1.6°`, jaws open `1.45` (`chan=55.0`) |
| 4. Switch to `real453` | `setR real453` | 91 | `q_deg≈[-1.6,-51.2,23.4,97.8,6.7]`, `ik_rot_err=1.33°` *(blended out in distillation)* |
| 5. Scoop yaw | `yaw -130 30` | 121 | `R_mode="real453+yaw-130"`, `q_deg≈[2.5,-45.4,23.3,102.0,-115.7]` |
| 6. Descend to scoop | `goto 0.122 0.004 0.015 30 1.45` | 151 | `site≈[0.1249,0.004,0.0203]`, `pan=+1.2°` (`+7.2 mm` shoulder margin), `wrist_flex=77.0°` |
| 7. Hook `cube_b` | `grip 0.85 15` | 166 | `jaw_contact: true`, `attached: false` (`c=0.444`), `cube_b≈[0.0866,-0.0411,0.0136]` |
| 8. Scoop `+X` clear | `goto 0.195 -0.022 0.015 35 0.80` | 201 | `cube_b≈[0.1270,-0.0450,0.0219]` (`+41 mm` X), `attached: false` (`c=0.481`), `grip_to_cube_a=28.6 mm` |
| 9. Lift open | `goto 0.165 -0.015 0.085 20 1.45` | 221 | `site≈[0.1665,-0.015,0.0714]`, `cube_b≈[0.1260,-0.0462,0.0209]` |
| 10. Top-down yaw | `yaw 34 25` | 246 | `R_mode="real453+yaw-130+yaw34"` (`yaw-96`), `q_deg≈[10.1,-32.7,33.1,88.9,-77.8]` |
| 11. Straddle `cube_b` | `goto 0.135 -0.025 0.022 25 1.45` | 271 | `site≈[0.1333,-0.022,0.0242]`, `pan=+18.9°`, `jaw_dist_fixed_mm=+0.30` (zero pre-touch) |
| 12. Compliant pinch | `grip 0.29 35` | 306 | `attached: true`, `grasp_contact_true` (f296, `c=0.718`), `joint_actual=0.4942`, `yield=+11.70°`, `cube_b≈[0.1233,-0.0396,0.0174]` |
| 13. Attach plateau | `hold 5` | 311 | Stable pinch (`joint_actual=0.4891`, `yield=+11.41°`) |
| 14. Vertical lift | `goto 0.135 -0.025 0.095 35` | 346 | `cube_b≈[0.1229,-0.0396,0.0846]` (`+67.4 mm`), `joint_actual=0.4834`, `yield=+11.08°` |
| 15. Carry stage 1 | `goto 0.195 0.010 0.095 30` | 376 | `cube_b≈[0.1728,-0.0095,0.0909]`, `joint_actual=0.4799`, `yield=+10.88°` (zero slip) |
| 16. Carry stage 2 | `goto 0.253 0.044 0.095 35` | 411 | `cube_b≈[0.2333,0.0243,0.0859]`, `joint_actual=0.4798`, `cube_b-site≈[-0.0140,-0.0145,-0.0022]` |
| 17. Align over `cube_a` | `goto 0.2555 0.0460 0.062 30` | 441 | `cube_b≈[0.2353,0.0288,0.0564]`, `grip_to_cube_a=19.6 mm` |
| 18. Place descent | `goto 0.2593 0.0468 0.0525 25` | 466 | `cube_b≈[0.2394,0.0309,0.0470]` (`Δxy=[0.0,+1.1] mm`, ~2.2 mm above flush), `grip_to_cube_a=10.6 mm` |
| 19. Crisp release | `grip 1.45 15` | 481 | `attached: false`, `release` (f475, `c=0.223`), zero `jaw_contact_lost_10_frames`, `grip_to_cube_a=9.3 mm` |
| 20. Jaw-open dwell | `hold 5` | 486 | `joint_actual→1.387`, `jaw_contact: false`, `grip_to_cube_a=8.87 mm` (run minimum) |
| 21. Retreat clear | `goto 0.2593 0.049 0.135 30 1.45` | 516 | `cube_b=[0.2383,0.0276,0.0448]`, `grip_to_cube_a=80.7 mm` |
| 22. Settle | `settle` | 516 | `sim_success: true`, `upper_contacts: ["cube_a"]`, `Δxy=[-0.98,-1.75] mm`, `grip_to_cube_a=87.1 mm` |

---

## 2026-11-27 — AGY SOLVE SESSION 5 (gemini-4-neon-medium, tier 3) — OUTCOME

- **Session 5 wall time:** `2026-11-27 13:04 CST` to `2026-11-27 13:12 CST` (~8 minutes)
- **Harness / scene:** `pilot489_v4.py (protocol v2 P1+P3+D2+P4+P5)` / `ENF/scenes-v4/block_stack__ep489_v0__v4cam.xml` (seat scene file, differing from VM original by 1 relative include path line).
- **Attempt accounting:** **1 of 10** solving attempts used (`Attempt 1`), following one unscored familiarization pass.
- **Stop condition reached:** **(a) Replay-clean candidate exists (SUCCESS)** — `enforce_replay.py --mode=certification --object=cube_b` returned `"aborted": false`, `"alarm_fired": false`, `"weld_ever_active": false`, and `"sim_success_criteria_read": true` on `agy5_cleaned.npz`. (Candidate only — never self-certified; judging happens later by other model families.)

### Plan Outcome & Familiarization Summary (Unscored)
- Set `phase familiarization - ep489 layout and top-down yaw-96 probe`.
- Visual inspection (`topcam`/`frontcam`/`wristcam`) confirmed both cubes sit cleanly in open mat space well clear of the booth walls and well outside ep411's `x < 0.12 m` shoulder collision wall: `cube_b` (manipulated, ORANGE) at `[0.2346, 0.0970, 0.0149]`, `cube_a` (base, PURPLE) at `[0.1922, -0.0713, 0.0149]` (~174 mm lateral traverse predominantly along `-Y`, `Δx = -42.4 mm, Δy = -168.3 mm`).
- Tier 1's **top-down `real453+yaw-96`** recipe transferred directly on the very first familiarization run with zero faults:
  - At `cube_b`'s lateral reach (`r ≈ 0.254 m`, `pan ≈ -30°`), P5 servo sag pulled `site_y` `-5.4 mm` toward `y = 0` and `site_z` `-7` to `-11 mm` when `wrist_flex` saturated at `+102.0°`. Compensating `target_site` to `[0.240, 0.125, 0.026]` placed actual `site` at `[0.2352, 0.1208, 0.0145..0.0204]` with `jaw_dist_fixed_mm = +2.2..+3.57 mm` (zero pre-touch) and jaw tips ~10 mm above `cube_b`'s bottom face.
  - `grip 0.29 35` locked an immediate compliant pinch (`grasp_contact_true` f167, `closedness = 0.745`, `joint_actual_rad = 0.4905`, `grip_yield_deg = +11.485°`, walking `cube_b` `+10.0 mm` in `+Y` to `y = 0.1070`).
  - Vertical lift (`+57.2 mm` to `cube_b_z = 0.0732`) and two-stage **174 mm lateral carry** (`[0.217, 0.035, 0.095]` → `[0.194, -0.055, 0.095]`, 35 frames each) held `joint_actual_rad` steady within `0.0006 rad` (`0.4860 -> 0.4854 -> 0.4857`, `yield = +11.2°`) with zero slip.
  - Place (`[0.1947, -0.0598, 0.0619]`), single-ramp crisp release (`grip 1.45 15`, zero `jaw_contact_lost_10_frames`, `grip_to_cube_a_min_mm = 20.89 mm`), vertical retreat, and `settle` passed on the first familiarization pass (`sim_success: true`, `Δxy = [-0.66, -6.81] mm`).

### Per-Attempt Log (1 of 10 used)
- **Attempt 1 (`attempt 1 - top-down yaw-96 lateral carry and stack`):**
  - Applied three refinements from familiarization: (i) `hold 30` at `folded_home` to guarantee `chan_p5 == 2.00`; (ii) clean two-stage descent (`z = 0.045 -> 0.026`) at `[0.240, 0.125]` avoiding any pre-touch; (iii) shifted place target `+0.4 mm X, +2.5 mm Y` (using alignment hover at `z = 0.066` and accounting for the `-2.2 mm` Y settle on release) to land `cube_b` dead-center on `cube_a`.
  - **Executed sequence & engine milestones (zero `fault_rewound` across all 396 live frames):**
    1. `init` (f1) → `hold 30` (f31): `folded_home`, `joint = 0.0` (`chan = 2.0`), anchoring `chan_p5 == 2.0`.
    2. `goto 0.2346 0.121 0.085 40 1.45` (f71) → `yaw -96 35` (f106): unfolded wide open (`chan = 55.0`) into `real453+yaw-96` (`q_deg = [-28.1, 16.4, -32.4, 102.0, -115.8]`, `joint_actual_rad = 1.449`).
    3. `goto 0.240 0.125 0.045 25 1.45` (f131) → `goto 0.240 0.125 0.026 20 1.45` (f151): clean straddle at `site = [0.2353, 0.1206, 0.0204]`, `jaw_dist_fixed_mm = +3.57` (zero pre-touch), `grip_to_cube_a_min_mm = 94.39`.
    4. **Attach milestone:** `grip 0.29 35` (f186) → `hold 5` (f191): `attached: true`, `events: [{"frame": 177, "type": "grasp_contact_true", "closedness": 0.745}]`, `joint_actual_rad = 0.4885 -> 0.4867`, `grip_yield_deg = +11.371° -> +11.272°`, `cube_b = [0.2357, 0.1072, 0.0163]`.
    5. **Lift milestone:** `goto 0.240 0.125 0.095 35` (f226): `cube_b = [0.2385, 0.1089, 0.0730]` (`+56.7 mm`), `joint_actual_rad = 0.4859`, `grip_yield_deg = +11.224°`, zero slip.
    6. **174 mm lateral carry & align milestone:** `goto 0.217 0.035 0.095 35` (f261) → `goto 0.195 -0.053 0.095 35` (f296) → `goto 0.1953 -0.0555 0.066 25` (f321): traversed ~174 mm across `Y` (`pan: -30.2° -> -8.7° -> +21.3° -> +22.0°`) with `0.0013 rad` total grip variation (`0.4859 -> 0.4853 -> 0.4854 -> 0.4846`, `yield = +11.15°..+11.22°`), `grip_to_cube_a_min_mm = 25.77`.
    7. **Place & release milestone:** `goto 0.1951 -0.0573 0.0610 25` (f346: `cube_b = [0.1930, -0.0702, 0.0449]`, `cube_a = [0.1922, -0.0714, 0.0146]`, `grip_to_cube_a_min_mm = 21.08`) → `grip 1.45 15` (f361) → `hold 5` (f366): `attached: false`, `events: [..., {"frame": 355, "type": "release", "closedness": 0.223}]`, zero `jaw_contact_lost_10_frames`, `grip_to_cube_a_min_mm = 20.05` (run minimum).
    8. **Retreat & live `settle` milestone:** `goto 0.1951 -0.054 0.135 30 1.45` (f396) → `settle`: `settle_fault: null`, `sim_success: true`, `checks: {"upper": "cube_b", "face_gap_m": -0.0001, "within_inset_footprint": true, "contact_with_lower": true, "no_other_support_contact": true, "upper_contacts": ["cube_a"]}`, `objects_final: {"cube_a": [0.19233, -0.07191, 0.01484], "cube_b": [0.19354, -0.07243, 0.04479]}` (`Δxy = [+1.21, -0.53] mm`), `grip_to_cube_a_min_mm = 90.92`.
  - **Export & Distillation:**
    - Exported `ENF/pilot_ep489_v4/agy5_live_trace.npz` (`396` frames, `chan_p5 = 2.0`, `chan_p99 = 55.0`).
    - Distilled `ENF/pilot_ep489_v4/agy5_cleaned.npz` (`396` frames, `chan_p5 = 2.00`, `chan_p99 = 55.00`, `37` frames `9.3%` at `chan <= 2.0`, `117` frames `29.5%` at `chan >= 55.0`): cosine-tapered Gaussian smoothing (`sigma = 2.5` on unfold/descent `f30..f150` and lift/carry/place `f190..f345`, `sigma = 1.5` on retreat `f365..f395`) on the 5 arm joints while pinning `folded_home`, straddle/pinch/attach (`f150..f190`), place/release/dwell (`f345..f365`), and final retreat (`max |dq| = [1.259, 3.623, 2.769, 1.216, 3.965] deg/f`, `max |ddq| = [0.148, 0.469, 0.358, 0.172, 0.327] deg/f²`).
  - **Official Certification Replay (`ENF/pilot_ep489_v4/agy5_replay.json`, verbatim key fields):**
    ```json
    {
      "mode": "certification",
      "harness": "enforce_replay.py (protocol v2 P1+P3+D2+P4+P5)",
      "n_frames": 396,
      "grip_channel_ok": true,
      "grip_norm": {"open_ref_deg": 2.0, "closed_ref_deg": 55.0},
      "weld_reachable": false,
      "weld_ever_active": false,
      "grasp_events": [
        {
          "frame": 177,
          "type": "grasp_contact_true",
          "closedness": 0.7451,
          "jaw_object_contacts": [
            ["moving_jaw_so101_v1_col_b001", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b006", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b007", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b016", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b017", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b018", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b029", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b030", "cube_b_geom"],
            ["moving_jaw_so101_v1_col_b031", "cube_b_geom"]
          ],
          "min_contact_dist_m": -0.000501
        },
        {"frame": 355, "type": "release", "closedness": 0.2228}
      ],
      "enforcement_fault": null,
      "aborted": false,
      "c2_trace": {
        "attached_phase_obj_support_contact_frames": 21,
        "attached_phase_in_contact_planar_travel_m": 0.00609
      },
      "arm_self_contacts_adjacent": {},
      "alarm": {
        "baseline_frame0": {},
        "alarm_fired": false,
        "alarm_pairs": {}
      },
      "final_eval_instrumentation": {
        "sim_success_criteria_read": true,
        "objects_final": {
          "cube_a": [0.1923229859492489, -0.07190309697073076, 0.014844486282625214],
          "cube_b": [0.19350631370733323, -0.07238686023370104, 0.044790520865281735]
        },
        "checks": {
          "upper": "cube_b",
          "face_gap_m": -0.0001,
          "within_inset_footprint": true,
          "contact_with_lower": true,
          "no_other_support_contact": true,
          "upper_contacts": ["cube_a"]
        }
      },
      "final_obj_pos": [0.19351, -0.07239, 0.04479]
    }
    ```

### Hints & Waypoint Cues Used (Parity Doctrine Disclosure)
- **Cameras:** Only the three registered cameras (`topcam`, `frontcam`, `wristcam`) were viewed throughout.
- **JSON Telemetry Hints Used:** `site`, `cube_b_manip`, `cube_a_base`, `q_deg`, `joint_actual_rad`, `grip_yield_deg`, `attached`, `jaw_contact`, `jaw_min_dist_mm`, `jaw_dist_fixed_*`, `jaw_dist_moving_*`, `grip_to_cube_a_min_mm`, `events`, `fault_rewound`, `ik_pos_err_mm`, `ik_rot_err_deg`, and `settle` criterion readouts.
- **Waypoint Cues Used:** None beyond the seat's own `init` cube coordinates and Tier 1 + Tier 2's playbook method.

### Tier-Down Measurement — Where Tier-1 + Tier-2 Notes SUFFICED vs Where They DID NOT
- **Where accumulated Tier-1 + Tier-2 notes SUFFICED completely (solved on Attempt 1 in ~8 min, zero faults across both familiarization and Attempt 1):**
  1. **Direct transfer of Tier 1's `real453+yaw-96` top-down architecture:** Because both ep489 cubes spawn in open space (`x >= 0.192 m`, well clear of Tier 2's `x < 0.12 m` shoulder collision wall), the entire Tier-1 recipe transferred directly without needing Tier 2's pre-grasp scoop.
  2. **Tier 2's two-stage long-carry guidance:** Splitting the ~174 mm lateral traverse across `Y` (`pan: -30.2° -> +21.3°`) into two 35-frame horizontal stages at `z = 0.095` held `joint_actual_rad` steady within `0.0013 rad` with zero slip.
  3. **P5 closed-loop `goto` compensation, release timing, and grip-channel discipline:** Tier 1's `grip 0.29 35` + `hold 5` pinch, `~2 mm` above-flush place height, single-ramp `grip 1.45 15` + `hold 5` crisp release (avoiding `jaw_contact_lost_10_frames`), `p5 = 2.0 / p99 = 55.0` anchoring, `--object=cube_b`, and plateau-pinned Gaussian distillation all worked verbatim on the first try.
- **Where scene-specific calibration was still required (minor, resolved in 1 familiarization pass):**
  1. **Larger P5 sag at extended lateral reach (`r ≈ 0.265 m`, `pan ≈ -30°`):** At ep489's pick (`y = +0.097`), `wrist_flex` saturated at `+102.0°` even at straddle height (`z = 0.026`), pulling `site_y` `-4.5` to `-5.4 mm` inward and `site_z` `-5.6` to `-11.5 mm` downward vs `target_site`. Raising `target_z` from `0.020` to `0.026` and shifting `target_y` `+4 mm` (`cube_b_y + 0.028`) restored clean `jaw_dist_fixed_mm = +3.57 mm` clearance and raised `grip_to_cube_a_min_mm` at place/release to `>= 20.05 mm` (more than double ep453/ep411's ~8–9 mm margin).
  2. **Higher `cube_b - site` vertical drop (`-7.5` to `-11.6 mm` in Z):** Picking higher on `cube_b`'s upper half (`target_z = 0.026`) increased `site_z - cube_b_z` at place from ~`5 mm` (ep453/ep411) to ~`12–13 mm`, so the place target was `target_z = 0.0610` rather than `0.052`, which in turn gave `>= 20.05 mm` `grip_to_cube_a_min_mm` clearance throughout release.

### What the Next Driver Should Know
- For any open-space block scene (`x >= 0.13 m` for both cubes, even with ~175 mm lateral `±Y` traverse), Tier 1's `real453+yaw-96` recipe + Tier 2's two-stage `z = 0.095` carry is a turnkey solution on the fixed P5 stack.
- When `cube_b` sits at large `|y|` (`r >= 0.25 m`), expect extra P5 radial/Z sag (`~5 mm` XY, `~6–11 mm` Z) and `wrist_flex = +102.0°` saturation at pick: compensate `target_z ≈ 0.026` and `target_y ≈ cube_b_y + 0.028` so `jaw_dist_fixed_mm > +1 mm`, then read `cube_b - site` at hover (`z = 0.066`) to compute the exact place `target_z` (`~0.061` here, giving `20+ mm` `grip_to_cube_a` clearance).

---

## TIER-3 ADDENDUM (session 5, ep489) — BLOCK-TASK PLAYBOOK

*Appended to Tier 1's `ep453 SCENE BRIEF + BLOCK-TASK PLAYBOOK` and Tier 2's `TIER-2 ADDENDUM (session 3, ep411)` without modifying prior text. Grounded in Session 5 live measurements on `block_stack ep489` (`pilot489_v4.py`, `protocol v2 P1+P3+D2+P4+P5`, `arm_padded_v4cam.xml`).*

### 1. What Transferred Unchanged from Tier 1 + Tier 2
- **Entire `real453+yaw-96` pick → lift → two-stage carry → align → place → crisp release → retreat architecture:** `hold 30` full-close anchor (`p5 = 2.0`), wide-open `1.45` unfold (`p99 = 55.0`), `grip 0.29 35` + `hold 5` compliant pinch (`joint_actual_rad ≈ 0.486–0.489`, `grip_yield_deg ≈ +11.2–11.4°`, `grasp_contact_true`), vertical lift to `z = 0.095`, two-stage horizontal carry (`35 + 35` frames at `z = 0.095`, zero slip), closed-loop alignment hover + place descent, single-ramp `grip 1.45 15` + `hold 5` crisp release (zero `jaw_contact_lost_10_frames`), vertical clear retreat, `--object=cube_b`, and plateau-pinned Gaussian distillation.
- **No pre-grasp scoop needed:** Because ep489's `cube_b` spawns at `x = 0.2346 m` (well clear of Tier 2's `x < 0.12 m` shoulder collision wall), Tier 1's direct top-down straddle succeeded on the first try.

### 2. What Had to Change for ep489 Geometry (Measured Numbers)
- **Scene geometry:** `cube_b` (manipulated, ORANGE) spawns at `[0.2346, 0.0970, 0.0149]`; `cube_a` (base, PURPLE) spawns at `[0.1922, -0.0713, 0.0149]` (`173.6 mm` centre-to-centre, predominantly lateral `-Y`: `Δx = -42.4 mm, Δy = -168.3 mm`).
- **Extended-reach P5 sag compensation at pick (`r ≈ 0.265 m`, `pan ≈ -30.3°`, `wrist_flex = +102.0°`):**
  - At `target_site = [0.236, 0.120, 0.025]`, P5 sag pulled actual `site` to `[0.2311, 0.1146, 0.0225]` (`-5.4 mm` in Y, grazing `jaw_dist_fixed_mm = -1.66`).
  - Adjusting straddle target to `[0.240, 0.125, 0.026]` (`cube_b_y + 28 mm`, `z = 26 mm`) landed actual `site = [0.2353, 0.1206, 0.0204]` with `jaw_dist_fixed_mm = +3.57 mm` (clean zero pre-touch) and jaw tips ~10 mm above `cube_b`'s bottom face.
- **Lateral carry across `Y` (`+0.125 -> -0.053`, `pan: -30.2° -> +21.3°`):**
  - Two 35-frame horizontal stages (`[0.217, 0.035, 0.095]` → `[0.195, -0.053, 0.095]`) carried `cube_b` across 174 mm with only `0.0005 rad` grip variation (`0.4859 -> 0.4853 -> 0.4854`, `yield = +11.2°`, zero slip).
- **Higher place `target_z` (`0.0610`) & double `grip_to_cube_a` clearance (`>= 20.05 mm`):**
  - Because the higher straddle (`target_z = 0.026`) gripped `cube_b`'s upper half (`cube_b_z - site_z = -11.6` to `-13.8 mm` at place), the place target was `z = 0.0610` (vs `0.052` in ep453/ep411), keeping `grip_to_cube_a_min_mm >= 20.05 mm` through place and release.

### 3. New Failure-Map Rows (ep489 / Large Lateral Reach on P5)
No hard failures or `fault_rewound` events occurred in Session 5 (0 faults across familiarization and Attempt 1). One minor calibration sensitivity was measured:

| Configuration / Action | Scene & Stack | Measured Behaviour | Mitigation |
| :--- | :--- | :--- | :--- |
| Uncompensated ep453 straddle offsets (`target = [cube_b_x+1.4mm, cube_b_y+23mm, 0.025]`) at `y = +0.097` (`r ≈ 0.265 m`, `pan ≈ -30°`) | ep489 fam (P5) | P5 servo sag pulled actual `site` `[-4.9, -5.4, -2.5] mm` inward, dropping `jaw_dist_fixed_mm` to `-1.66 mm` (close graze before pinch). | Shift straddle target outward (`[0.240, 0.125, 0.026]`, i.e., `cube_b_y + 28 mm`) → `jaw_dist_fixed_mm = +3.57 mm`, clean zero pre-touch. |

### 4. ep489 Working Recipe (Verified Live Attempt 1 + Certification Replay Clean)
Verified live in Session 5 Attempt 1 (`agy5_live_trace.npz`, 396 frames) and distilled to `agy5_cleaned.npz` (396 frames; `chan_p5 = 2.0, chan_p99 = 55.0`; `agy5_replay.json`: `aborted: false`, `alarm_fired: false`, `sim_success_criteria_read: true`).

| Step | Command (`seat.py`) | Cum. Frames | Expected Telemetry Signature (P5) |
| :--- | :--- | :---: | :--- |
| 1. Init | `init` | 1 | `folded_home`, `joint=0.0`, `cube_b=[0.2346,0.0970,0.0149]`, `cube_a=[0.1922,-0.0713,0.0149]` |
| 2. Close anchor | `hold 30` | 31 | 31 frames at `joint=0.0` (`chan=2.0`), guarantees `p5=2.0` (`>=6%` of trace) |
| 3. Unfold & open | `goto 0.2346 0.121 0.085 40 1.45` | 71 | `site≈[0.2408,0.1155,0.1015]`, `pan=-31.0°`, jaws open `1.45` (`chan=55.0`) |
| 4. Top-down yaw | `yaw -96 35` | 106 | `R_mode="real453+yaw-96"`, `q_deg≈[-28.1,16.4,-32.4,102.0,-115.8]`, `joint_actual≈1.449` |
| 5. Hover over `cube_b` | `goto 0.240 0.125 0.045 25 1.45` | 131 | `site≈[0.2371,0.1194,0.0416]`, `jaw_dist_fixed_mm=+7.77`, `grip_to_cube_a=99.6 mm` |
| 6. Straddle `cube_b` | `goto 0.240 0.125 0.026 20 1.45` | 151 | `site≈[0.2353,0.1206,0.0204]`, `jaw_dist_fixed_mm=+3.57` (zero pre-touch), `grip_to_cube_a=94.4 mm` |
| 7. Compliant pinch | `grip 0.29 35` | 186 | `attached: true`, `grasp_contact_true` (f177, `c=0.745`), `joint_actual=0.4885`, `yield=+11.37°`, `cube_b≈[0.2354,0.1070,0.0160]` |
| 8. Attach plateau | `hold 5` | 191 | Stable pinch (`joint_actual=0.4867`, `yield=+11.27°`) |
| 9. Vertical lift | `goto 0.240 0.125 0.095 35` | 226 | `cube_b≈[0.2385,0.1089,0.0730]` (`+56.7 mm`), `joint_actual=0.4859`, `yield=+11.22°` |
| 10. Lateral carry 1 | `goto 0.217 0.035 0.095 35` | 261 | `cube_b≈[0.2161,0.0252,0.0795]`, `pan=-8.7°`, `joint_actual=0.4853` (zero slip) |
| 11. Lateral carry 2 | `goto 0.195 -0.053 0.095 35` | 296 | `cube_b≈[0.1945,-0.0616,0.0796]`, `pan=+21.3°`, `joint_actual=0.4854`, `grip_to_cube_a=52.0 mm` |
| 12. Align over `cube_a` | `goto 0.1953 -0.0555 0.066 25` | 321 | `cube_b≈[0.1924,-0.0675,0.0520]`, `grip_to_cube_a=25.8 mm` |
| 13. Place descent | `goto 0.1951 -0.0573 0.0610 25` | 346 | `cube_b≈[0.1930,-0.0702,0.0449]`, `grip_to_cube_a=21.1 mm` |
| 14. Crisp release | `grip 1.45 15` | 361 | `attached: false`, `release` (f355, `c=0.223`), zero `jaw_contact_lost_10_frames`, `grip_to_cube_a=20.4 mm` |
| 15. Jaw-open dwell | `hold 5` | 366 | `joint_actual→1.387`, `jaw_contact: false`, `grip_to_cube_a=20.05 mm` (run minimum) |
| 16. Retreat clear | `goto 0.1951 -0.054 0.135 30 1.45` | 396 | `cube_b=[0.1935,-0.0724,0.0448]`, `grip_to_cube_a=84.8 mm` |
| 17. Settle | `settle` | 396 | `sim_success: true`, `upper_contacts: ["cube_a"]`, `Δxy=[+1.21,-0.53] mm`, `grip_to_cube_a=90.9 mm` |
