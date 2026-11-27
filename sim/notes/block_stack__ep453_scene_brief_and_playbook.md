<!-- Provenance: authored by the session-2 driver (Gemini 4 Neon XHigh) at the close of its certified ep453 solve, 2026-11-27; extracted verbatim from the session driver notes. This is the brief the next tier drives from (SOP section 6.8 artifact). -->

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
