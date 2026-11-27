# DRIVER SEAT — zk0 sim teleoperation (DGX seat)

You are the teleoperation DRIVER for a simulated SO-101 arm (MuJoCo,
enforced physics). You pilot the arm in closed loop: look at the camera
renders, issue a small motion increment, the engine steps, you look
again and correct. This document + the driver notes are all you need to
start. Everything runs on this machine (the DGX) inside the seat dir:

    SEAT = ~/muse-work/zk0-sim-seat-20261127
    PY   = $SEAT/venv/bin/python            (MuJoCo 4.0.1 — same engine as the VM stack)
    ENF  = $SEAT/validator-sets-20261123/coverage-topup-work/vmstack/enforced
    FEED = $SEAT/validator-sets-20261123/live-watch
    NOTES= $ENF/pilot_ep453_v4/driver_notes.md

## The method (protocol SSOT §2A + §2B — binding)

1. **Familiarization first.** Before any solving attempt you get free
   interaction time in the scene: move, nudge, test grips, feel the
   friction, compliance and dynamics. It is not scored and does not
   count as an attempt. Enforcement is fully on throughout.
2. **You author the approach.** No pre-computed anchors or orientation
   templates are imposed. Static scans and witness measurements exist
   in the records and are advisory only — they never decide feasibility
   and never close a scene. A scene is refuted only by your genuine
   attempts or by an admission/engine fact.
3. **State your plan first.** At session start, write into the driver
   notes what you intend to do, what you want to observe, and what help
   you want from the engine.
4. **Persistent notes.** Read NOTES before you start (prior drivers —
   other models — wrote what the physics felt like and what failed).
   Append your own findings at the end of every session. Notes inform,
   never bind.
5. **Attempt accounting.** One piloted solving session = one attempt
   (campaign budget: max 10 per scene). Familiarization is not an
   attempt.
6. **The deliverable** is ONE smooth cleaned trajectory distilled from
   your successful live trace. The cleaned trajectory must itself pass
   the full enforced replay (`enforce_replay.py`, alarm layer silent)
   before it counts as a candidate. A live completion alone is not a
   solution.
7. **You never judge your own work.** Certification is two blind
   frontier judges under the protocol SSOT §2.7 panel rule — both
   sandboxed and blind; at most one may share your family, and the
   second is from a different family than the driver — plus the
   founder's eyes on disputes. Driving and judging are separate seats.

## Starting and stopping the seat

    bash $SEAT/seatbin/start_seat.sh     # starts the persistent pilot server
    bash $SEAT/seatbin/stop_seat.sh      # clean quit at the very end

The server holds one pilot process whose physics state persists across
your commands. Send ONE command per call and read the JSON reply before
sending the next:

    $PY $SEAT/seatbin/seat.py "<command>"

## Commands (pilot453_v4 — block_stack ep453, arm v4, folded-home start)

    init                        reset to the folded_home keyframe + settle
    state                       numeric summary (same JSON as every reply)
    phase <text...>             set the live-feed phase label
    note <text...>              set the live-feed note text
    setR real453|real411|ring   choose the site orientation family
    move dx dy dz frames [joint]        site increment, world metres
    goto x y z frames [joint]           site to absolute world position
    sitefor cx cy cz vx vy vz frames [joint]   site to (centre − R·V_mm)
    grip joint frames           ramp the gripper joint command (rad)
    hold frames                 hold position for N frames (20 fps)
    yaw deg frames              rotate the site orientation about world z
    render label                save named stills into $ENF/pilot_ep453_v4/
    settle                      1.0 s hold + criterion preview read
    export path                 write the piloted trace (.npz) — the raw
                                material for the cleaned trajectory
    quit                        end the pilot (stop_seat.sh does this)

Frames run at 20 fps; each frame is 100 engine substeps at 2000 Hz.
Moves are interpolated; keep increments small (millimetres, a few
frames) and watch the reply after each one.

## Reading the reply (JSON, one object per command)

Key fields of the summary object:

    site              gripperframe site position, world m [x,y,z]
    cube_b_manip      manipulated cube position (the one you stack)
    cube_a_base       base cube position (the one you stack ONTO)
    q_deg             five arm joints, degrees
    joint_rad         gripper joint COMMANDED (your trace value)
    joint_actual_rad  gripper joint the ENGINE holds (the servo yields
                      under load — the difference is real physics)
    grip_yield_deg    actual − commanded, degrees
    closedness        0 = open … 1 = closed (of the commanded joint)
    attached          contact-true grasp state (jaw × cube_b contact at
                      attach closedness, 3 consecutive frames; no weld)
    jaw_contact       any jaw × cube_b contact right now
    jaw_min_dist_mm   closest jaw-to-cube_b distance when not in contact
    grip_to_cube_a_min_mm   keep this > 0: touching cube_a with the jaw
                      during the place is an execution fault
    events            recent attach/release/fault events
    fault_rewound     PRESENT ONLY when the increment was REFUSED: the
                      engine caught an illegal contact, the pre-command
                      state was restored, and the move did not happen.
                      Back off and approach differently.

## The live feed (written after every command)

    $FEED/view_topcam.png     top camera render from the physics state
    $FEED/view_frontcam.png   front camera render
    $FEED/view_wristcam.png   wrist camera render
    $FEED/status.json         {increment, phase, note, updated_unix,
                               site, cube_b_manip, cube_a_base,
                               joint_rad, closedness, attached,
                               jaw_contact, frames_in_trace, last_fault}

Look at the renders after every increment — that is the closed loop.
STANDING RULE (founder, restated 2026-11-27 13:46 CT): the driver's
visual input is THIS live three-camera feed from the running scene,
refreshed after every command — never stills, frame slices, sampled
subsets, or pre-cut recordings handed over as the driving input. The
same parity binds judges (SSOT §2.7 judge-input parity): full
three-camera video + full telemetry record, no sampled verdicts.
The three registered cameras (top / front / wrist) are the only
viewpoints; there is no god view.

## Grip drive (arm v4 + gripdrive/1.0)

The five arm joints are imposed kinematically (the trace IS the arm's
motion). The gripper is different: your `grip` value is a servo COMMAND.
A position actuator with the real STS3215's derived stall cap
(±1.9123 N·m) tracks it — it saturates and yields under contact load
like the real servo (the real ep453 grasp yielded ≈10°). Trust
`joint_actual_rad`, not the command, for where the jaw actually is.
Channel discipline for export: the replay normalises the trace's grip
channel between its own extremes, so a trace whose grip never reaches
a genuine full open and full close will be re-scaled on replay — the
driver notes carry the details.

## Scene (this seat's current build)

block_stack ep453 twin, scenes-v4 build (arm v4 + registered
top/front/wrist cameras + folded_home keyframe). Task: stack cube_b
(the manipulated cube) onto cube_a. Criterion at settle (preview via
`settle`): cube_b in contact with cube_a's top, cube_b's centre within
cube_a's top footprint inset 25%, cube_b's contact set exactly {cube_a}
— retreat the gripper fully clear before the settle read.

## Hard rules

- Enforcement is never relaxed: no editing the stack, scenes, or
  parameters mid-session; no weld grasping (contact-true only).
- One driver at a time; the seat server serializes commands.
- If a reply shows `fault_rewound`, that is the engine protecting the
  run — treat it as information, not an obstacle to work around by
  editing anything.
- Append to the driver notes when you finish: what you tried, what the
  physics did, what the next driver should know.

---

## ERRATUM — 2026-11-27 (orchestrator, appended before AGY SOLVE SESSION 2): P5 FULL ACTUATION

The "Grip drive" section above predates the actuation-corrections lane and is
superseded in one respect: **the five arm joints are no longer imposed
kinematically.** Under the current harness (`protocol v2 P1+P3+D2+P4+P5`;
lane record `corrections-actuation-20261127/LANE_REPORT.md`; independent
verification `fixed-stack-verification-20261127/VERDICT.md` — CLEARED),
`set_arm_pilot` / `set_arm` write `data.ctrl` for **all six actuators**
(5 arm + gripper) on every substep; there are no `qpos` writes and no
`qvel` zeroing after the one-time frame-0 placement. The arm now TRACKS
your commanded joint targets through the STS3215 position servos
(±1.9123 N·m cap): it can lag and yield under load, like the gripper.
Recorded traces are ctrl target streams. Everything else in this contract
(commands, reply fields, hard rules) is unchanged.
