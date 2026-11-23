# Study 001 Twin-Replay — Pre-registration (2026-11-22)

Written by the twin-replay coordinator BEFORE any twin validation run.
Founder directive: 2026-11-22 17:15 CT (see MEMORY.md, TWIN-REPLAY EVAL DIRECTIVE).
Scope this round: Phases 1–5 (scene inventory, physics parameters, per-episode
initial states, twin validation by demo replay, independent verification).
Phase 6 (closed-loop checkpoint evals) and Phase 7 (blinded rollouts) are NOT
authorized this round and must not start.

## Frozen inputs

- Held-out eval set: 100 episodes (20/task × 5 tasks), seed 20261122, disjoint
  from training (proven in the eval workstream). Episode identity = (task,
  episode id) as it appears in
  `run/eval-traj-20261122/records/` (checkpoint prefix stripped) and
  `run/eval-traj-20261122/split/split.json`.
- Recorded trajectories: `records/<checkpoint>__<task>__ep<id>.npz`, key `gt`
  (T×6 joint positions, degrees; joint order shoulder_pan, shoulder_lift,
  elbow_flex, wrist_flex, wrist_roll, gripper). The `gt` array is identical
  across checkpoints for the same episode (verify once, then use `base` copy).
- Recorded outcome per episode: the FINAL judgment in the eval results
  (`results/` + per-episode judgment records) plus the recorded final frames.
  For demo-replay validation the relevant truth is the HUMAN DEMO's outcome
  (the recorded episode itself), not any checkpoint's judged completion.

## Twin validation — pre-registered criteria

Each episode's RECORDED joint trajectory is executed in its twin scene:
arm driven kinematically (qpos set per recorded frame, 30 fps), objects fully
dynamic, grasp via the recorded gripper trace (grasp mechanism — e.g.
proximity weld on close — is a declared modeling device, documented in the
parameter table, never hidden).

After the trajectory ends, physics settles for 1.0 s before final state read.

### Primary criterion: outcome agreement (per episode)

Sim final outcome class == recorded demo outcome class, where outcome class
is defined mechanically per task:

- **block_stack** — SUCCESS iff one cube rests stably on the other at settle:
  upper cube in contact with lower cube's top, upper cube's center within the
  lower cube's top footprint (inset by 25%), no other support contact.
- **eye_drops_to_basket** — SUCCESS iff the box's final resting support is the
  basket (box center inside the basket's interior footprint and box supported
  by basket floor/walls), not the platform or the mat.
- **ring_insert** — SUCCESS iff the ring is seated on the peg: peg passes
  through the ring hole, ring's lowest point within 1 cm of the board top
  surface around the peg.
- **tool_insert** — SUCCESS iff the tool's final pose lies in its target tray
  slot: tool center within the slot's footprint and tool axis within 20° of
  the slot axis, resting in the tray.
- **cable_clip** — SUCCESS iff the cable is seated in the clip (cable segment
  inside the clip channel with the clip closed over it). KNOWN TRAP: the
  held-out cable_clip references are FAILED human executions; the expected
  outcome class for those episodes is FAILURE, and reproducing the failure
  is agreement.

### Secondary criterion: final object position

Manipulated object's final planar (x,y) position vs its recorded final
position (estimated from recorded final frames by the same estimation method
as the initial poses, Phase 3 method): within **3.0 cm** (block_stack,
eye_drops_to_basket, tool_insert); ring_insert **2.0 cm** when not engaged
(boolean governs when engaged); cable_clip exempt (deformable — outcome
class only).

### Certification bar (per task), fixed in advance

A task's twins are CERTIFIED iff, over episodes with medium-or-better
initial-pose confidence:
- outcome agreement ≥ 80%, AND
- no unexplained systematic failure mode (a repeated miss pattern with no
  diagnosed cause in the validation report).

Below the bar: the task's twins are reported NOT CERTIFIED with the diagnosed
causes; Phase 6 may not use them. The independent verifier (Phase 5) rules on
whether this pre-registration was honored; the coordinator does not certify
its own builders' work.

## Honesty rules (binding on all builders)

1. Every physics parameter carries a source: measured-from-recording
   (method stated), physical-prop knowledge (prop identified), or
   literature/engine default (labeled DEFAULT). No silent inventions.
2. Parameter changes during validation are GLOBAL (per task/material), never
   per-episode. Every change is logged with reason and its before/after
   effect on the full validation set.
3. Episodes whose initial state cannot be pinned are named and excluded from
   the validation denominator — never silently guessed into place.
4. Grasp-device artifacts are reported as twin limitations, not edited away.
5. The 3 sealed BENCH-002 held-out tasks in the dataset are out of scope —
   do not extract, view, or use their episodes.

## DGX safety

Training (sequential chain + specialists) runs on the DGX until ~02:00 CT.
Twin work is CPU-only, `nice -n 10` or lower priority, ≤ 8 concurrent
processes, whole-box ceiling rules apply. No GPU use. Check load before
batch jobs.
