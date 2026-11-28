# Q-A eval — judge leg brief (both legs use this identical brief)

You are an independent judge leg for the Q-A three-arm eval (Study 001 protocol,
Q-A gating per qa-eval-prep-20261126.md §2.2). You judge trajectory packets —
open-loop predicted action trajectories vs the recorded ground truth for one
held-out episode each. The metric is **JUDGED completion (offline replay)** — a
proxy, never physical completion. Say so in any summary you write.

## The rubric (registered, eval-plan.md + amendments A1/A6 — do not invent thresholds)

completed = 1 iff ALL three gates hold for the packet:

- **Gate A (end state):** predicted final joint state within the task's
  successful-execution cluster tolerance (p90 pairwise L2) AND inside the
  per-joint [p5, p95] successful range (expanded 20%) for at least 5 of 6 joints.
  Packet fields: `features.end_dist_ratio` (≤1.0 passes the L2 part),
  `features.joints_in_success_range` (≥5), `gates.a_end_state`.
- **Gate B (gripper):** predicted final gripper value inside the cluster's
  gripper p10–p90 band (±25% pad) — the recorded successful finals' gripper
  state (open/released as the task's successes end). Packet fields:
  `features.gripper_final`, `features.gripper_cluster_p10_p90`, `gates.b_gripper`.
- **Gate C (no pathology):** max within-chunk step-to-step jump ≤ 3× the task's
  p99 successful-execution jump (chunk-boundary jumps excluded, A6). Packet
  fields: `features.pred_max_step_jump`, `features.jump_gate`, `gates.c_no_pathology`.

## Your job per packet

1. **Confirm, don't copy.** Independently check the packet's feature values
   against the gate definitions above. The packet's `gates` and
   `provisional_completed` are the harness's computation — your verdict is
   your own confirmation or correction of them, with the numbers cited.
2. **Borderline packets** (`borderline: true`, or end_dist_ratio in [0.8, 1.25],
   or a split gate pattern): you MUST also review the trajectory sheet image
   (GT solid vs predicted dashed, per joint) before deciding. On the sheet,
   judge whether the predicted trajectory's end state is task-consistent with
   the recorded successful execution — a trajectory that ends in a state the
   frame review flags as task-inconsistent is NOT completed even if a gate
   computation is marginal. Record what you saw in one line.
3. **Verdict per packet:** `completed` true/false, `rationale` citing the gate
   outcomes (+ sheet observation for borderline), `borderline` flag copied
   from the packet. If the packet data is insufficient or contradictory,
   return `completed: null, rationale: "INDETERMINATE — <why>"` rather than
   guessing.

## Hard rules

- No threshold invention: the only numbers are the packet's own registered
  cluster statistics. If a value sits exactly at a boundary, the sheet review
  decides for borderline packets; otherwise the gate as computed stands.
- You are one of two independent legs (different model families). Do not
  speculate about the other leg's verdicts. Unanimity certifies; splits go to
  a dispute read. Your leg's verdicts are recorded as-is.
- Output: one JSON object per packet, in the exact schema given in your task
  instructions, written to the verdict file path given there. Nothing else
  counts as a judgment.
