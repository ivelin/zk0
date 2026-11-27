# zk0 sim validation kit

The enforced-physics simulation stack behind zk0's certified sim
solutions, plus the driver notes that accumulate as solutions are found.
Everything here is published so that **a guest with access only to
public assets can reproduce a certified solution end to end**: this
repo for the code, the Hugging Face dataset
[`zk0bot/sim-scene-solutions`](https://huggingface.co/datasets/zk0bot/sim-scene-solutions)
for the scenes/solutions data and the certified episode packages.

Licence: Apache-2.0 (code). Datasets: CC BY 4.0.

## Layout

| Path | What it is |
|---|---|
| `vmstack/enforced/` | The enforced harness and everything it loads: arm model v4 (+ builder + meshes), the P5 replay/pilot code, grip drive, admission gate, canary drills, scenes, seat contract, drive-configuration record. Start with `vmstack/enforced/VERSION.md` — it names the stack version and every file's sha256. |
| `vmstack/` | Support modules the harness imports: `twin_params.py` (criteria + physics constants), `build_scenes.py`, `replay.py` (criteria reader used by the final-eval instrumentation). |
| `arm_fk.py` | Arm forward/inverse kinematics used by the pilot. |
| `staging/scenes/` | Calibration inputs the scene builder reads at import (`calibration_v2.json`, `calibration_fk_affine.json`). |
| `notes/` | Driver notes per scene (SOP §6.8 artifacts): the scene brief + task playbook the next driver tier starts from. |
| `REPRODUCE.md` | The guest runbook: reproduce the first certified solution (block_stack ep453) step by step. |

## The certified set

Certified solutions live in the dataset, one package per scene under
`certified/<episode>/`, indexed by the dataset's `manifest.jsonl`.
First entry (2026-11-27): **block_stack ep453** — the first sim solution
certified under the current protocol (enforced full-motion replay +
two frontier judges — both sandboxed and blind, at most one from the
driver's family — with unanimity required).

Solutions produced before the enforced stack existed are labelled
EXECUTED in the manifest, with their exact stage; they are not
certified, whatever older wording elsewhere in the dataset may say —
corrections are appended to the dataset README as families re-certify,
never silently edited.

## Method in one paragraph

A frontier model drives the arm by teleoperation inside the running
simulation — small motion increments through the same action channel a
VLA policy emits (per-step joint targets + gripper channel, LeRobot
action space), never teleporting, never writing simulator state. The
live trace is distilled into one smooth trajectory, saved as a LeRobot
episode, and certified only if (1) the recorded actions replay under
the enforced harness with no abort and a silent alarm trace, and
(2) two frontier judges, both completely sandboxed and blind to the
driver's session and to each other, watching the path-exposing video,
independently pass it with commentary. The driver never judges its own
solution; at most one judge may share the driver's family, and the
second judge is from a different family than the driver.
