# Reproduce the first certified sim solution (block_stack ep453)

Everything below uses public assets only. Expected time: ~10 minutes,
of which the replay itself takes ~1 minute.

## 1. Get the code (this repo) and the episode package (the dataset)

```bash
git clone https://github.com/ivelin/zk0.git
# the certified episode package:
#   https://huggingface.co/datasets/zk0bot/sim-scene-solutions
#   -> folder certified/ep453/
# e.g. with the huggingface CLI:
huggingface-cli download zk0bot/sim-scene-solutions \
  --repo-type dataset --include "certified/ep453/*" --local-dir sim-scene-solutions
```

The package contains the LeRobot episode (`episode/`), the cleaned
trajectory the certification replay consumed (`agy2_cleaned.npz`), the
official replay record (`agy2_replay.json`), the certification record
with both judges' verdicts (`certification_record.md`), and the
path-exposing replay video (`agy2_total_solution_replay.mp4`,
sha256 `968dd0816ca7e4be7c520711d4f454602d8a800b315fd81d7d404969c7582d4a`).

## 2. Install the engine

```bash
python3 -m venv venv && source venv/bin/activate
pip install "mujoco==4.0.1" numpy
```

MuJoCo 4.0.1 is the exact engine version of record. No other
dependencies are needed for the replay.

## 3. Run the certification replay

From the repo root (the scene path must be **absolute** — MuJoCo
resolves the arm's mesh directory from it):

```bash
cd zk0
python sim/vmstack/enforced/enforce_replay.py block_stack 453 agy2 \
  "$PWD/sim/vmstack/enforced/scenes-v4/block_stack__ep453_v0__v4cam.xml" \
  /path/to/sim-scene-solutions/certified/ep453/agy2_cleaned.npz \
  my_replay.json --mode=certification --object=cube_b
```

## 4. Compare against the official record

Your `my_replay.json` should match the package's `agy2_replay.json`
(sha256 `37b81117d8006c7ad661410d04ba34158f5c0a6143acf5efafddbe79ae69420c`)
on every field except `scene` and `trajectory` (your input paths).
The headline fields:

- `harness`: `enforce_replay.py (protocol v2 P1+P3+D2+P4+P5)`
- `n_frames`: 374 · `aborted`: false · `enforcement_fault`: null
- `alarm.alarm_fired`: false · `weld_ever_active`: false
- `grip_channel_ok`: true
- `final_eval_instrumentation.sim_success_criteria_read`: true
- `final_eval_instrumentation.objects_final.cube_b`:
  `[0.25693170991895303, 0.04630475749868331, 0.04480197082047864]`

This exact comparison was run against this snapshot before
publication (2026-11-27) and matched bit-for-bit; the snapshot record
is `sim/vmstack/enforced/VERSION.md`.

## 5. (Optional) Watch it, and read how it was judged

- The video in the package is the rendering of the official replay:
  top | front | wrist cameras, the gripper path visible throughout.
- `certification_record.md` carries both judges' verdicts and
  commentary, the judge model ids, and the seal (scene / arm / engine /
  harness hashes) the certification is pinned to.
- The driver's own brief for this scene — the plan, the measured
  failure map, the working recipe — is `sim/notes/block_stack__ep453_scene_brief_and_playbook.md`.

## 6. (Optional) Run the driver seat

The seat contract is `sim/vmstack/enforced/driver-rotation-20261127/DRIVER_SEAT.md`.
The seat server resolves the campaign tree layout below its root, so
recreate it with a symlink first:

```bash
mkdir -p seatroot/validator-sets-20261123/coverage-topup-work
ln -s "$PWD/sim/vmstack" seatroot/validator-sets-20261123/coverage-topup-work/vmstack
# then follow DRIVER_SEAT.md from seatroot/
```
