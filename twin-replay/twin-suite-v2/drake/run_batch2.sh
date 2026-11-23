#!/bin/bash
set -u
cd /home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake
DK=/home/openclaw/muse-work/zk0-sim-eval2-20261115/drake/venv/bin/python3
for spec in "ring_insert 7" "ring_insert 33" "ring_insert 167" "ring_insert 195" "tool_insert 683" "tool_insert 697" "tool_insert 852" "tool_insert 883"; do
  echo "== DRAKE $spec"; nice -n 19 $DK drake_replay.py $spec --out results 2>&1 | tail -1
done
$DK compare.py mujoco_results results results/equiv_summary.json
echo BATCH2_DONE
