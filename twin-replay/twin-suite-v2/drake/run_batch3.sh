#!/bin/bash
set -u
cd /home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake
DK=/home/openclaw/muse-work/zk0-sim-eval2-20261115/drake/venv/bin/python3
declare -A EPS=( [block_stack]="411 453 489 498" [eye_drops_to_basket]="1630 1746 1783 1798" [ring_insert]="7 33 167 195" [tool_insert]="683 697 852 883" )
for task in block_stack eye_drops_to_basket ring_insert tool_insert; do
  for ep in ${EPS[$task]}; do
    echo "== DRAKE $task $ep"; nice -n 19 $DK drake_replay.py $task $ep --out results 2>&1 | tail -1
  done
done
$DK compare.py mujoco_results results results/equiv_summary.json
echo BATCH3_DONE
