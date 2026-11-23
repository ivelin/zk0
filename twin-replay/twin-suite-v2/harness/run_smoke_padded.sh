#!/bin/bash
set -u
cd /home/openclaw/muse-work/zk0-study001-twin-v2-20261122
SIM=/home/openclaw/muse-work/zk0-sim-v01-20261115/.venv/bin/python
CKPT=/home/openclaw/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
for ep in 411 453; do
  for var in nominal 0; do
    nice -n 10 $SIM harness/sim_runner_padded.py --task block_stack --episode $ep       --variant $var --checkpoint-id smolvla_base@5e8d12a6       --checkpoint-path $CKPT --policy service --out smoke/padded_final --save-frames
  done
done
echo SMOKE_PADDED_DONE
