#!/bin/bash
# Twin-suite v2 Stage 1.4 smoke: base checkpoint, block_stack ep411+453,
# nominal + variant v0, CPU policy inference. Sequential; the runner's
# JSONL resume makes this kill-safe.
set -u
cd /home/openclaw/muse-work/zk0-study001-twin-v2-20261122
SIM=/home/openclaw/muse-work/zk0-sim-v01-20261115/.venv/bin/python
CKPT=/home/openclaw/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
for ep in 411 453; do
  for var in nominal 0; do
    nice -n 10 $SIM harness/sim_runner.py --task block_stack --episode $ep \
      --variant $var --checkpoint-id smolvla_base@5e8d12a6 \
      --checkpoint-path $CKPT --policy service --out smoke/base_final --save-frames
  done
done
echo SMOKE_DONE
