#!/bin/bash
set -u
cd /home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake_actuated
DK=/home/openclaw/muse-work/zk0-sim-eval2-20261115/drake/venv/bin/python3
CKPT=/home/openclaw/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
echo "== RENDER PROOF (fixed camera + visuals, ep411 frame0 + frame130)"
DK_CONTACT=hydro nice -n 19 $DK render_proof.py 2>&1 | tail -n 20
echo "== RE-SMOKE base ep411 nominal (fixed renders, fresh out dir)"
DK_CONTACT=hydro nice -n 19 $DK sim_runner_drake.py --task block_stack --episode 411 \
  --variant nominal --checkpoint-id smolvla_base@5e8d12a6 \
  --checkpoint-path $CKPT --policy service --out smoke/base_drake_fixed --save-frames 2>&1 | tail -n 6
echo BATCH_FIX_DONE
