#!/bin/bash
set -u
cd /home/openclaw/muse-work/zk0-study001-twin-v2-20261122/drake_actuated
DK=/home/openclaw/muse-work/zk0-sim-eval2-20261115/drake/venv/bin/python3
MJ=/home/openclaw/muse-work/zk0-sim-v01-20261115/.venv/bin/python3
CKPT=/home/openclaw/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
echo "== DRAKE PROBES (hydro)"
DK_CONTACT=hydro nice -n 19 $DK grasp_probes_drake.py results/probe_results_hydro.json 2>&1 | tail -n 12
echo "== MUJOCO FINGERPAP PROBES (re-run, padded model, xoff -0.020)"
nice -n 19 $MJ ../harness/fingerpad_probes.py 2>&1 | tail -n 6
echo "== DRAKE SMOKE base ep411 nominal"
DK_CONTACT=hydro nice -n 19 $DK sim_runner_drake.py --task block_stack --episode 411 \
  --variant nominal --checkpoint-id smolvla_base@5e8d12a6 \
  --checkpoint-path $CKPT --policy service --out smoke/base_drake --save-frames 2>&1 | tail -n 6
echo BATCH_DONE
