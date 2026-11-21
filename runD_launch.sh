#!/usr/bin/env bash
# Stage A re-run (runD), 2026-11-21: identical scale/launch to runC, from the
# committed branch tree (repro/fl-baseline-dgx-20261121 @ 217c8a7).
set -uo pipefail
cd /home/openclaw/muse-work/zk0-repro-stageA-20261121
export JOB_GUARD_MIRROR_TARGET=/home/openclaw/muse-work/zk0-repro-stageA-20261121/mirrorD
export JOB_GUARD_MIRROR_EVERY_SEC=60
exec /home/openclaw/muse-work/tools/launch_dgx_job.sh /home/openclaw/muse-work/zk0-repro-stageA-20261121/runD stageA-repro active -- /usr/bin/env PATH=/home/openclaw/muse-work/zk0-repro-stageA-20261121/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/openclaw/muse-work/zk0-repro-stageA-20261121/venv/bin/flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=1 local-epochs=1 eval_batches=1 use-wandb=false checkpoint_interval=2"
