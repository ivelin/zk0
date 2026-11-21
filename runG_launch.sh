#!/usr/bin/env bash
# Stage C (runG), 2026-11-21: execution-model row of the DGX reproduction.
# Subject: Study 001's in-process fl_driver.py (sha256
# 0eccaaf8f13686354d1c969fa06adcc908811bd6b40f6f76b16fc0105358a5cd, unmodified),
# production node-runtime venv, miniature scale via the driver's own CLI
# count knobs only: 2 nodes (0,1 = block_stack, eye_drops_to_basket),
# 1 round, 60 steps/round, batch 32 (production value), fresh fl-base-dir.
set -uo pipefail
mkdir -p /home/openclaw/muse-work/zk0-repro-stageC-20261121
export JOB_GUARD_MIRROR_TARGET=/home/openclaw/muse-work/zk0-repro-stageC-20261121/mirrorG
export JOB_GUARD_MIRROR_EVERY_SEC=60
exec /home/openclaw/muse-work/tools/launch_dgx_job.sh /home/openclaw/muse-work/zk0-repro-stageC-20261121/runG stageC-repro active -- /home/openclaw/muse-work/zk0-node-build-20261117/repo/node/runtime/venv/bin/python /home/openclaw/muse-work/zk0-study001-20261119/run/scripts/fl_driver.py train --rounds 1 --steps-per-round 60 --batch-size 32 --nodes 0,1 --fl-base-dir /home/openclaw/muse-work/zk0-repro-stageC-20261121/fl
