#!/usr/bin/env bash
# Study 001 FL arm RE-EXECUTION (D-11) - PRODUCTION RUN, 2026-11-21.
# Serialized Flower substrate, study copy branch
# study001/fl-substrate-20261121 @ d796990. 30 rounds x 100 local
# steps x 5 nodes, batch 32, FedProx mu=0.1 fixed, dynamic weighting,
# D-9 pooled preprocessor, evals off, checkpoint every round,
# hf_repo_id ABSENT (no Hub push). Guard-wrapped (70% ceiling,
# telemetry mirrored). Run manifest:
# ~/muse-work/zk0-study001-20261119/run/fl-substrate-20261121/run_manifest.json
set -uo pipefail
cd /home/openclaw/muse-work/zk0-study001-fl-20261121
export JOB_GUARD_MIRROR_TARGET=/home/openclaw/muse-work/zk0-study001-20261119/run/fl-substrate-20261121/mirror
export JOB_GUARD_MIRROR_EVERY_SEC=60
exec /home/openclaw/muse-work/tools/launch_dgx_job.sh /home/openclaw/muse-work/zk0-study001-20261119/run/fl-substrate-20261121/prod fl-study001-prod active -- /usr/bin/env PATH=/home/openclaw/muse-work/zk0-study001-fl-20261121/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/openclaw/muse-work/zk0-study001-fl-20261121/venv/bin/flwr run . local-simulation-serialized-gpu --run-config "use-wandb=false"
