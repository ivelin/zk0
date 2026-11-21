#!/usr/bin/env bash
# Fix-verification SOAK (runI), 2026-11-21: serialized Flower substrate at
# study scale - 5 supernodes (baseline pool subset), local-epochs=2, batch 16,
# FedProx mu=0.01, bf16 (Stage B fp32 variation reverted), 3 rounds,
# checkpoint_interval=1 (local save every round; hf_repo_id commented out in
# pyproject so no Hub push). Gate: >=52 min clean + >=1 round checkpoint.
# See reproduction protocol, soak pre-run expectation.
set -uo pipefail
cd /home/openclaw/muse-work/zk0-repro-stageA-20261121
export JOB_GUARD_MIRROR_TARGET=/home/openclaw/muse-work/zk0-repro-stageA-20261121/mirrorI
export JOB_GUARD_MIRROR_EVERY_SEC=60
exec /home/openclaw/muse-work/tools/launch_dgx_job.sh /home/openclaw/muse-work/zk0-repro-stageA-20261121/runI soak-fixverify active -- /usr/bin/env PATH=/home/openclaw/muse-work/zk0-repro-stageA-20261121/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/openclaw/muse-work/zk0-repro-stageA-20261121/venv/bin/flwr run . local-simulation-serialized-gpu --run-config "use-wandb=false"
