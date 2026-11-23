#!/bin/bash
# Launch wrapper for the Study 001 fair specialist baselines
# (founder decision 2026-11-22). Guard discipline mirrors the sequential
# chain launch: outer job_guard via launch_dgx_job.sh with JOB_LOCK=0
# (the driver takes the fleet gpu-large-model lock per block — exactly
# one lock layer), mirror target set (guard hard gate).
export JOB_GUARD_MIRROR_TARGET=/home/openclaw/muse-work/zk0-study001-20261119/run/train/spec_exp_mirror
export JOB_LOCK=0
exec /home/openclaw/muse-work/tools/launch_dgx_job.sh \
  /home/openclaw/muse-work/zk0-study001-20261119/run/train/spec_exp_guard2 \
  spec_exp_chain active -- \
  bash /home/openclaw/muse-work/zk0-study001-fl-20261121/study_prep/spec_exp_driver.sh
