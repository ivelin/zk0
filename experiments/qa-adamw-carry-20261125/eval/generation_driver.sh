#!/usr/bin/env bash
# Q-A eval trajectory generation (R-1 order). Sequential, guard-launched, skip-if-done.
set +e
export STUDY_GATE_ALLOW_OVERRIDES=1
export JOB_GUARD_MIRROR_TARGET=/home/openclaw/muse-work/zk0-qa-resume-20261126/eval/mirror
mkdir -p "$JOB_GUARD_MIRROR_TARGET"
E=/home/openclaw/muse-work/zk0-qa-resume-20261126/eval
W3=/home/openclaw/muse-work/zk0-qa-resume-20261126
PY=/home/openclaw/muse-work/zk0-bench-20261117/venv/bin/python
TRAJ=/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/scripts/eval_traj.py
run_ckpt () {
  name=$1; path=$2
  if [ -f "$E/records/summary__${name}.json" ]; then echo "[gen] skip $name (summary exists)"; return; fi
  echo "[gen] === $name $path $(date -u +%FT%TZ)"
  ~/muse-work/tools/launch_dgx_job.sh $E/guard-$name qa-eval-gen active -- $PY $TRAJ --ckpt-name "$name" --ckpt-path "$path" --split $E/split/split.json --out $E/records >> $E/logs/generation.log 2>&1
  echo "[gen] $name exit=$?"
}
run_ckpt fl_qa_r30 $E/packaged/fl_qa_r30
run_ckpt centralized_qa $W3/runs/centralized/out/checkpoints/015000/pretrained_model
run_ckpt spec_block_stack $W3/runs/specialist_block_stack/out/checkpoints/003000/pretrained_model
run_ckpt spec_eye_drops_to_basket $W3/runs/specialist_eye_drops_to_basket/out/checkpoints/003000/pretrained_model
run_ckpt spec_ring_insert $W3/runs/specialist_ring_insert/out/checkpoints/003000/pretrained_model
run_ckpt spec_tool_insert $W3/runs/specialist_tool_insert/out/checkpoints/003000/pretrained_model
run_ckpt spec_cable_clip $W3/runs/specialist_cable_clip/out/checkpoints/003000/pretrained_model
run_ckpt fl_qa_r20 $E/packaged/fl_qa_r20
run_ckpt fl_qa_r10 $E/packaged/fl_qa_r10
echo "[gen] ALL GENERATION DONE $(date -u +%FT%TZ)"
