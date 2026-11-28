#!/usr/bin/env bash
# Q-A eval trajectory generation (R-1 order). Sequential, guard-launched, skip-if-done.
set +e
export STUDY_GATE_ALLOW_OVERRIDES=1
# Paths parameterized 2026-11-28 (Phase 1 verification fix); as-run values were
# the build machine's run tree. Guest defaults: E = this script's folder;
# W3 = a local tree mirroring the run layout (or point the run_ckpt paths at
# the HF downloads from zk0bot/smolvla-qa-20261127); PY = the eval venv python.
E=${QA_EVAL_TREE:-$(cd "$(dirname "$0")" && pwd)}
W3=${QA_RUN_HOME:-$HOME/zk0-qa-run}
PY=${QA_EVAL_PYTHON:-python3}
TRAJ=${QA_TRAJ_SCRIPT:-$E/eval_traj.py}
export JOB_GUARD_MIRROR_TARGET=${JOB_GUARD_MIRROR_TARGET:-$E/mirror}
mkdir -p "$JOB_GUARD_MIRROR_TARGET"
run_ckpt () {
  name=$1; path=$2
  if [ -f "$E/records/summary__${name}.json" ]; then echo "[gen] skip $name (summary exists)"; return; fi
  echo "[gen] === $name $path $(date -u +%FT%TZ)"
  # QA_LAUNCH_WRAPPER: the build machine's job-guard launcher (shared-machine
  # safety on our box). Unset = run the generator directly, as a guest would.
  if [ -n "${QA_LAUNCH_WRAPPER:-}" ]; then
    "$QA_LAUNCH_WRAPPER" $E/guard-$name qa-eval-gen active -- $PY $TRAJ --ckpt-name "$name" --ckpt-path "$path" --split ${QA_SPLIT:-$E/split.json} --out $E/records >> $E/logs/generation.log 2>&1
  else
    $PY $TRAJ --ckpt-name "$name" --ckpt-path "$path" --split ${QA_SPLIT:-$E/split.json} --out $E/records >> $E/logs/generation.log 2>&1
  fi
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
