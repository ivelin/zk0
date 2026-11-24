#!/bin/bash
# Spec-only solvability sweep for one task (twin-suite v2, Amendment 3).
# Runs BOTH controller versions over scenes 0..19 (rev2.1 first - the
# stronger controller - then frozen), 3 workers, per-run timeout 1500 s.
# Resumable: scenes with an existing result file are skipped.
# Usage: spec_sweep.sh <task>   (launch under job_guard.sh)
set -u
TASK=$1
BASE=$HOME/muse-work/zk0-spec-only-20261123
cd $BASE/work/reference
export PYTHONPATH=$BASE/pylibs
export SPEC_MANIFEST=$BASE/work/spec_only_full/suite_manifest.json
export SPEC_STAGED=$BASE/work/spec_only_full/staged
RES=$BASE/results
mkdir -p $RES
run_one() {
  local ver=$1 task=$2 idx=$3
  local out=$RES/${task}__spec$(printf %02d $idx)_${ver}.json
  [ -f "$out" ] && return 0
  timeout 1500 nice -n 10 python3 run_reference_spec.py $ver $task $idx $out \
    > $RES/log_${task}_$(printf %02d $idx)_${ver}.txt 2>&1
  echo "done $ver $task $idx rc=$?"
}
export -f run_one
export RES
{ for i in $(seq 0 19); do echo "rev21 $TASK $i"; done
  for i in $(seq 0 19); do echo "frozen $TASK $i"; done
  if [ "$TASK" = "ring_insert" ]; then
    for i in $(seq 0 19); do echo "rev3 $TASK $i"; done
  fi } \
  | xargs -P 3 -L 1 bash -c 'run_one "$@"' _
echo "SWEEP DONE $TASK"
