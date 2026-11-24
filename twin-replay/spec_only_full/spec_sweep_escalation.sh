#!/bin/bash
# Fair-shot escalation for one task (founder rule 2026-11-23 13:11 CT;
# budget in attempt-budget-and-rules-20261123.md).
# Route B: scenes unproven after Route A -> best controller x k=1..4.
# Route C: scenes still unproven -> best controller x pose corners
# k=5..8 (precision tasks only; cable has no frozen corners).
# Resumable: existing result files are skipped.
# Usage: spec_sweep_escalation.sh <task>   (launch under job_guard.sh)
set -u
TASK=$1
BASE=$HOME/muse-work/zk0-spec-only-20261123
cd $BASE/work/reference
export PYTHONPATH=$BASE/pylibs
export SPEC_MANIFEST=$BASE/work/spec_only_full/suite_manifest.json
export SPEC_STAGED=$BASE/work/spec_only_full/staged
RES=$BASE/results
BEST=$(TASK=$TASK python3 - <<EOF
import json, glob
from collections import Counter
p = Counter()
for f in glob.glob("$RES/${TASK}__spec*.json"):
    if "_k" in f: continue
    r = json.load(open(f))
    ver = "frozen" if r["controller"].startswith("frozen") else "rev21" if r["controller"].startswith("rev2.1") else "rev3"
    if r["sim_success"]: p[ver] += 1
import os
if p:
    print(p.most_common(1)[0][0])
elif os.environ.get("TASK") == "ring_insert":
    print("rev3")  # ring specialist controller is the escalation route even at 0 passes
else:
    print("rev21")
EOF
)
echo "escalation task=$TASK best_controller=$BEST"
runv() {
  local idx=$1 k=$2
  local out=$RES/${TASK}__spec$(printf %02d $idx)_${BEST}_k${k}.json
  [ -f "$out" ] && return 0
  SPEC_VARIANT=$k timeout 1500 nice -n 10 python3 run_reference_spec.py $BEST $TASK $idx $out \
    > $RES/log_${TASK}_$(printf %02d $idx)_${BEST}_k${k}.txt 2>&1
  echo "esc $TASK idx=$idx k=$k rc=$?"
}
export -f runv
export BEST RES TASK
unproven() {
python3 - <<EOF
import json, glob
passed = set()
for f in glob.glob("$RES/${TASK}__spec*.json"):
    r = json.load(open(f))
    if r["sim_success"]: passed.add(r["scene_index"])
print(" ".join(str(i) for i in range(20) if i not in passed))
EOF
}
U1=$(unproven); echo "unproven after Route A: $U1"
for i in $U1; do for k in 1 2 3 4; do echo "$i $k"; done; done \
  | xargs -P 3 -L 1 bash -c 'runv "$@"' _
U2=$(unproven); echo "unproven after Route B: $U2"
if [ "$TASK" != "cable_clip" ]; then
  for i in $U2; do for k in 5 6 7 8; do echo "$i $k"; done; done \
    | xargs -P 3 -L 1 bash -c 'runv "$@"' _
fi
touch $RES/escalation_done_${TASK}.marker
echo "ESCALATION DONE $TASK"
