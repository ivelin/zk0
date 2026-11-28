#!/usr/bin/env bash
# Q-A RESUME SWITCH — serial three-arm chain, ARTIFACT-GATED (SSOT S1.3).
# Founder directive 2026-11-26 ~11:48-11:57 CT + Audits A/B (CONFIRMED):
# finish the ORIGINAL experiment by resuming, preserving banked work.
#   1. FL RESUME segment: registered rounds 12-30 from Nov 25
#      checkpoint_round_11, round_offset=11, num-server-rounds=19,
#      initial_model_path QUOTED. State root is THIS segment's own
#      ($W3/fl_state), pre-populated with COPIES of the archived
#      post-round-12 client optimizer states (DECISIONS #11 registered
#      deviation: one-round AdamW moment lead on the round-12 re-fit).
#      Gate: combined checker over BOTH segments (rounds 1-11 in the
#      Nov 25 output dir + 12-30 here), artifacts only.
#   2. Centralized pooled: resume 007600 -> 15,000 steps (run_train.py).
#   3. Specialists x5: 3,000 steps each, full fine-tune (run_train.py).
# Stage transitions are gated on ARTIFACTS ONLY. Process exit codes are
# logged, never gated on (flwr exits 0 on failure — 2026-11-25). Any gate
# failure aborts the chain loudly: chain_failed.json + nonzero exit, no
# downstream arm starts. Launched via launch_dgx_job.sh only.
set -uo pipefail
W3=/home/openclaw/muse-work/zk0-qa-resume-20261126
W1=/home/openclaw/muse-work/zk0-qa-relaunch-20261125
R=$W1/repo
SEG1=$R/outputs/2026-11-25_15-16-28
CKPT11=$SEG1/models/checkpoint_round_11
FLV=/home/openclaw/muse-work/zk0-study001-fl-20261121/venv/bin
LOG=$W3/logs/chain_log.txt
mkdir -p "$W3/logs" "$W3/runs" "$W3/runs/fl"
exec >> "$LOG" 2>&1
echo "=== chain3 (RESUME) start $(date -Is) ==="

abort() { # stage, reason
  "$FLV/python" - "$W3" "$1" "$2" <<'PYEOF'
import json, sys, datetime
w, stage, reason = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump({"stage": stage, "reason": reason,
           "at": datetime.datetime.now().astimezone().isoformat()},
          open(w + "/chain_failed.json", "w"), indent=2)
PYEOF
  echo "CHAIN ABORT stage=$1 reason=$2 $(date -Is)"
  exit 1
}
stage_status() { # stage, status
  "$FLV/python" - "$W3" "$1" "$2" <<'PYEOF'
import json, sys, datetime
w, stage, status = sys.argv[1], sys.argv[2], sys.argv[3]
p = w + "/chain_status.json"
try: d = json.load(open(p))
except Exception: d = {}
d[stage] = {"status": status, "at": datetime.datetime.now().astimezone().isoformat()}
json.dump(d, open(p, "w"), indent=2)
PYEOF
}

cd "$R" || abort "setup" "repo cd failed"
export PATH=$FLV:$PATH

# ROOT-CAUSE FIX 2026-11-26 (gate env-fail): the study venv inherits
# torch/safetensors/lerobot from the bench venv via bench_inherit.pth;
# that venv was deleted mid-run (19:03) and the FL artifact gate then
# ERRORED on imports instead of running. Every stage now preflights the
# exact interpreter it is about to use, before doing any work.
check_env() { # stage
  "$FLV/python" -c "import torch, safetensors, lerobot, num2words" 2>/dev/null \
    || abort "${1:-setup}" "interpreter env broken: $FLV/python cannot import torch/safetensors/lerobot (bench_inherit.pth target zk0-bench-20261117/venv missing?)"
}
check_env setup

# ---- 1. FL resume segment (registered rounds 12-30) ----
stage_status fl_resume running
export ZK0_OPT_STATE_ROOT=$W3/fl_state
# Segment isolation (Audit B condition 2): the segment state root must
# hold COPIES of the archived post-round-12 states; the archive itself
# is never written to. Refuse to launch on any missing state — the
# loader has no round check, so a missing file would silently reset
# that client's carry (DECISIONS #11 / Audit A handling (b) REFUTED).
for i in 0 1 2 3 4; do
  dst=$ZK0_OPT_STATE_ROOT/client_$i/optimizer_state.pt
  if [ ! -f "$dst" ]; then
    mkdir -p "$ZK0_OPT_STATE_ROOT/client_$i"
    cp "$W1/archived-opt-states/client_$i/optimizer_state.pt" "$dst" \
      || abort "fl_resume" "archived state copy failed client_$i"
    echo "state client_$i copied from archive $(date -Is)"
  fi
done
# Short RAY_TMPDIR (AF_UNIX 107-byte socket cap); symlinked into the run
# home so logs stay on-disk + findable (SSOT S2.4.16).
export RAY_TMPDIR=/home/openclaw/rt-qa3/fl
mkdir -p "$RAY_TMPDIR"
ln -sfn "$RAY_TMPDIR" "$W3/runs/fl/ray_tmp"
if [ "${QA_CHAIN_FROM:-fl}" = "centralized" ]; then
  # RESUME-AT-CENTRALIZED (founder 2026-11-26 19:17 CT): FL segment is
  # complete on disk; do not re-run it. Re-verify the combined artifact
  # gate below on the recorded segment dir, then packaging, then proceed.
  check_env fl_resume
  OUT=$(cat "$W3/runs/fl_segment_output_dir.txt")
  [ -d "$OUT" ] || abort "fl_resume" "skip-FL: recorded segment output dir missing: $OUT"
  [ ! -f "$W3/chain_failed.json" ] || mv "$W3/chain_failed.json" "$W3/chain_failed_envfail_20261126.json"
  echo "SKIP-FL (QA_CHAIN_FROM=centralized): re-verifying combined gate on $OUT $(date -Is)"
else
ls "$R/outputs" | sort > "$W3/runs/fl_outputs.before"
"$FLV/flwr" run . local-simulation-serialized-gpu \
  --run-config "num-server-rounds=19 initial_model_path=\"$CKPT11\" round_offset=11" \
  > "$W3/logs/fl_flwr.log" 2>&1
rcFL=$?
echo "FL resume flwr rc=$rcFL (logged, not gated) $(date -Is)"
D=$(comm -13 "$W3/runs/fl_outputs.before" <(ls "$R/outputs" | sort) | tail -1)
[ -n "$D" ] || abort "fl_resume" "no new output dir appeared"
OUT=$R/outputs/$D
echo "FL resume segment output dir: $OUT"
echo "$OUT" > "$W3/runs/fl_segment_output_dir.txt"
fi
"$FLV/python" "$W3/fl_gate_check_combined.py" "$SEG1" "$OUT" "$W3/fl_state" \
  "$W3/logs/fl_flwr.log" 30 11 \
  > "$W3/runs/fl_gate.json" 2>&1
[ $? -eq 0 ] || abort "fl_resume" "combined artifact gate FAIL (see runs/fl_gate.json)"
# S1.3 item 5: packaging step on a copy of the final checkpoint; the
# packaged copy must verify-load standalone. Raw checkpoint stays pristine.
PKG=$W3/runs/fl/packaged_final
rm -rf "$PKG"; cp -r "$OUT/models/checkpoint_round_30" "$PKG"
"$FLV/python" study_prep/package_final_checkpoint.py "$PKG" > "$W3/logs/fl_package.log" 2>&1
"$FLV/python" study_prep/package_final_checkpoint.py --verify-only "$PKG" \
  > "$W3/logs/fl_package_verify.log" 2>&1
[ $? -eq 0 ] || abort "fl_resume" "packaging verify FAIL (see logs/fl_package_verify.log)"
stage_status fl_resume complete
echo "FL RESUME COMPLETE (combined artifact gate PASS, rounds 1-30) $(date -Is)"

# ---- 2. Centralized resume 7600 -> 15000 ----
check_env centralized
stage_status centralized running
"$FLV/python" "$W1/scripts/run_train.py" --arm centralized --steps 15000 \
  --resume "$W1/runs/centralized/out/checkpoints/007600" \
  --output-dir "$W3/runs/centralized/out" > "$W3/logs/centralized.log" 2>&1
rcC=$?
echo "centralized rc=$rcC (logged) $(date -Is)"
[ -d "$W3/runs/centralized/out/checkpoints/015000/training_state" ] \
  || abort "centralized" "step-15000 full-state checkpoint missing"
stage_status centralized complete
echo "CENTRALIZED COMPLETE $(date -Is)"

# ---- 3. Specialists x5 ----
check_env specialists
for t in block_stack eye_drops_to_basket ring_insert tool_insert cable_clip; do
  stage_status "specialist_$t" running
  "$FLV/python" "$W1/scripts/run_train.py" --arm specialist --task "$t" --steps 3000 \
    --output-dir "$W3/runs/specialist_$t/out" > "$W3/logs/specialist_$t.log" 2>&1
  rcS=$?
  echo "specialist $t rc=$rcS (logged) $(date -Is)"
  [ -d "$W3/runs/specialist_$t/out/checkpoints/003000/training_state" ] \
    || abort "specialist_$t" "step-3000 full-state checkpoint missing"
  stage_status "specialist_$t" complete
  echo "SPECIALIST $t COMPLETE $(date -Is)"
done

"$FLV/python" - "$W3" "$SEG1" "$OUT" <<'PYEOF'
import json, sys, datetime
w, seg1, fl_out = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump({
  "chain": "qa-resume-switch-20261126",
  "completed_at": datetime.datetime.now().astimezone().isoformat(),
  "fl_segment1_output_dir": seg1,
  "fl_resume_segment_output_dir": fl_out,
  "fl_registered_rounds": "1-11 (segment 1) + 12-30 (resume segment, round_offset=11)",
  "fl_deviation": "DECISIONS #11: one-round AdamW moment lead on round-12 re-fit (post-round-12 states kept as-is)",
  "fl_gate": json.load(open(w + "/runs/fl_gate.json")),
  "centralized_final": w + "/runs/centralized/out/checkpoints/015000",
  "specialists_final": {t: w + f"/runs/specialist_{t}/out/checkpoints/003000"
                        for t in ["block_stack", "eye_drops_to_basket",
                                  "ring_insert", "tool_insert", "cable_clip"]},
}, open(w + "/chain_done.json", "w"), indent=2)
PYEOF
stage_status chain complete
echo "=== chain3 (RESUME) complete $(date -Is) ==="
