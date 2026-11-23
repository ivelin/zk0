#!/bin/bash
# Study 001 FAIR SPECIALIST baselines (founder decision 2026-11-22,
# study note study001-specialists-scope-decision-20261122.md): the five
# specialists retrained EXPERT-ONLY (certified 155-tensor / 99,880,992
# action-expert scope, same as C-exp and FL) so every comparison in the
# study is apples-to-apples. The full-FT specialists remain the ceiling
# reference and are NOT touched. Each run is independent, from the
# pinned base snapshot (NOT chained — the sequential arm is separate).
#
# Sequencing (founder constraint): this driver WAITS for the sequential
# chain's "SEQ CHAIN COMPLETE" in sequence.log before any GPU work; if
# the chain fails instead, it aborts without training. Then a miniature
# full-path smoke (8 steps, save at 4/8, fresh-process reload), then the
# five 3,000-step blocks back-to-back.
#
# v2 (2026-11-22): the wait-loop greps are ANCHORED to the chain driver's
# own log-line shape ("] SEQ CHAIN COMPLETE" at end of line). v1 used an
# unanchored grep that matched this driver's own start line (which quoted
# the marker), proceeded instantly, and was killed within ~2 minutes —
# before the smoke could take the fleet lock, so NO training ran and the
# sequential chain was never disturbed. Recorded here per postmortem
# discipline.
#
# Guard/lock discipline identical to the sequential chain launch: the
# ceiling guard is the OUTER launch_dgx_job.sh job_guard (mirror on,
# JOB_LOCK=0 at the wrapper); this driver takes the fleet gpu-large-model
# lock per block via run_guarded. Exactly one lock layer per block.
# Blocks are resumable: re-running skips DONE blocks and resumes an
# interrupted block from its latest complete checkpoint.
W=/home/openclaw/muse-work/zk0-study001-20261119
S=$W/run/scripts
T=$W/run/train
PYB=/home/openclaw/muse-work/zk0-bench-20261117/venv/bin/python
export HF_HOME=/home/openclaw/muse-work/zk0-bench-20261117/hf
LOCK=/home/openclaw/fleet/bin/fleet-lock
mkdir -p $T
log(){ echo "[seq $(date -u +%FT%TZ)] $*" >> $T/sequence.log; echo "[seq] $*"; }
st(){ $PYB $S/seq_status.py "$1" "$2" $3 >> $T/sequence.log 2>&1; }

run_guarded(){ # rundir label cmd...
  local rd=$1 lbl=$2; shift 2
  [ "$1" = "--" ] && shift
  for i in $(seq 1 40); do
    $LOCK Muse gpu-large-model -- "$@"
    rc=$?
    if [ $rc -eq 75 ]; then log "lock busy for $lbl (try $i); waiting 90s"; sleep 90; continue; fi
    return $rc
  done
  return 75
}

latest_ckpt(){ # out_dir -> highest complete step dir (has training_state+pretrained_model)
  python3 - "$1" <<'PY'
import sys
from pathlib import Path
ck = Path(sys.argv[1]) / "checkpoints"
best = None
if ck.is_dir():
    for d in ck.iterdir():
        if d.is_dir() and d.name.isdigit() and (d/"training_state").is_dir() and (d/"pretrained_model").is_dir():
            if best is None or int(d.name) > int(best.name):
                best = d
print(best if best else "")
PY
}

prune_ckpts(){ # out_dir keep_n: drop all but newest keep_n complete ckpts
  python3 - "$1" "$2" <<'PY'
import shutil, sys
from pathlib import Path
ck = Path(sys.argv[1]) / "checkpoints"; keep = int(sys.argv[2])
if not ck.is_dir(): sys.exit(0)
dirs = sorted([d for d in ck.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name))
for d in dirs[:-keep] if len(dirs) > keep else []:
    shutil.rmtree(d)
PY
}

ft_block(){ # name target_steps extra run_train args...
  local name=$1 target=$2; shift 2
  local out=$T/$name/out
  if [ -f $T/$name.done ]; then log "SKIP $name (done)"; return 0; fi
  st $name running
  local attempt rc ck
  for attempt in 1 2 3 4 5 6 7 8; do
    ck=$(latest_ckpt $out)
    local extra=()
    if [ -n "$ck" ]; then
      local step=$((10#$(basename $ck)))
      if [ $step -ge $target ]; then break; fi
      extra=(--resume $ck)
      log "$name attempt $attempt: resume from $ck"
    else
      if [ -d $out/checkpoints ]; then mv $out/checkpoints $out/checkpoints_orphan_$attempt; fi
      log "$name attempt $attempt: fresh start"
    fi
    run_guarded $T/guard_$name $name -- $PYB $S/run_train.py "$@" \
      --steps $target --save-freq 375 --output-dir $out "${extra[@]}"
    rc=$?
    log "$name attempt $attempt rc=$rc"
    prune_ckpts $out 3
    ck=$(latest_ckpt $out)
    if [ -n "$ck" ] && [ $((10#$(basename $ck))) -ge $target ]; then break; fi
    sleep 60
  done
  ck=$(latest_ckpt $out)
  if [ -n "$ck" ] && [ $((10#$(basename $ck))) -ge $target ]; then
    echo "rc=0 final=$ck/pretrained_model $(date -u +%FT%TZ)" > $T/$name.done
    prune_ckpts $out 2
    st $name done "$ck/pretrained_model"
    log "DONE $name -> $ck/pretrained_model"
    return 0
  fi
  st $name FAILED
  log "BLOCK FAILED $name after attempts"
  return 1
}

# ---- Phase 0: wait for the sequential chain -------------------------------
log "SPEC EXP DRIVER START v2: waiting for the sequential chain to complete (founder decision 2026-11-22)"
hb=0
while true; do
  if grep -qE "\] SEQ CHAIN COMPLETE$" $T/sequence.log; then
    log "SPEC EXP: sequential chain complete; proceeding"
    break
  fi
  if grep -qE "\] BLOCK FAILED seq_" $T/sequence.log; then
    log "SPEC EXP ABORT: sequential chain failed; no specialist_exp training launched"
    echo "aborted: sequential chain failed $(date -u +%FT%TZ)" > $T/spec_exp_chain.aborted
    exit 1
  fi
  if ! pgrep -f "seq_chain_driver.sh" > /dev/null; then
    log "SPEC EXP ABORT: sequential chain driver process gone without SEQ CHAIN COMPLETE"
    echo "aborted: chain driver gone $(date -u +%FT%TZ)" > $T/spec_exp_chain.aborted
    exit 1
  fi
  hb=$((hb+1))
  if [ $((hb % 15)) -eq 0 ]; then log "SPEC EXP: still waiting for sequential chain (~$((hb*2)) min)"; fi
  sleep 120
done

# ---- Phase 1: miniature full-path smoke -----------------------------------
# EVERY production code path at tiny scale: train (run_train.py
# --arm specialist_exp, the exact production entry point), checkpoint
# save at 4 and 8, fresh-process reload with the 500-tensor census.
SMOKE_OUT=$W/run/smoke/spec_exp_smoke/out
log "SPEC EXP SMOKE: 8 steps, save-freq 4"
run_guarded $T/guard_spec_exp_smoke spec_exp_smoke -- $PYB $S/run_train.py \
  --arm specialist_exp --task block_stack --steps 8 --save-freq 4 \
  --output-dir $SMOKE_OUT
smoke_rc=$?
log "SPEC EXP SMOKE train rc=$smoke_rc"
if [ $smoke_rc -ne 0 ] || [ ! -d $SMOKE_OUT/checkpoints/000004/pretrained_model ] \
   || [ ! -d $SMOKE_OUT/checkpoints/000008/pretrained_model ]; then
  log "SPEC EXP SMOKE FAILED (train/save path); aborting before production blocks"
  echo "smoke failed $(date -u +%FT%TZ)" > $T/spec_exp_chain.aborted
  exit 1
fi
$PYB - "$SMOKE_OUT/checkpoints/000008/pretrained_model" <<'PY'
import sys
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
p = SmolVLAPolicy.from_pretrained(sys.argv[1])
n = len(p.state_dict())
print(f"SMOKE_RELOAD tensors={n}")
assert n == 500, n
print("SMOKE_RELOAD_OK")
PY
if [ $? -ne 0 ]; then
  log "SPEC EXP SMOKE FAILED (reload path); aborting before production blocks"
  echo "smoke reload failed $(date -u +%FT%TZ)" > $T/spec_exp_chain.aborted
  exit 1
fi
log "SPEC EXP SMOKE PASS (train + save 4/8 + reload 500 tensors)"

# ---- Phase 2: the five fair specialist baselines ---------------------------
log "SPEC EXP CHAIN START (founder decision 2026-11-22: expert-only specialists, from pinned base)"
ft_block spec_exp_block_stack   3000 --arm specialist_exp --task block_stack          || exit 1
ft_block spec_exp_eye_drops     3000 --arm specialist_exp --task eye_drops_to_basket  || exit 1
ft_block spec_exp_ring_insert   3000 --arm specialist_exp --task ring_insert          || exit 1
ft_block spec_exp_tool_insert   3000 --arm specialist_exp --task tool_insert          || exit 1
ft_block spec_exp_cable_clip    3000 --arm specialist_exp --task cable_clip           || exit 1
log "SPEC EXP CHAIN COMPLETE"
echo "complete $(date -u +%FT%TZ)" > $T/spec_exp_chain.done
st spec_exp_chain complete
