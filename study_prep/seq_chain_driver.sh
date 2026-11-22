#!/bin/bash
# Study 001 training sequencer (preregistration §6 order):
# specialists x5 -> centralized -> FL -> sequential chain x5.
# Every block: fleet-lock gpu-large-model; the ceiling guard is the OUTER
# launch_dgx_job.sh job_guard (mirror on) for the chain launch (D-11 run). Blocks are resumable:
# re-running this script skips DONE blocks and resumes an interrupted
# block from its latest complete checkpoint.
W=/home/openclaw/muse-work/zk0-study001-20261119
S=$W/run/scripts
T=$W/run/train
PYB=/home/openclaw/muse-work/zk0-bench-20261117/venv/bin/python
PYN=/home/openclaw/muse-work/zk0-node-build-20261117/repo/node/runtime/venv/bin/python
export HF_HOME=/home/openclaw/muse-work/zk0-bench-20261117/hf
LOCK=/home/openclaw/fleet/bin/fleet-lock
GUARD=/home/openclaw/muse-work/zk0-onarm-20261117/ft/scripts/job_guard.sh
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

fl_block(){
  local name=fl
  if [ -f $T/$name.done ]; then log "SKIP fl (done)"; return 0; fi
  st fl running
  local attempt rc
  for attempt in 1 2 3 4 5 6 7 8; do
    run_guarded $T/guard_fl fl -- $PYN $S/fl_driver.py train
    rc=$?
    log "fl attempt $attempt rc=$rc"
    if $PYN -c "
import json
from pathlib import Path
r = Path('$T/fl/rounds/round_30.json')
d = json.loads(r.read_text()) if r.exists() else {}
raise SystemExit(0 if d.get('ckpt_hash_ok') else 1)"; then
      echo "rc=0 final=$T/fl/rounds/round_30/ckpt $(date -u +%FT%TZ)" > $T/fl.done
      st fl done "$T/fl/rounds/round_30/ckpt"
      log "DONE fl -> $T/fl/rounds/round_30/ckpt"
      return 0
    fi
    sleep 60
  done
  st fl FAILED
  log "BLOCK FAILED fl after attempts"
  return 1
}

log "SEQ CHAIN START (D-10/D-11 sequential arm, expert-only, after FL re-execution)"
ft_block seq_block_stack        3000 --arm sequential --task block_stack          || exit 1
ft_block seq_eye_drops          3000 --arm sequential --task eye_drops_to_basket   --init-ckpt $T/seq_block_stack/out/checkpoints/003000/pretrained_model          || exit 1
ft_block seq_ring_insert        3000 --arm sequential --task ring_insert   --init-ckpt $T/seq_eye_drops/out/checkpoints/003000/pretrained_model            || exit 1
ft_block seq_tool_insert        3000 --arm sequential --task tool_insert   --init-ckpt $T/seq_ring_insert/out/checkpoints/003000/pretrained_model          || exit 1
ft_block seq_cable_clip         3000 --arm sequential --task cable_clip   --init-ckpt $T/seq_tool_insert/out/checkpoints/003000/pretrained_model          || exit 1
log "SEQ CHAIN COMPLETE"
st sequence complete
