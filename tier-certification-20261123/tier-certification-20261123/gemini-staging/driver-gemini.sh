#!/bin/bash
# Gemini (agy) leg driver — sequential, quota-disciplined.
# Usage: driver-gemini.sh <runs-file>   (one run name per line)
# For each run: agy -p "$(cat gemini/prompts/<run>.txt)" with the
# Gemini model; writes gemini/out/<run>.{raw,err,done}. Skips runs
# with an existing .done. Stops the whole leg if a run's output
# mentions quota/limit exhaustion (writes gemini/QUOTA_STOP).
cd /home/openclaw/muse-work/zk0-tier10h-cert-20261123 || exit 1
mkdir -p gemini/out
MODEL="${GEMINI_MODEL:-gemini-3.8-flash-high}"
while IFS= read -r b; do
  [ -z "$b" ] && continue
  [ -f "gemini/out/$b.done" ] && continue
  /home/openclaw/.local/bin/agy -p "$(cat "gemini/prompts/$b.txt")" \
    --model "$MODEL" --dangerously-skip-permissions > "gemini/out/$b.raw" 2> "gemini/out/$b.err"
  echo $? > "gemini/out/$b.done"
  if grep -qiE "quota|rate.?limit|resource.?exhausted|weekly limit" "gemini/out/$b.raw" "gemini/out/$b.err" 2>/dev/null; then
    echo "quota stop after $b" > gemini/QUOTA_STOP
    break
  fi
done < "$1"
touch gemini/GEMINI_DRIVER_DONE
