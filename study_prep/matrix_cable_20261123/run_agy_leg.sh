#!/bin/bash
# Gemini (agy) leg for the cable judging layer — run AFTER the agy quota resets.
# Batches of 8 items, round-robin models: gemini-4-neon-xhigh -> gemini-4-carbon-high -> claude-opus-5-5-high.
# Blind judging: prompts carry blind codes + frame paths ONLY (no checkpoint names).
# Usage: bash run_agy_leg.sh <batch_number_starting_at_1>  (prints the agy command; review then run with RUN=1)
set -u
WL=/home/openclaw/muse-work/zk0-study001-matrix-20261123/judging_worklist_final.jsonl
FR=/home/openclaw/muse-work/zk0-study001-matrix-20261123/judging_frames
MODELS=(gemini-4-neon-xhigh gemini-4-carbon-high claude-opus-5-5-high)
BATCH=${1:-1}
MODEL=${MODELS[$(( (BATCH-1) % 3 ))]}
START=$(( (BATCH-1)*8 + 1 )); END=$(( BATCH*8 ))
ITEMS=$(python3 - "$WL" "$FR" "$START" "$END" <<'PY'
import json,sys
wl=[json.loads(l) for l in open(sys.argv[1])]
fr=sys.argv[2]; s=int(sys.argv[3]); e=int(sys.argv[4])
for it in wl[s-1:e]:
    d=f"{fr}/{it['worklist_id']}_{it['blind_code']}"
    print(f"{it['worklist_id']}: {d}/end.jpg and {d}/late.jpg and {d}/start.jpg")
PY
)
PROMPT="You are a blind visual judge. Use your file/image viewing tools to open the images listed for each item. Each image shows robot rollout frames (left panel: side view, right panel: top view); start.jpg is the scene setup, late.jpg is ~3s before the end, end.jpg is the final settled frame. CRITERION (cable_clip): PASS = at the end of the rollout the DisplayPort cable is seated/held in the cable holder (clip) on the white block - the cable end segment sits in the clip channel on top of the white block and is retained by the clip at settle. Anything else (cable on the mat, beside the block, held only by the gripper away from the clip, dropped) = FAIL. If the evidence is genuinely insufficient to tell, INDETERMINATE - never guess. Items: $ITEMS . Reply with exactly one line per item: <id> <PASS|FAIL|INDETERMINATE> <one-clause reason>. No other text."
echo "Batch $BATCH items $START-$END model $MODEL"
if [ "${RUN:-0}" = "1" ]; then
  timeout 280 /home/openclaw/.local/bin/agy --model "$MODEL" -p "$PROMPT"
else
  echo "(dry run) agy --model $MODEL -p <prompt with items:> $ITEMS"
fi
