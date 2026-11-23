#!/bin/bash
set -u
cd /home/openclaw/muse-work/zk0-study001-twin-v2-20261122/controls_battery
python3 - <<'PY'
import battery, json
units=[json.loads(l) for l in open('worklist.jsonl')]
pick=[u for u in units if u['layer']=='L1' and u['checkpoint_id']=='base' and u['task']=='block_stack' and u['episode']==411 and u['variant']=='0'][0]
pick2=[u for u in units if u['layer']=='L2' and u['checkpoint_id']=='base' and u['corner']=='friction_low' and u['episode']==499][0]
pick3=[u for u in units if u['layer']=='L3' and u['checkpoint_id']=='base' and u['episode']==499][0]
for u in (pick,pick2,pick3):
    rc=battery.run_unit(u); print('SMOKE unit', u['layer'], u['task'], u['episode'], u['corner'], 'rc', rc, 'recorded', battery.recorded(u), flush=True)
print('SMOKE_DONE')
PY
