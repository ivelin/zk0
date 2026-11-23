#!/bin/bash
set -u
cd /home/openclaw/muse-work/zk0-study001-twin-v2-20261122/controls_battery
python3 - <<'PY'
import battery, json
units=[json.loads(l) for l in open('worklist.jsonl')]
u=[x for x in units if x['layer']=='L2' and x['checkpoint_id']=='base' and x['corner']=='friction_low' and x['episode']==499][0]
rc=battery.run_unit(u); print('SMOKE2', rc, battery.recorded(u), flush=True)
print('SMOKE2_DONE')
PY
