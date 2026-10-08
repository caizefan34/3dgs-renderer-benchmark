#!/bin/bash
cat /root/epic05_p14/artifacts/P14_REPORT.md
echo "===================== VERDICT tail ====================="
/root/miniforge3/envs/gsplat/bin/python -c "
import json
v = json.load(open('/root/epic05_p14/artifacts/P14_VERDICT.json'))
print('O2 chain:', json.dumps(v['gate_chain'].get('O2'), indent=1))
print('stop_reason:', v['stop_reason'])
print('verdict:', v['verdict'])
c = json.load(open('/root/epic05_p14/artifacts/P14_CONTAMINATION.json'))
print('freeze violations:', c['freeze_semantics']['freeze_violations'],
      '| n_F_arms:', c['freeze_semantics']['n_F_arms'],
      '| nonfinite:', c['freeze_semantics']['nonfinite_arms'])
m = json.load(open('/root/epic05_p14/artifacts/MANIFEST_P14.json'))
print('manifest artifacts:', len(m['artifacts']), '| scripts:', len(m['scripts']))
"
