import json
for arm in ["NATIVE","E2D","OPA"]:
    d=json.load(open(f'h0_run/_smoke_room25k_{arm}.json'))
    r=d['rollouts'][0]
    holds=[t['native']['holdout_obj'] for t in r['trajectory']]
    print(f"{arm:7s} K{r['K']} seq={r['update_seq']} holdout_obj轨迹={[round(x,6) for x in holds]}")
