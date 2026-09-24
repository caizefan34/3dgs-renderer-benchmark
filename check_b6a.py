import json, io
d = json.load(io.open('results/c42_adaptive/b6a_unknown_sensitivity.json', encoding='utf-8'))
a = d['tau_results']['0.020']['assignments']
for x in sorted(a, key=lambda x: x['advantage_ms'] if x['advantage_ms'] is not None else 999)[:5]:
    print(f"  {x['assignment']}: oracle_avg={x['oracle_avg_cost_ms']}, fixed={x['best_fixed_feasible_cost_ms']}, adv={x['advantage_ms']}")
print("\n--- Top 5 max advantage ---")
for x in sorted(a, key=lambda x: x['advantage_ms'] if x['advantage_ms'] is not None else -1, reverse=True)[:5]:
    print(f"  {x['assignment']}: oracle_avg={x['oracle_avg_cost_ms']}, fixed={x['best_fixed_feasible_cost_ms']}, adv={x['advantage_ms']}")
