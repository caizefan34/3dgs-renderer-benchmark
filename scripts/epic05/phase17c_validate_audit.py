"""Validate C17-2 data integrity audit JSON output."""
import json

with open('results/phase-a100/c17_2_data_integrity_cross_tile_audit.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

assert d['status'] == 'OPTIMIZATION: NOT STARTED', 'BAD STATUS'
assert d['c17_2_status'] in ('CONTINUE', 'CONTINUE WITH REDESIGN', 'FREEZE', 'DROP'), 'BAD C17-2 STATUS'

print('=== Validation PASS ===')
print('Status:', d['status'])
print('C17-2 status:', d['c17_2_status'])
print('Reason:', d['reason'][:80])

# Verify N_isect reconciliation all YES
for k, v in d['n_isect_reconciliation'].items():
    assert v['reconciled'], f'{k} not reconciled'
    print(f'  {k}: reconclied={v["reconciled"]} root={v["root_cause"]}')

print()

# Verify arithmetic
for k, v in d['gaussian_count_reconciliation'].items():
    if not isinstance(v, dict) or 'diff' not in v:
        continue
    print(f'  {k}: diff={v["diff"]} {"OK" if v["diff"] == 0 else "BUG!"}')
    assert v['diff'] == 0, f'{k} arithmetic bug!'

print()

# Show cross-tile overlap summary
for k, v in d['cross_tile_overlap']['scenes'].items():
    h = v['horizontal']
    vv = v['vertical']
    print(f'  {k}: H-Jaccard P50={h["jaccard_p50"]:.4f} V-Jaccard P50={vv["jaccard_p50"]:.4f}')

print()
print('All checks passed.')
