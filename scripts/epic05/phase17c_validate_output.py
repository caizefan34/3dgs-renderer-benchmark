import json

with open('results/phase-a100/ordered_membership_representation_audit.json', encoding='utf-8') as f:
    d = json.load(f)

print('Status:', d['status'])
assert d['status'] == 'OPTIMIZATION: NOT STARTED', 'BAD STATUS'

print('Adjudication:', d['adjudication'])
scenes = list(d['data_collection']['scenes'].keys())
print('Scenes:', scenes)
for s in scenes:
    sc = d['data_collection']['scenes'][s]
    print('  %s: isects=%d flat_B=%d tile_B=%d dup=%d' % (
        s, sc['n_isects'], sc['flatten_ids_bytes'], sc['tile_offsets_bytes'], sc['per_tile_duplicates']))
    print('    delta: ' + sc['delta_locality']['conclusion'])
    print('    id_range: ' + sc['id_range']['conclusion'])

print('Opportunities:', list(d['opportunity_matrix'].keys()))
print('Terminal: ' + d['terminal_conclusion'][:100])
print('JSON VALIDATION PASSED')
