"""Validate C17-2 feasibility audit JSON."""
import json

with open('results/phase-a100/c17_2_cross_tile_differential_feasibility.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

required = ["workload_identity", "membership_size", "overlap", "insertions", "removals",
            "order_consistency", "lcs", "insertion_structure", "tile_path_reuse",
            "dependency_chain", "backward_implications", "reconstruction_cost",
            "amortized_cost", "worst_case", "feasibility", "c17_2_status", "reason"]
            
print("=== Validation ===")
print("Status:", d['status'])
assert d['status'] == 'OPTIMIZATION: NOT STARTED', 'BAD STATUS'
print("  OK")

for k in required:
    assert k in d, f'MISSING SECTION: {k}'
print("All required sections present: YES")

print("Feasibility:", d['feasibility'])
assert d['feasibility'] in ('PROMISING', 'CONDITIONAL', 'WEAK', 'DROP'), 'BAD FEASIBILITY'

print("C17-2 status:", d['c17_2_status'])
assert d['c17_2_status'] in ('CONTINUE', 'CONTINUE WITH REDESIGN', 'FREEZE', 'DROP'), 'BAD STATUS'

print("\nKey data sanity checks:")
print("  room H rho_I p50=%.4f (should be 0.15-0.40)" % d['overlap']['room']['h']['rho_I_p50'])
print("  bicycle H rho_I p50=%.4f (should be 0.20-0.50)" % d['overlap']['bicycle']['h']['rho_I_p50'])
print("  room order_preserving=%.0f/%.0f" % (d['order_consistency']['room']['h']['order_preserving_count'], d['order_consistency']['room']['h']['total']))
assert d['order_consistency']['room']['h']['ratio'] == 1.0
assert d['order_consistency']['bicycle']['h']['ratio'] == 1.0
print("  Order consistency: ALL PASS")

print("\n=== ALL VALIDATIONS PASSED ===")
