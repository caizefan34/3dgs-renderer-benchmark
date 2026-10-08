PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
base="/home/liaoyuanjun/cgt_realscene/final"
print("=== G2 dense aggregate pooled ===")
d=json.load(open(base+"/G2_CONTINUOUS_DOMAIN_CERTIFICATE.json"))
print("pooled:",json.dumps(d["pooled_all_scenes"],indent=1))
print("monotonicity:",json.dumps(d["monotonicity"],indent=1))
print("=== G2 rigorous aggregate pooled ===")
r=json.load(open(base+"/G2_RIGOROUS_CERTIFICATE.json"))
print("pooled:",json.dumps(r["pooled_all_scenes"],indent=1))
print("=== G3 aggregate ===")
g=json.load(open(base+"/G3_GPU_MICROBENCH.json"))
print("agg_intrinsic:",json.dumps(g["aggregate_intrinsic_gamma"],indent=1))
print("agg_per_tile:",json.dumps(g["aggregate_per_tile_gamma"],indent=1))
print("agg_S:",json.dumps(g["aggregate_S_measured_with_h"],indent=1))
print("=== G2 benefit budget key vals ===")
b=json.load(open(base+"/G2_BENEFIT_BUDGET.json"))
print("opt_bound_g1:",b["optimistic_upper_bound_gamma1_min_over_combos"])
print("opt_bound_g4:",b["optimistic_upper_bound_gamma4_min_over_combos"])
print("binding h_max 3x g2:",b["binding_constraint"]["h_max_3x_gamma2_min_over_combos"])
print("binding h_max 3x g4:",b["binding_constraint"]["h_max_3x_gamma4_min_over_combos"])
PYEOF
