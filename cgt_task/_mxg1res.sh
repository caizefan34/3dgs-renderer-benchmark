PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
d=json.load(open("/home/liaoyuanjun/cgt_realscene/final/G1_REAL_SCENE_COMPRESSIBILITY.json"))
print("G1_overall_verdict:",d["G1_overall_verdict"])
print("combos:",list(d["combos"].keys()))
for k,c in d["combos"].items():
    print(f"\n[{k}] n_tiles={c['n_tiles']} incomp={c['incompressible_work_frac']:.4f} verdict={c['verdict']}")
    print("  q_weighted(cumulative):"," ".join(f"r{R}={c['q_weighted_cumulative'][str(R)]:.4f}" for R in (4,8,16)))
    print("  S_inference:"," ".join(f"g{g}={c['S_inference_actual'][str(g)]:.2f}" for g in (1.0,2.0,4.0)))
PYEOF
