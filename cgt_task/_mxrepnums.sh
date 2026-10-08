PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
$PY - <<'PYEOF'
import json
base="/home/liaoyuanjun/cgt_realscene/final"
g1=json.load(open(base+"/G1_REAL_SCENE_COMPRESSIBILITY.json"))
print("=== G1 combos (ordered) ===")
order=["room|/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/iter_5000.pt",
"room|/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/iter_10000.pt",
"room|/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/iter_15000.pt",
"room|/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/iter_30000.pt",
"garden|/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_garden_30k/checkpoints/iter_30000.pt",
"bicycle|/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_bicycle_30k/checkpoints/iter_30000.pt"]
for k in order:
    c=g1["combos"][k]
    print(k.split("|")[0],k.split("iter_")[-1].replace(".pt",""),"q4=%.3f q8=%.3f q16=%.3f incomp=%.4f S2=%.1f S4=%.1f"%(c["q_weighted_cumulative"]["4"],c["q_weighted_cumulative"]["8"],c["q_weighted_cumulative"]["16"],c["incompressible_work_frac"],c["S_inference_actual"]["2.0"],c["S_inference_actual"]["4.0"]))
print("=== G4 batching curve R4 ===")
g4=json.load(open(base+"/G4_BATCHING_CEILING.json"))
for pt in g4["student_batching_curve_R4"]:
    print("B=%d stu/tile=%.5f S_batch=%.2f"%(pt["B"],pt["student_per_tile_ms"],pt["S_batch"]))
print("B1_to_B256 reduction:",g4["batching_reduction_factor_B1_to_B256"])
print("=== dense worst_tiles ===")
d=json.load(open(base+"/G2_CONTINUOUS_DOMAIN_CERTIFICATE.json"))
for w in d["worst_tiles"][:5]: print(w)
print("=== rigorous worst ===")
r=json.load(open(base+"/G2_RIGOROUS_CERTIFICATE.json"))
for w in r["worst_tiles"][:5]: print(w)
PYEOF
