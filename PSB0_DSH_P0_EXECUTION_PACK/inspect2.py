import json
p="/root/dsh_20261001/artifacts/DSH7_BARS0/BARS0_SUMMARY.json"
d=json.load(open(p))
for k in ["p0","decision","evidence_level","phase_reached","structural_finding","mean_oracle_reduction","per_scene_oracle_reduction","gate_status","claim_scope"]:
    print(k,"=",d.get(k))
print("----- TILE_TRACE -----")
t=json.load(open("/root/dsh_20261001/artifacts/DSH7_BARS0/BARS0_TILE_TRACE.json"))
print("K=",t.get("K"),"scenes=",t.get("scenes"))
p0=t.get("p0")
print("p0 type", type(p0))
if isinstance(p0,dict): print("p0 keys", list(p0.keys())[:30])
