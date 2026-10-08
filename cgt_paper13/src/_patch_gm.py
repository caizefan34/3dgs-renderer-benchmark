import json
from pathlib import Path
HOME=Path("/mnt/storage_pool/liaoyuanjun")
f=HOME/"cgt_paper13/results/PAPER13_SOURCE_AUDIT.json"
d=json.loads(f.read_text())
F0={"bicycle":"B1_bicycle_30k","garden":"B1_garden_30k","room":"B1_room_multiphase_v2"}
TARGET="b8cb0e251bfb0822291cf788911c2ca38d5977fea591f54f1c740052e8fc5f5f"
for s,sub in F0.items():
    pj=HOME/"sota_rebase_v2_runs"/sub/"provenance.json"
    pv=json.loads(pj.read_text())
    gm=pv.get("gaussian_model_hash") or pv.get("gaussian_model_sha256")
    d["per_scene"][s]["provenance"]["gaussian_model_hash"]=gm
    d["per_scene"][s]["provenance"]["reference_v1_gm_hash_match"]=(gm==TARGET)
    print(f"{s}: gm={str(gm)[:12]} match={gm==TARGET} trainer={pv.get('training_script_path','')[-40:]} trainer_hash={pv.get('training_script_sha256','')[:12]}")
f.write_text(json.dumps(d,indent=2,default=str))
print("PATCHED. F0 gm_match now:",[d['per_scene'][s]['provenance']['reference_v1_gm_hash_match'] for s in F0])