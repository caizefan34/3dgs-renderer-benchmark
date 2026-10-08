import json
from pathlib import Path
HOME=Path("/mnt/storage_pool/liaoyuanjun")
scenes=["bonsai","counter","flowers","kitchen","stump","treehill"]
tt=["train","truck"]; db=["drjohnson","playroom"]
rows=[]
def load(ds,s):
    p=HOME/"c42_13scene_results"/ds/s/"reference"/"provenance.json"
    return json.loads(p.read_text()) if p.exists() else None
for s in scenes:
    d=load("mipnerf360",s)
    if d: rows.append((s,d.get("git_head"),d.get("absgrad"),d.get("seed"),d.get("iterations"),(d.get("gaussian_model_hash") or "")[:12],d.get("config_base"),d.get("trainer_hash","")[:12]))
for s in tt:
    d=load("tanksandtemples",s)
    if d: rows.append((s,d.get("git_head"),d.get("absgrad"),d.get("seed"),d.get("iterations"),(d.get("gaussian_model_hash") or "")[:12],d.get("config_base"),d.get("trainer_hash","")[:12]))
for s in db:
    d=load("deepblending",s)
    if d: rows.append((s,d.get("git_head"),d.get("absgrad"),d.get("seed"),d.get("iterations"),(d.get("gaussian_model_hash") or "")[:12],d.get("config_base"),d.get("trainer_hash","")[:12]))
print(f"{'scene':10s} {'git_head':12s} absgrad seed iter gm_hash   cfg_base")
for r in rows:
    print(f"{r[0]:10s} {str(r[1])[:12]:12s} {str(r[2]):6s} {str(r[3]):4s} {str(r[4]):4s} {r[5]:11s} {r[6]}")
# consistency check
gh=set(str(r[1]) for r in rows); ab=set(str(r[2]) for r in rows); sd=set(str(r[3]) for r in rows); it=set(str(r[4]) for r in rows); gm=set(str(r[5]) for r in rows); th=set(str(r[7]) for r in rows)
print("\nCONSISTENCY across 10 c42 scenes:")
print("  distinct git_head:",gh)
print("  distinct absgrad:",ab)
print("  distinct seed:",sd)
print("  distinct iterations:",it)
print("  distinct gaussian_model_hash:",gm)
print("  distinct trainer_hash:",th)