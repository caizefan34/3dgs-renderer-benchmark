import json, os, glob, hashlib, sys
from pathlib import Path

HOME = Path("/mnt/storage_pool/liaoyuanjun")
SCENES = ["bicycle","bonsai","counter","drjohnson","flowers","garden",
          "kitchen","playroom","room","stump","train","treehill","truck"]
DATASET_OF = {
 "bicycle":"mipnerf360","bonsai":"mipnerf360","counter":"mipnerf360",
 "flowers":"mipnerf360","garden":"mipnerf360","kitchen":"mipnerf360",
 "room":"mipnerf360","stump":"mipnerf360","treehill":"mipnerf360",
 "train":"tanksandtemples","truck":"tanksandtemples",
 "drjohnson":"deepblending","playroom":"deepblending"}

# F0 frozen cohort checkpoints (authoritative)
F0_CKPT = {
 "bicycle": str(HOME/"sota_rebase_v2_runs/B1_bicycle_30k/checkpoints/iter_30000.pt"),
 "garden":  str(HOME/"sota_rebase_v2_runs/B1_garden_30k/checkpoints/iter_30000.pt"),
 "room":    str(HOME/"sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/iter_15000.pt"),
}
# c42 reference (ABSGRAD) checkpoints for the other 10
C42_REF = {
 "bonsai":  str(HOME/"c42_13scene_results/mipnerf360/bonsai/reference/checkpoints/iter_30000.pt"),
 "counter": str(HOME/"c42_13scene_results/mipnerf360/counter/reference/checkpoints/iter_30000.pt"),
 "flowers": str(HOME/"c42_13scene_results/mipnerf360/flowers/reference/checkpoints/iter_30000.pt"),
 "kitchen": str(HOME/"c42_13scene_results/mipnerf360/kitchen/reference/checkpoints/iter_30000.pt"),
 "stump":   str(HOME/"c42_13scene_results/mipnerf360/stump/reference/checkpoints/iter_30000.pt"),
 "treehill":str(HOME/"c42_13scene_results/mipnerf360/treehill/reference/checkpoints/iter_30000.pt"),
 "train":   str(HOME/"c42_13scene_results/tanksandtemples/train/reference/checkpoints/iter_30000.pt"),
 "truck":   str(HOME/"c42_13scene_results/tanksandtemples/truck/reference/checkpoints/iter_30000.pt"),
 "drjohnson":str(HOME/"c42_13scene_results/deepblending/drjohnson/reference/checkpoints/iter_30000.pt"),
 "playroom":str(HOME/"c42_13scene_results/deepblending/playroom/reference/checkpoints/iter_30000.pt"),
}

def sha256(p, n=16):
    try:
        h=hashlib.sha256()
        with open(p,"rb") as f:
            for c in iter(lambda:f.read(1<<20),b""): h.update(c)
        return h.hexdigest()[:n]
    except Exception: return None

def find_data(scene):
    cands=[Path(f"/tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360/{scene}"),
           HOME/f"benchmark_data/mipnerf360/{scene}",
           HOME/f"benchmark_data/tanksandtemples/{scene}",
           HOME/f"benchmark_data/deepblending/{scene}"]
    for c in cands:
        if c.exists(): return c
    for root in [Path("/tmp/3dgs-renderer-benchmark/data/datasets"),HOME/"benchmark_data"]:
        for h in root.rglob(scene):
            if h.is_dir(): return h
    return None

def data_info(d):
    info={"images_count":None,"sparse0":None,"cameras_bin":None}
    if d is None: return info
    imgs=d/"images"
    if imgs.exists(): info["images_count"]=len(list(imgs.iterdir()))
    for sp in [d/"sparse"/"0",d/"sparse",d/"colmap/sparse/0"]:
        if sp.exists():
            info["sparse0"]=str(sp); info["cameras_bin"]=(sp/"cameras.bin").exists(); break
    return info

def load_ckpt_meta(p):
    """Load checkpoint WITHOUT full materialization where possible."""
    import torch
    meta={"keys":None,"N":None,"sh_degree":None,"active_sh":None,"iter":None,"err":None}
    try:
        obj=torch.load(p, map_location="cpu", weights_only=False)
        meta["keys"]=list(obj.keys())[:40] if isinstance(obj,dict) else type(obj).__name__
        # common 3dgs checkpoint layouts
        sd = obj.get("splats", obj.get("gaussians", obj)) if isinstance(obj,dict) else obj
        if isinstance(sd, dict):
            for nk in ["means","means2d","xyz","coords"]:
                if nk in sd:
                    try: meta["N"]=int(sd[nk].shape[0]); break
                    except Exception: pass
            for sk in ["sh0","shN","features","sh_coeffs","colors"]:
                if sk in sd:
                    try: meta["sh_degree"]=int(sd[sk].shape[1]-1); break
                    except Exception: pass
        if isinstance(obj,dict):
            meta["iter"]=obj.get("iter", obj.get("iteration", obj.get("step")))
    except Exception as e:
        meta["err"]=str(e)[:200]
    return meta

# reference gaussian_model hash (schema identity anchor from provenance)
REF_GM_HASH="b8cb0e251bfb0822291cf788911c2ca38d5977fea591f54f1c740052e8fc5f5f"

registry=[]
for s in SCENES:
    d=find_data(s); di=data_info(d)
    f0=F0_CKPT.get(s); c42=C42_REF.get(s)
    entry={"scene":s,"dataset":DATASET_OF[s],
           "data_path":str(d) if d else None,"data_exists":d is not None,
           "images_count":di["images_count"],"cameras_bin":di["cameras_bin"],
           "f0_ckpt":f0,"f0_ckpt_exists":bool(f0 and Path(f0).exists()),
           "c42_ref_ckpt":c42,"c42_ref_exists":bool(c42 and Path(c42).exists())}
    # choose checkpoint to probe: prefer F0, else c42
    probe = f0 if entry["f0_ckpt_exists"] else (c42 if entry["c42_ref_exists"] else None)
    entry["ckpt_source"]="F0" if entry["f0_ckpt_exists"] else ("C42_ABSGRAD" if entry["c42_ref_exists"] else None)
    entry["ckpt_path"]=probe
    if probe:
        entry["ckpt_sha256_16"]=sha256(probe)
        entry["ckpt_size_bytes"]=Path(probe).stat().st_size
        m=load_ckpt_meta(probe)
        entry["ckpt_meta"]=m
    # classification
    if entry["f0_ckpt_exists"]:
        entry["classification"]="EXACT_COHORT_F0"
    elif entry["c42_ref_exists"]:
        entry["classification"]="COMPATIBLE_CANONICAL_ABSGRAD"
    elif entry["data_exists"]:
        entry["classification"]="MISSING_CHECKPOINT_TRAIN_REQUIRED"
    else:
        entry["classification"]="MISSING_DATA"
    registry.append(entry)

# load c42 provenance for cohort comparison
c42prov={}
for s in SCENES:
    c42=C42_REF.get(s)
    if c42:
        pj=Path(c42).parent.parent/"provenance.json"
        if pj.exists():
            try: c42prov[s]=json.loads(pj.read_text())
            except Exception: pass

out={"registry":registry,"c42_provenance_summary":{
        "git_head":c42prov.get("counter",{}).get("git_head"),
        "git_describe":c42prov.get("counter",{}).get("git_describe"),
        "absgrad":c42prov.get("counter",{}).get("absgrad"),
        "seed":c42prov.get("counter",{}).get("seed"),
        "iterations":c42prov.get("counter",{}).get("iterations"),
        "gaussian_model_hash":c42prov.get("counter",{}).get("gaussian_model_hash"),
        "reference_v1_gm_hash_match": c42prov.get("counter",{}).get("gaussian_model_hash")==REF_GM_HASH,
      },
     "summary":{
       "total":len(SCENES),
       "exact_cohort_f0":sum(1 for r in registry if r["classification"]=="EXACT_COHORT_F0"),
       "compatible_canonical_absgrad":sum(1 for r in registry if r["classification"]=="COMPATIBLE_CANONICAL_ABSGRAD"),
       "missing_ckpt_train_required":sum(1 for r in registry if r["classification"]=="MISSING_CHECKPOINT_TRAIN_REQUIRED"),
       "missing_data":sum(1 for r in registry if r["classification"]=="MISSING_DATA"),
     }}
outpath=HOME/"cgt_paper13/results/PAPER13_SCENE_REGISTRY.json"
outpath.write_text(json.dumps(out,indent=2,default=str))
print(json.dumps(out["summary"],indent=2))
print("c42 cohort:",json.dumps(out["c42_provenance_summary"],indent=2))
for r in registry:
    m=r.get("ckpt_meta",{})
    print(f"{r['scene']:10s} {r['dataset']:15s} {r['ckpt_source']:14s} N={str(m.get('N')):8s} sh={str(m.get('sh_degree')):4s} iter={str(m.get('iter')):6s} imgs={str(r['images_count']):5s} {r['classification']}")
    if m.get('err'): print("   ERR:",m['err'])