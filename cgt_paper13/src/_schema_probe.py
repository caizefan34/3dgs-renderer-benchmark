import torch, json
from pathlib import Path
HOME=Path("/mnt/storage_pool/liaoyuanjun")
CKPTS={
 "bicycle":str(HOME/"sota_rebase_v2_runs/B1_bicycle_30k/checkpoints/iter_30000.pt"),
 "garden":str(HOME/"sota_rebase_v2_runs/B1_garden_30k/checkpoints/iter_30000.pt"),
 "room":str(HOME/"sota_rebase_v2_runs/B1_room_multiphase_v2/checkpoints/iter_15000.pt"),
 "bonsai":str(HOME/"c42_13scene_results/mipnerf360/bonsai/reference/checkpoints/iter_30000.pt"),
 "counter":str(HOME/"c42_13scene_results/mipnerf360/counter/reference/checkpoints/iter_30000.pt"),
 "flowers":str(HOME/"c42_13scene_results/mipnerf360/flowers/reference/checkpoints/iter_30000.pt"),
 "kitchen":str(HOME/"c42_13scene_results/mipnerf360/kitchen/reference/checkpoints/iter_30000.pt"),
 "stump":str(HOME/"c42_13scene_results/mipnerf360/stump/reference/checkpoints/iter_30000.pt"),
 "treehill":str(HOME/"c42_13scene_results/mipnerf360/treehill/reference/checkpoints/iter_30000.pt"),
 "train":str(HOME/"c42_13scene_results/tanksandtemples/train/reference/checkpoints/iter_30000.pt"),
 "truck":str(HOME/"c42_13scene_results/tanksandtemples/truck/reference/checkpoints/iter_30000.pt"),
 "drjohnson":str(HOME/"c42_13scene_results/deepblending/drjohnson/reference/checkpoints/iter_30000.pt"),
 "playroom":str(HOME/"c42_13scene_results/deepblending/playroom/reference/checkpoints/iter_30000.pt"),
}
REQ=["xyz","scaling","rotation","opacity","shs","active_sh_degree"]
res={}
for s,p in CKPTS.items():
    try:
        ck=torch.load(p,map_location="cpu",weights_only=False)
        has={k:(k in ck) for k in REQ}
        N=int(ck["xyz"].shape[0]) if "xyz" in ck else None
        shd=int(ck["active_sh_degree"]) if "active_sh_degree" in ck else None
        shshape=list(ck["shs"].shape) if "shs" in ck else None
        allok=all(has.values())
        res[s]={"N":N,"sh_degree":shd,"shs_shape":shshape,"required_keys":has,
                "all_required_present":allok,"top_keys":list(ck.keys())[:25]}
        print(f"{s:10s} N={str(N):8s} shd={str(shd):3s} shs={str(shshape):22s} F0_ingestible={allok}")
        if not allok: print("   MISSING:",[k for k,v in has.items() if not v],"| top:",list(ck.keys())[:25])
    except Exception as e:
        res[s]={"err":str(e)[:200]}
        print(f"{s:10s} ERROR {str(e)[:120]}")
outp=HOME/"cgt_paper13/results/_schema_probe.json"
outp.write_text(json.dumps(res,indent=2,default=str))
n_ok=sum(1 for v in res.values() if v.get("all_required_present"))
print(f"\nF0_INGESTIBLE: {n_ok}/13")