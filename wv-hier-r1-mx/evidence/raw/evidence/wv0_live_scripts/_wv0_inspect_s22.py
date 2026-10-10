import torch, json, os, glob
base="/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1"
paths=[]
for scene in ("garden","bicycle"):
    for arm in ("baseline","c42"):
        d=f"{base}/s22/{scene}/checkpoints"
        for p in sorted(glob.glob(d+f"/*{arm}*.pt")):
            paths.append((f"s22_{scene}_{arm}",p))
def scalar(v):
    if isinstance(v,torch.Tensor): return int(v.item()) if v.numel()==1 else f"t{tuple(v.shape)}"
    return v
out={}
for k,p in paths:
    o=torch.load(p,map_location="cpu",weights_only=False)
    xyz=o.get("xyz")
    N=int(o.get("num_points", xyz.shape[0] if xyz is not None else 0))
    out[k]={"N":N,"active_sh":scalar(o.get("active_sh_degree")),"bytes":os.path.getsize(p)}
# also check sidecar json near these checkpoints
for scene in ("garden","bicycle","room"):
    for p in glob.glob(f"{base}/s22/{scene}/*.json")+glob.glob(f"{base}/s22/{scene}/*metrics*"):
        out.setdefault("_sidecars",[]).append(p)
print(json.dumps(out,indent=2))
