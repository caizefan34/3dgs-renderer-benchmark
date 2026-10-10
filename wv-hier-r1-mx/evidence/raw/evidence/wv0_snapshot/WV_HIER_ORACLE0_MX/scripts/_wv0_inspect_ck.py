import torch, sys, json, hashlib
paths = {
 "room_5000":  "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_5000.pt",
 "room_15000": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_15000.pt",
 "room_30000": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_30000.pt",
 "room_29970": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_29970.pt",
 "ck14k_5000": "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/ckpt_14k/checkpoints/iter_5000.pt",
 "ck14k_10000":"/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/ckpt_14k/checkpoints/iter_10000.pt",
 "ck14k_14000":"/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/ckpt_14k/checkpoints/iter_14000.pt",
 "c42_5000":   "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/c42_30k/checkpoints/iter_5000.pt",
 "c42_15000":  "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/c42_30k/checkpoints/iter_15000.pt",
 "c42_30000":  "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/c42_30k/checkpoints/iter_30000.pt",
 "s23_garden_30000":"/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s23/garden/checkpoints/s0.750_iter_30000.pt",
 "s23_bicycle_10000":"/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s23/bicycle/checkpoints/s0.750_iter_10000.pt",
 "s23_bicycle_30000":"/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s23/bicycle/checkpoints/s0.625_iter_30000.pt",
}
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def scalar(v):
    if isinstance(v,torch.Tensor):
        return int(v.item()) if v.numel()==1 else f"tensor{tuple(v.shape)}"
    return v
out={}
for k,p in paths.items():
    try:
        o=torch.load(p,map_location="cpu",weights_only=False)
        keys=list(o.keys())
        N=None
        for nk in ("num_points","gaussian_count","n_points","_n_points"):
            if nk in o:
                val=o[nk]; N=int(val.item()) if torch.is_tensor(val) and val.numel()==1 else val; break
        def pick(*names):
            for n in names:
                if n in o: return o[n]
            return None
        xyz=pick("means","xyz","positions")
        if N is None and xyz is not None: N=int(xyz.shape[0])
        out[k]={"scene":scalar(o.get("scene")),"iter":scalar(o.get("iteration",o.get("step"))),
                "N":int(N) if isinstance(N,(int,)) else None,
                "active_sh":scalar(o.get("active_sh_degree",o.get("sh_degree"))),
                "sh_key_shape":list(o["shs"].shape) if "shs" in o else None,
                "keys":keys[:12],"bytes":__import__("os").path.getsize(p)}
    except Exception as e:
        out[k]={"ERR":repr(e)[:200]}
print(json.dumps(out,indent=2))
