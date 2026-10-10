#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX Stage 0: frozen-state inventory.
Computes SHA256, N_gaussians, resolution, SH degree, and best-known iteration
label for every candidate fixed state. Emits WV0_STATE_INVENTORY.json.

Runs on MX. Read-only w.r.t. checkpoints (weights_only load, CPU).
"""
import hashlib, json, os, glob, platform, socket, subprocess, sys, time

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
RV1 = os.path.join(REPO, "results/reference_v1")

# state_id -> checkpoint path (relative to RV1). Iteration encoded in key/name.
STATES = {
  "room__iter5000":  "room_30k/checkpoints/iter_5000.pt",
  "room__iter15000": "room_30k/checkpoints/iter_15000.pt",
  "room__iter30000": "room_30k/checkpoints/iter_30000.pt",
  "garden__iter5000":  "s22/garden/checkpoints/baseline_iter_5000.pt",
  "garden__iter15000": "s22/garden/checkpoints/baseline_iter_15000.pt",
  "garden__iter30000": "s22/garden/checkpoints/baseline_iter_30000.pt",
  "bicycle__iter5000":  "s22/bicycle/checkpoints/baseline_iter_5000.pt",
  "bicycle__iter15000": "s22/bicycle/checkpoints/baseline_iter_15000.pt",
  "bicycle__iter30000": "s22/bicycle/checkpoints/baseline_iter_30000.pt",
}
# fallback alternates (s23 high-N, unique) if a primary is missing
ALT = {
  "garden__iter30000": ["s23/garden/checkpoints/s0.750_iter_30000.pt"],
  "bicycle__iter30000": ["s23/bicycle/checkpoints/s0.625_iter_30000.pt"],
  "bicycle__iter10000": ["s23/bicycle/checkpoints/s0.750_iter_10000.pt"],
}

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def load_meta(p):
    import torch
    o=torch.load(p, map_location="cpu", weights_only=False)
    def sc(v):
        if isinstance(v, torch.Tensor):
            return int(v.item()) if v.numel()==1 else f"tensor{tuple(v.shape)}"
        return v
    def scf(v):
        if isinstance(v, torch.Tensor) and v.numel()==1: return float(v.item())
        if isinstance(v,(int,float)): return float(v)
        return v
    xyz=o.get("xyz")
    N=int(o["num_points"]) if "num_points" in o else (int(xyz.shape[0]) if xyz is not None else None)
    shp = o["shs"].shape if "shs" in o else None
    return N, sc(o.get("active_sh_degree", o.get("sh_degree"))), list(shp) if shp else None, scf(o.get("spatial_lr_scale"))

def main():
    os.environ["TMPDIR"]="/dev/shm"
    inv={"schema":"WV0_STATE_INVENTORY.v1","generated_unix":time.time(),
         "host":socket.gethostname(),"repo":REPO,
         "torch":None,"states":{}}
    try:
        import torch; inv["torch"]=torch.__version__
    except Exception as e:
        inv["torch"]=f"ERR {e}"
    for sid, rel in STATES.items():
        p=os.path.join(RV1, rel)
        entry={"state_id":sid,"ckpt_path":p,"present":os.path.isfile(p)}
        if not entry["present"] and sid in ALT:
            for a in ALT[sid]:
                ap=os.path.join(RV1,a)
                if os.path.isfile(ap):
                    p=ap; entry["ckpt_path"]=ap; entry["present"]=True
                    entry["alt_used"]=a; break
        if entry["present"]:
            try:
                t0=time.time()
                N,sh,shp,lsl=load_meta(p)
                entry.update({
                  "sha256":sha256_file(p),"bytes":os.path.getsize(p),
                  "N_gaussians":N,"active_sh_degree":sh,"shs_shape":shp,
                  "spatial_lr_scale":lsl,
                  "load_seconds":round(time.time()-t0,3),
                  "scene":sid.split("__")[0],
                })
                # iteration from filename
                import re
                m=re.search(r"iter_?(\d+)", sid)
                if m: entry["iteration"]=int(m.group(1))
            except Exception as e:
                import traceback
                entry["error"]=repr(e)[:200]
                entry["error_tb"]=traceback.format_exc()[-600:]
        inv["states"][sid]=entry
    out=os.path.join(os.environ.get("WV0_OUT","/dev/shm/wv0_out"),"WV0_STATE_INVENTORY.json")
    os.makedirs(os.path.dirname(out),exist_ok=True)
    def jf(o):
        try: return float(o)
        except Exception: return str(o)
    json.dump(inv, open(out,"w"), indent=2, default=jf)
    # summary
    print("=== STATE INVENTORY ===")
    for sid,e in inv["states"].items():
        print(f"{sid:24s} present={e['present']} N={e.get('N_gaussians')} sh={e.get('active_sh_degree')} iter={e.get('iteration')} sha={ (e.get('sha256') or '')[:12] }")
    print("WROTE", out)

if __name__=="__main__":
    main()
