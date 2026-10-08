#!/usr/bin/env python
"""env_synth - FAST synthetic multi-tile envelope-vs-dense diagnostic (NO guard, NO projection).
Isolates the production envelope bug from real-data confounds. Multi-tile (2x2), 32 gaussians,
4 groups. Compares fused dense kernel, envelope fast path, and trusted comax_scan oracle.
Run directly (debug, non-authoritative): python env_synth.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/comax0_mx/package/COMAX0_MX_DSH/scripts")
import torch
PROD = Path("/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_CUDA_PROD")
for pp in (str(PROD/"src"), str(PROD/"logs"/"torch_ext"/"comax_prod")):
    sys.path.insert(0, pp)
from comax_gs.comax_render import comax_scan
import comax_prod_ext as cpe
import comax_envelope_ext as cee

DEV="cuda"; F64=torch.float64; TILE=16
torch.manual_seed(0)

def build(rotated):
    N=32; W=H=32; tw=th=2
    means2d=(torch.rand(N,2,device=DEV))*torch.tensor([W,H],device=DEV)
    if rotated:
        ang=torch.rand(N,device=DEV)*3.14159
        cxx=0.02+0.02*torch.rand(N,device=DEV); cyy=0.02+0.02*torch.rand(N,device=DEV)
        cxy=(0.5*torch.min(cxx,cyy))*torch.sin(2*ang)
    else:
        cxx=0.03+0.02*torch.rand(N,device=DEV); cyy=0.03+0.02*torch.rand(N,device=DEV); cxy=torch.zeros(N,device=DEV)
    conic=torch.stack([cxx,cxy,cyy],dim=1).contiguous()
    opacity=(torch.rand(N,device=DEV)*0.7+0.2).contiguous()
    color=torch.rand(N,3,device=DEV).contiguous()
    depth=torch.rand(N,device=DEV).contiguous()
    groups=(torch.arange(N,device=DEV)%4).to(torch.int32).contiguous()
    return means2d,conic,opacity,color,depth,groups,W,H,tw,th

def ref_scan(means2d,conic,opacity,color,groups,W,H):
    ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    px=xs.reshape(-1).to(F64); py=ys.reshape(-1).to(F64)
    mx=means2d[:,0].to(F64); my=means2d[:,1].to(F64)
    cxx=conic[:,0].to(F64); cxy=conic[:,1].to(F64); cyy=conic[:,2].to(F64)
    op=opacity.to(F64)
    dx=px[None,:]-mx[:,None]; dy=py[None,:]-my[:,None]
    power=cxx[:,None]*dx*dx+2*cxy[:,None]*dx*dy+cyy[:,None]*dy*dy
    alpha=(op[:,None]*torch.exp(-0.5*power)).clamp(0,1)
    rgb,T,_=comax_scan(alpha,color.to(F64),groups.to(F64).to(torch.long),torch.zeros(3,device=DEV,dtype=F64),eta=1.0)
    return rgb.reshape(H,W,3)

def run(rotated):
    means2d,conic,opacity,color,depth,groups,W,H,tw,th=build(rotated)
    N=means2d.shape[0]
    rt=3.0
    tile_lists=[[] for _ in range(tw*th)]
    mxy=means2d.to(torch.float64); dpt=depth.to(torch.float64)
    for gid in range(N):
        mx=mxy[gid,0].item(); my=mxy[gid,1].item(); d=dpt[gid].item()
        tx0=max(0,int((mx-rt*TILE)//TILE)); tx1=min(tw-1,int((mx+rt*TILE)//TILE))
        ty0=max(0,int((my-rt*TILE)//TILE)); ty1=min(th-1,int((my+rt*TILE)//TILE))
        for ty in range(ty0,ty1+1):
            for tx in range(tx0,tx1+1):
                tile_lists[ty*tw+tx].append((d,gid))
    ids=[]; offs=[0]
    for t in range(tw*th):
        for d,gid in sorted(tile_lists[t]):
            ids.append(gid)
        offs.append(len(ids))
    tile_ids=torch.tensor(ids,dtype=torch.int32,device=DEV)
    tile_offsets=torch.tensor(offs,dtype=torch.int32,device=DEV)
    bg=torch.zeros(3,device=DEV)
    d_rgb,d_a,d_ctr=cpe.comax_render_dense(means2d,conic,opacity,color,groups,tile_ids,tile_offsets,W,H,tw,th,background=bg)
    bundle=cee.comax_build_envelope(means2d,conic,opacity,color,depth,groups,tile_ids,tile_offsets,tw*th,tw,th,W,H,background=bg)
    e_rgb,e_a,e_ctr=cee.comax_render_envelope(bundle)
    ref=ref_scan(means2d,conic,opacity,color,groups,W,H)
    def ap(x): return (x.to(F64).reshape(-1,3)-ref.to(F64).reshape(-1,3)).abs()
    d=ap(d_rgb); e=ap(e_rgb); ed=(e_rgb.to(F64).reshape(-1,3)-d_rgb.to(F64).reshape(-1,3)).abs()
    print("rotated=%s  dense_vs_ref max=%.3e p99=%.3e | env_vs_ref max=%.3e p99=%.3e | env_vs_dense max=%.3e p99=%.3e" % (
        rotated, float(d.max()), float(torch.quantile(d,0.99)),
        float(e.max()), float(torch.quantile(e,0.99)),
        float(ed.max()), float(torch.quantile(ed,0.99))))
    print("   dense_ctr",d_ctr.tolist(),"env_ctr",e_ctr.tolist())

print("=== env_synth (multi-tile 2x2, 32 gaussians, 4 groups) ===")
run(False)
run(True)
