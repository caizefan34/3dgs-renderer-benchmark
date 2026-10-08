import sys, os, json, time
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/eb0_exact_backward_oracle_mx")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat")
import numpy as np, torch
import trace as T
import compositor as CX
import gsplat
torch.manual_seed(0); np.random.seed(0)
BASE="/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360"
scene="room"; NCAM=2; DOWN=4
sc=T.load_ply(f"{BASE}/{scene}/native/point_cloud/iteration_30000/point_cloud.ply")
cams=json.load(open(f"{BASE}/{scene}/c42/cameras.json"))
idxs=T.lock_cameras(cams,NCAM)
vms,Ks,W,H=T.build_views(cams,idxs,DOWN)
C=len(idxs); N=int(sc["n"]); CDIM=3
ct=lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float32).cuda()
t0=time.time()
means=ct(sc["xyz"]); quats=ct(sc["rot"]); scales=ct(sc["scale"])
opac=ct(sc["opacity"])           # [N] ACTIVATED (gsplat 1.5.3 does not sigmoid)
colors=ct(sc["color"])            # [N,3]
viewmats=ct(vms); Ks_t=ct(Ks)
# gsplat rasterization with packed=False -> per-gaussian meta; grad on colors+opac
colors_g=colors.clone().requires_grad_(True)
opac_g=opac.clone().requires_grad_(True)
rc,ra,meta=gsplat.rasterization(means,quats,scales,opac_g,colors_g,viewmats,Ks_t,W,H,
        tile_size=16,packed=False,sh_degree=None)
print("render_colors",tuple(rc.shape),"render_alphas",tuple(ra.shape),"N",N,"C",C,"W",W,"H",H, flush=True)
# upstream grad (same for both paths)
vrc=torch.rand_like(rc); vra=torch.rand_like(ra)   # [C,H,W,3],[C,H,W,1]
loss=(rc*vrc).sum()+(ra*vra).sum()
loss.backward()
g_col_gs=colors_g.grad.detach()       # [N,3] dL/d(color)
g_opa_gs=opac_g.grad.detach()         # [N]   dL/d(ACTIVATED opacity)  [gsplat 1.5.3: no sigmoid]
print("gsplat grads: col",tuple(g_col_gs.shape),"opa",tuple(g_opa_gs.shape),"t=%.1fs"%(time.time()-t0), flush=True)
# feed compositor the EXACT arrays gsplat differentiated
m2d=meta["means2d"].detach()          # [C,N,2]
con=meta["conics"].detach()           # [C,N,3]
opa=meta["opacities"].detach()[0].clone()   # [N] (camera-independent)
col=colors.detach()                   # [N,3]
off=meta["isect_offsets"].detach()    # [C,tile_H,tile_W]
fid=meta["flatten_ids"].detach().reshape(-1).to(torch.int32) % N   # [nnz]
tile_W=int(meta["tile_width"]); tile_H=int(meta["tile_height"])
print("compositor inputs: m2d",tuple(m2d.shape),"con",tuple(con.shape),"opa",tuple(opa.shape),"off",tuple(off.shape),"fid",tuple(fid.shape),"tile",tile_W,tile_H, flush=True)
t1=time.time()
g_m2d,g_con,g_col_cx,g_opa_cx,stats=CX.canon(m2d,con,col,opa,off,fid,tile_W,tile_H,C,W,H,16,vrc,vra,reduction="atomic")
g_col_cx=g_col_cx.detach(); g_opa_cx=g_opa_cx.detach()
print("compositor done t=%.1fs stats=%s"%(time.time()-t1,stats), flush=True)
def cos(a,b):
    a=a.reshape(-1).double(); b=b.reshape(-1).double()
    return float((a@b)/(a.norm()*b.norm()+1e-30))
vis_col=(g_col_gs.abs().sum(1)>0)
vis_opa=(g_opa_gs.abs()>0)
print("=== COLOR grad (dL/dcolor) ===")
print("  visible gaussians:",int(vis_col.sum().item()),"of",N)
print("  cosine ALL:     ",cos(g_col_cx,g_col_gs))
print("  cosine VISIBLE: ",cos(g_col_cx[vis_col],g_col_gs[vis_col]))
print("  maxabserr VISIBLE:",float((g_col_cx[vis_col]-g_col_gs[vis_col]).abs().max()))
print("  rel L2 VISIBLE:   ",float((g_col_cx[vis_col]-g_col_gs[vis_col]).norm()/(g_col_gs[vis_col].norm()+1e-30)))
print("=== OPACITY grad (dL/dACTIVATED opacity) ===")
print("  visible gaussians:",int(vis_opa.sum().item()),"of",N)
print("  cosine ALL:     ",cos(g_opa_cx,g_opa_gs))
print("  cosine VISIBLE: ",cos(g_opa_cx[vis_opa],g_opa_gs[vis_opa]))
print("  maxabserr VISIBLE:",float((g_opa_cx[vis_opa]-g_opa_gs[vis_opa]).abs().max()))
print("  rel L2 VISIBLE:   ",float((g_opa_cx[vis_opa]-g_opa_gs[vis_opa]).norm()/(g_opa_gs[vis_opa].norm()+1e-30)))
# verdict against exactness gate 0.99999
cc=cos(g_col_cx[vis_col],g_col_gs[vis_col]); co=cos(g_opa_cx[vis_opa],g_opa_gs[vis_opa])
print("=== VERDICT ===")
print("  color cos>=0.99999:",cc>=0.99999," opacity cos>=0.99999:",co>=0.99999)
print("  color cos>=0.999:  ",cc>=0.999, " opacity cos>=0.999:  ",co>=0.999)
print("COMPOSITOR_VALIDATION_DONE")