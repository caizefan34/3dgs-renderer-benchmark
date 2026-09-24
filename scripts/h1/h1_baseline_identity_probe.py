import hashlib, json, math, os, sys
import numpy as np
import torch
from plyfile import PlyData

TREE = os.environ["GSPLAT_TREE"]
sys.path.insert(0, TREE)
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles
from gsplat.rendering import rasterization

PLY = "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply"
CAMS = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"
W, H = 2048, 1365

def digest(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def metrics(a, b):
 d=(a.float()-b.float()).abs(); mse=(a.float()-b.float()).square().mean().item()
 return {"max_abs":float(d.max()),"mean_abs":float(d.mean()),"relative_l2":float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)),"psnr":60. if mse<1e-12 else float(-10*np.log10(mse))}
v=PlyData.read(PLY)["vertex"]
def col(n): return torch.tensor(v[n],device="cuda",dtype=torch.float32)
means=torch.stack([col("x"),col("y"),col("z")],-1); q=torch.stack([col("rot_0"),col("rot_1"),col("rot_2"),col("rot_3")],-1); q=q/q.norm(dim=-1,keepdim=True)
s=torch.exp(torch.stack([col("scale_0"),col("scale_1"),col("scale_2")],-1)); o=torch.sigmoid(col("opacity")); sh=torch.zeros(len(v),16,3,device="cuda"); sh[:,0]=torch.stack([col("f_dc_0"),col("f_dc_1"),col("f_dc_2")],-1)
for i in range(15): sh[:,i+1]=torch.stack([col(f"f_rest_{3*i+j}") for j in range(3)],-1)
c=json.load(open(CAMS))[0]; R=np.asarray(c["rotation"]); p=np.asarray(c["position"]); vm=np.eye(4); vm[:3,:3]=R.T; vm[:3,3]=-R.T@p; scale=W/c["width"]; K=np.array([[c["fx"]*scale,0,(W-1)/2],[0,c["fy"]*scale,(H-1)/2],[0,0,1]],np.float32)
vm=torch.tensor(vm,device="cuda",dtype=torch.float32)[None,None]; K=torch.tensor(K,device="cuda")[None,None]
radii,m2d,depths,conics,_=fully_fused_projection(means[None],None,q[None],s[None],vm,K,W,H,0.3,0.01,1e10,0.,False,False,"pinhole")
ob=o[None,None]
def structure(use):
 t,ids,flat=isect_tiles(m2d,radii,depths,16,math.ceil(W/16),math.ceil(H/16),packed=False,n_images=1,conics=conics if use else None,opacities=ob if use else None)
 return {"N_visible":int((radii>0).any(-1).sum()),"N_intersections":int(t.sum()),"tiles_per_gaussian_hash":digest(t),"isect_ids_hash":digest(ids),"flatten_ids_hash":digest(flat),"active_tiles":int(torch.unique((ids>>32)).numel())}
out={"extension":__import__("gsplat").csrc.__file__,"A_original":structure(True),"B_explicit_accutile":structure(True),"C_explicit_clean":structure(False)}
render={}
for name,flag in (("A_original",None),("B_explicit_accutile",True),("C_explicit_clean",False)):
 kw={} if flag is None else {"accutile":flag}; x=rasterization(means[None],q[None],s[None],o[None],sh,vm,K,W,H,sh_degree=3,packed=False,**kw); render[name]=(x[0],x[1])
out["render"]={"A_vs_B_rgb":metrics(render["A_original"][0],render["B_explicit_accutile"][0]),"A_vs_C_rgb":metrics(render["A_original"][0],render["C_explicit_clean"][0]),"A_vs_B_alpha":metrics(render["A_original"][1],render["B_explicit_accutile"][1]),"A_vs_C_alpha":metrics(render["A_original"][1],render["C_explicit_clean"][1])}
print(json.dumps(out,indent=2))
