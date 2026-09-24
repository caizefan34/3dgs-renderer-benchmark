#!/usr/bin/env python3
"""R2 targeted, bounded PyTorch CUDA profile of one rasterization training step."""
import argparse, json, sys
from pathlib import Path
import torch, gsplat
from torch.profiler import profile, ProfilerActivity
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--camera',type=int,default=0);a=p.parse_args()
torch.manual_seed(0); W,H=1920,1080
scene=load_ply(str(ROOT/'data/official/mipnerf360/room/point_cloud.ply'),device='cuda')
cams=resize_cameras(load_cameras_from_json(str(ROOT/'data/official/mipnerf360/room/cameras.json'),device='cuda'),W,H);cam=cams[a.camera]
means=scene['xyz'].detach().clone().requires_grad_(True);quats=torch.nn.functional.normalize(scene['rotations'].detach().clone(),dim=-1).requires_grad_(True);scales=scene['scales'].detach().clone().exp().requires_grad_(True);opac=scene['opacity'].detach().clone().requires_grad_(True);shs=scene['shs'].detach().clone().requires_grad_(True);bg=torch.zeros(1,3,device='cuda')
def step():
 for x in (means,quats,scales,opac,shs):
  if x.grad is not None:x.grad=None
 rgb,alpha,_=gsplat.rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=shs,viewmats=cam.world_view_transform[None].contiguous(),Ks=cam.K[None].contiguous(),width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,packed=False,tile_size=16,backgrounds=bg,render_mode='RGB',sparse_grad=False,absgrad=False,rasterize_mode='classic')
 (rgb.float().mean()+alpha.float().mean()).backward()
for _ in range(2): step();torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,profile_memory=True) as prof:
 step();torch.cuda.synchronize()
rows=[]
for e in prof.key_averages():
 if e.device_time_total>0:
  rows.append({'name':e.key,'device_total_us':float(e.device_time_total),'count':int(e.count),'cpu_time_us':float(e.cpu_time_total) if hasattr(e,'cpu_time_total') else 0})
rows.sort(key=lambda x:x['device_total_us'],reverse=True)
out={'schema_version':1,'phase':'R2 forward/backward targeted profile','protocol':{'scene':'room','camera':a.camera,'resolution':'1920x1080','tile_size':16,'warmups':2,'profiled_steps':1},'kernel_rows':rows[:80],'limitations':'One bounded training step; kernel names and timings are profiling evidence, not a proposed execution change.'}
Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(a.out,'w'),indent=2);print('saved',a.out);[print(r) for r in rows[:15]]