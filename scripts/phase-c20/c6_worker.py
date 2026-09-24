#!/usr/bin/env python3
"""Per-GPU worker for C20/C6; invoked only by c6_multigpu_schedule.py."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import torch,gsplat
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--camera-ids",type=int,nargs="+",required=True);p.add_argument("--repeats",type=int,default=5);a=p.parse_args();torch.set_grad_enabled(False)
scene=load_ply(str(ROOT/"data/official/mipnerf360/room/point_cloud.ply"),device="cuda");cams=resize_cameras(load_cameras_from_json(str(ROOT/"data/official/mipnerf360/room/cameras.json"),device="cuda"),1920,1080)
means=scene["xyz"].contiguous();q=torch.nn.functional.normalize(scene["rotations"],dim=-1).contiguous();s=scene["scales"].exp().contiguous();o=torch.sigmoid(scene["opacity"]).contiguous();c=scene["shs"].contiguous();bg=torch.zeros(1,3,device="cuda")
def f(cam):return gsplat.rasterization(means=means,quats=q,scales=s,opacities=o,colors=c,viewmats=cam.world_view_transform.unsqueeze(0).contiguous(),Ks=cam.K.unsqueeze(0).contiguous(),width=1920,height=1080,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,packed=False,tile_size=16,backgrounds=bg,render_mode="RGB",sparse_grad=False,absgrad=False,rasterize_mode="classic")
for i in a.camera_ids:f(cams[i]);torch.cuda.synchronize()
t=time.perf_counter()
for _ in range(a.repeats):
 for i in a.camera_ids:f(cams[i])
torch.cuda.synchronize();out={"gpu":torch.cuda.get_device_name(),"camera_ids":a.camera_ids,"repeats":a.repeats,"elapsed_ms":(time.perf_counter()-t)*1000};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out));print(json.dumps(out))
