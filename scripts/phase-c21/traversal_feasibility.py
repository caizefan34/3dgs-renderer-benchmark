#!/usr/bin/env python3
"""C21 N1/N3/N7 feasibility: depth-truncated replay (stock gsplat, no kernel mod).

Constructs per-tile sorted intersection ranges at depth fraction thresholds
(0.25, 0.50, 0.75, full) and compares pixel alpha. Establishes feasibility
bounds for opacity-aware truncation without modifying the renderer.

N1: effective/intersection work ratio via first ~50% alpha convergence.
N3/N7: skippable suffix fraction after opacity saturation.
"""
from __future__ import annotations
import argparse,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch,gsplat
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from benchmark_framework import load_ply,load_cameras_from_json,resize_cameras
def stats(a):
 a=np.asarray(a,float)
 return {k:float(v) for k,v in zip(['mean','p50','p90','p95','p99','max'],[a.mean(),np.percentile(a,50),np.percentile(a,90),np.percentile(a,95),np.percentile(a,99),a.max()])}
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',choices=['N1','N3','N7'],required=True);p.add_argument('--out',required=True);p.add_argument('--camera-offset',type=int,default=0);p.add_argument('--camera-count',type=int,default=4);a=p.parse_args();torch.set_grad_enabled(False)
 W,H,TS=1920,1080,16
 scene=load_ply(str(ROOT/'data/official/mipnerf360/room/point_cloud.ply'),device='cuda');cams=resize_cameras(load_cameras_from_json(str(ROOT/'data/official/mipnerf360/room/cameras.json'),device='cuda'),W,H)
 means=scene['xyz'].contiguous();q=torch.nn.functional.normalize(scene['rotations'],dim=-1).contiguous();s=scene['scales'].exp().contiguous();o=torch.sigmoid(scene['opacity']).contiguous();c=scene['shs'].contiguous();bg=torch.zeros(1,3,device='cuda')
 tw,th=math.ceil(W/TS),math.ceil(H/TS);nt=tw*th
 recs=[]
 for cid in range(a.camera_offset,a.camera_offset+a.camera_count):
  cam=cams[cid%len(cams)];vw=cam.world_view_transform[None].contiguous();Ks=cam.K[None].contiguous()
  # stable intermediate data from a full render
  _,_,meta=gsplat.rasterization(means=means,quats=q,scales=s,opacities=o,colors=c,viewmats=vw,Ks=Ks,width=W,height=H,near_plane=.01,far_plane=1e10,radius_clip=0.,eps2d=.3,sh_degree=3,packed=False,tile_size=TS,backgrounds=bg,render_mode='RGB',sparse_grad=False,absgrad=False,rasterize_mode='classic');torch.cuda.synchronize()
  m2d=meta['means2d'].contiguous();r=meta['radii'].contiguous();d=meta['depths'].contiguous();cn=meta['conics'].contiguous();op=meta['opacities'].contiguous()
  _,iid,fid=gsplat.isect_tiles(m2d,r,d,TS,tw,th,sort=True);ioff=gsplat.isect_offset_encode(iid,1,tw,th)
  dir_=cam.camera_center.to('cuda')-means;dir_=dir_/dir_.norm(dim=-1,keepdim=True);col=gsplat.spherical_harmonics(3,dir_,c).unsqueeze(0)
  full_alpha=gsplat.rasterize_to_pixels(m2d,cn,col,op,W,H,TS,ioff,fid,bg)[1][0,...,0]
  torch.cuda.synchronize()
  off=ioff[0].reshape(-1);tile_pairs=[(int(off[i]),int(off[i+1] if i+1<nt else iid.numel())) for i in range(nt)]
  totals=[hi-lo for lo,hi in tile_pairs];n_tot=sum(totals)
  per_tile_trunc=[]
  for frac in [0.25,0.50,0.75]:
   chunks=[]
   for lo,hi in tile_pairs:
    take=lo+int((hi-lo)*frac+0.5)
    chunks.append(fid[lo:max(lo,min(take,hi))])
   trunc=torch.cat(chunks) if chunks else fid.new_empty(0)
   trunc_off=torch.zeros(nt+1,dtype=torch.int32,device='cuda');trunc_off[1:]=torch.cumsum(torch.tensor([len(x) for x in chunks],device='cuda'),0)
   ca=gsplat.rasterize_to_pixels(m2d,cn,col,op,W,H,TS,trunc_off[:nt].reshape(1,th,tw),trunc,bg)[1][0,...,0]
   torch.cuda.synchronize();err=torch.abs(ca-full_alpha)
   per_tile_trunc.append({'fraction':frac,'total_processed_intersections':int(trunc.numel()),'mse_alpha':float(torch.mean(err*err).item()),'max_alpha_error':float(err.max().item()),'pixel_fraction_alpha_error_lt_1e_3':float((err<1e-3).float().mean().item())})
  # per-tile alpha saturation analysis using only full alpha output
  alp=full_alpha;tw_=tw
  sat_tile_frac=[];candidate_suffix_ints_total=0;candidate_suffix_tile_count=0
  for ti,(lo,hi) in enumerate(tile_pairs):
   y,x=divmod(ti,tw_);pix=alp[y*TS:min((y+1)*TS,H),x*TS:min((x+1)*TS,W)];opq=float((pix>=0.999).float().mean().item())
   sat_tile_frac.append(opq)
   if opq>=1.0 and hi>lo:
    candidate_suffix_ints_total+=int(hi-lo);candidate_suffix_tile_count+=1
  recs.append({'camera':cid,'total_intersections':n_tot,'tiles_with_all_pixels_saturated':candidate_suffix_tile_count,'total_saturated_tile_intersection_fraction':candidate_suffix_ints_total/max(n_tot,1),'per_tile_pixel_saturation_fraction':stats(sat_tile_frac),'truncation_feasibility':per_tile_trunc})
 sat_frac=[x['total_saturated_tile_intersection_fraction'] for x in recs];px_sat=[x['per_tile_pixel_saturation_fraction']['mean'] for x in recs];t50=recs[0]['truncation_feasibility'][1]
 output={'schema_version':1,'candidate':a.candidate,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'environment':{'gpu':torch.cuda.get_device_name(),'gsplat':gsplat.__version__},'protocol':{'scene':'room official Mip-NeRF360','resolution':'1920x1080','tile_size':16,'cameras':list(range(a.camera_offset,a.camera_offset+a.camera_count))},'instrumentation_boundary':{'last_ids':'UNAVAILABLE: stock gsplat API does not return per-pixel last committed index. Replaced by depth-fraction replay comparison.','transmittance_after_each_batch':'UNAVAILABLE: replaced by per-tile truncation replay.','actual_early_termination_threshold':'CUDA source: next_T <= 1e-4'},'camera_records':recs,'aggregate':{'saturated_tile_intersection_fraction':stats(sat_frac),'per_tile_pixel_saturation_fraction':stats(px_sat),'truncation_50pct':{'alpha_mse':t50['mse_alpha'],'pixels_alpha_error_lt_1e_3':t50['pixel_fraction_alpha_error_lt_1e_3']}},'interpretation':'This is a feasibility bound, not a speedup measurement. The 50% truncation test renders half the depth-sorted intersections per tile; if alpha error is negligible, it bounds the potential for opacity-aware truncation.'}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(output,indent=2));print(a.out)
if __name__=='__main__':main()
