#!/usr/bin/env python3
"""H3-FWD-0 FP32 structural oracle.  Validation only; does not rasterize."""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from h2_bwd_0_structural import load_ply_scene, load_cameras, TILE_SIZE, SH_DEGREE

MTW, MTH, BATCH = 8, 4, 1024
AT = 1.0 / 255.0
MAX_EXTEND = 4096.0

def f2_b2(scene, cam, max_long_side, device):
    configs = {
      "room": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply","/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",3114,2075),
      "bicycle": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply","/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",4946,3286),
      "garden": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply","/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json",5187,3361),
    }
    ply,cams,nw,nh=configs[scene]; s=min(1.,max_long_side/max(nw,nh)); w,h=round(nw*s),round(nh*s)
    means,quats,scales,opacities,colors=load_ply_scene(ply,device)
    vm,K=load_cameras(cams,w,h,device); vm,K=vm[:,[cam]],K[:,[cam]]
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched,_gather_visible_native
    with torch.no_grad():
      ids,_,_= _cull_gaussians_batched(means,quats,scales,vm,K,w,h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=0.,camera_model="pinhole")
      m,q,s,o,c=_gather_visible_native(means,quats,scales,opacities,colors,ids)
      radii,m2d,depths,conics,_=fully_fused_projection(means=m.unsqueeze(0),covars=None,quats=q.unsqueeze(0),scales=s.unsqueeze(0),viewmats=vm,Ks=K,width=w,height=h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,packed=False,calc_compensations=False,camera_model="pinhole")
      opa=o[None,None].expand(1,1,-1).contiguous()
      tw,th=math.ceil(w/TILE_SIZE),math.ceil(h/TILE_SIZE)
      try:
       _,isect,flat=torch.ops.gsplat.intersect_tile(m2d.contiguous(),radii.contiguous(),depths.contiguous(),conics.contiguous(),opa.contiguous(),None,None,1,TILE_SIZE,tw,th,True,False,None)
      except RuntimeError as e:
       if "expected at most 13" not in str(e): raise
       _,isect,flat=torch.ops.gsplat.intersect_tile(m2d.contiguous(),radii.contiguous(),depths.contiguous(),conics.contiguous(),opa.contiguous(),None,None,1,TILE_SIZE,tw,th,True,False)
      offs=isect_offset_encode(isect,1,tw,th).reshape(th,tw)
    return dict(width=w,height=h,tw=tw,th=th,m2d=m2d[0,0].cpu().numpy(),conics=conics[0,0].cpu().numpy(),depth=depths[0,0].cpu().numpy(),opacity=o.cpu().numpy(),radii=radii[0,0].cpu().numpy().astype(np.int32),flat=flat.cpu().numpy().astype(np.int32),offs=offs.cpu().numpy(),nvis=len(o))

def emit(A,B,C,t,cx,cy,cols,rows,sx,sy):
    disc=B*B-A*C
    if not (disc < 0 and t > 0): return []
    ex=math.sqrt(-t*C/disc); ey=math.sqrt(-t*A/disc)
    xmin,xmax=cx-ex,cx+ex; ymin,ymax=cy-ey,cy+ey
    rx0=max(0,min(cols,int(xmin/sx))); rx1=max(0,min(cols,int(xmax/sx+1)))
    ry0=max(0,min(rows,int(ymin/sy))); ry1=max(0,min(rows,int(ymax/sy+1)))
    out=[]
    # Exact ellipse vs axis-aligned tile rectangle: include a tile iff ellipse intersects it.
    # Min quadratic over a rectangle is evaluated with edge candidates plus corners.
    for y in range(ry0,ry1):
      for x in range(rx0,rx1):
        x0,x1=x*sx,(x+1)*sx; y0,y1=y*sy,(y+1)*sy
        # convex quadratic minimization over rectangle; evaluate projected stationary points on each edge.
        pts=[(min(max(cx,x0),x1),min(max(cy,y0),y1))]
        for xx in (x0,x1):
          yy=min(max(cy-B*(xx-cx)/C,y0),y1); pts.append((xx,yy))
        for yy in (y0,y1):
          xx=min(max(cx-B*(yy-cy)/A,x0),x1); pts.append((xx,yy))
        q=min(A*(xx-cx)**2+2*B*(xx-cx)*(yy-cy)+C*(yy-cy)**2 for xx,yy in pts)
        if q <= t: out.append(y*cols+x)
    return out

def oracle(f):
    tw,th,n=f["tw"],f["th"],f["nvis"]; mw,mh=math.ceil(tw/MTW),math.ceil(th/MTH)
    macros=[[] for _ in range(mw*mh)]; fine=[[] for _ in range(tw*th)]
    for g,((cx,cy),(A,B,C),z,o) in enumerate(zip(f["m2d"],f["conics"],f["depth"],f["opacity"])):
      if not (o >= AT and A>0 and C>0): continue
      t=min(MAX_EXTEND*MAX_EXTEND,2*math.log(o/AT))
      mts=emit(A,B,C,t,float(cx),float(cy),mw,mh,MTW*TILE_SIZE,MTH*TILE_SIZE)
      tiles=emit(A,B,C,t,float(cx),float(cy),tw,th,TILE_SIZE,TILE_SIZE)
      for mt in mts: macros[mt].append(g)
      for tile in tiles: fine[tile].append(g)
    # depth then Gaussian ID is the explicit deterministic oracle tie policy.
    for xs in macros: xs.sort(key=lambda g:(float(f["depth"][g]),g))
    for xs in fine: xs.sort(key=lambda g:(float(f["depth"][g]),g))
    ref=[]; off=np.append(f["offs"].reshape(-1),len(f["flat"]))
    for i in range(tw*th): ref.append(list(map(int,f["flat"][off[i]:off[i+1]])))
    miss=extra=dups=wrong=order=inv=ties=0; diffs=[]; od=[]
    for i,(a,b) in enumerate(zip(ref,fine)):
      sa,sb=set(a),set(b); m=sa-sb;e=sb-sa
      miss+=len(m);extra+=len(e);dups+=len(b)-len(sb);wrong+=len(m)+len(e)
      if a==b: order+=1
      else:
       pos={g:j for j,g in enumerate(b)}
       common=[g for g in a if g in pos]
       iv=sum(pos[common[x]]>pos[common[y]] for x in range(len(common)) for y in range(x+1,len(common)))
       inv+=iv; ties+=sum(f["depth"][common[x]]==f["depth"][common[y]] for x in range(len(common)) for y in range(x+1,len(common)))
       diffs.append((i,len(a),len(b),len(m),len(e),len(b)-len(sb)))
       od.append((i,iv,len(common)))
    pops=[]
    for mt,xs in enumerate(macros):
      local=set(xs); base=(mt//mw)*MTH*tw+(mt%mw)*MTW
      for g in local:
       mask=sum(1<<j for j in range(32) if (base+(j//MTW)*tw+(j%MTW)) in range(tw*th) and g in fine[base+(j//MTW)*tw+(j%MTW)])
       pops.append(mask.bit_count())
    return dict(macro=macros,fine=fine,ref=ref,diffs=diffs,od=od,stats=dict(N_B2_tile_gaussian_pairs=len(f["flat"]),N_HiGS_macro_entries=sum(map(len,macros)),N_HiGS_fine_pairs_after_masks=sum(map(len,fine)),missing_pairs=miss,extra_pairs=extra,duplicate_pairs=dups,wrong_gaussian_ids=wrong,exact_order_match_fraction=order/(tw*th),pairwise_order_inversions=inv,tie_cases=ties,mask_popcounts=pops))

def main():
 p=argparse.ArgumentParser();p.add_argument("--scene",default="room");p.add_argument("--cam",type=int,default=0);p.add_argument("--max-long-side",type=int,default=2048);p.add_argument("--out",required=True);a=p.parse_args()
 f=f2_b2(a.scene,a.cam,a.max_long_side,"cuda"); r=oracle(f); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 s=r["stats"]; pops=np.asarray(s.pop("mask_popcounts"))
 s["mask"]={k:(float(np.mean(pops)) if k=="mean" else int(np.max(pops)) if k=="max" else float(np.percentile(pops,int(k[1:]))) ) for k in ["mean","p50","p90","p95","p99","max"]};s["compression"]=s["N_HiGS_fine_pairs_after_masks"]/s["N_HiGS_macro_entries"] if s["N_HiGS_macro_entries"] else 0
 (out/"structural_equivalence.json").write_text(json.dumps(s,indent=2))
 np.savetxt(out/"tile_pair_diff.csv",np.asarray(r["diffs"],dtype=np.int64),fmt="%d",delimiter=",",header="tile,b2_count,higs_count,missing,extra,duplicates",comments="")
 np.savetxt(out/"ordering_diff.csv",np.asarray(r["od"],dtype=np.int64),fmt="%d",delimiter=",",header="tile,inversions,common_count",comments="")
 (out/"mask_statistics.json").write_text(json.dumps(s["mask"],indent=2));print(json.dumps(s,indent=2))
if __name__=="__main__": main()
