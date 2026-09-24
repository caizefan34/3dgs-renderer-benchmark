#!/usr/bin/env python3
"""H6-0R direct-bucketed exact one-touch validation on mx."""
import argparse,csv,importlib.util,json,math,runpy,sys
from pathlib import Path
import numpy as np
import torch
from h3_fwd_0_oracle import f2_b2
from h3_fwd_1a_r_validate import tile_lists,cuda_lists
def load(source,core):
 sys.path.insert(0,str(Path(source)));s=importlib.util.spec_from_file_location('gsplat_cuda',core);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);sys.modules['gsplat.csrc']=m;return runpy.run_path(str(Path(source)/'gsplat/experimental/render/kernels/cuda/build.py'))['build_and_load_experimental_gaussian_render_inference_scene']()
def stats(x):
 x=np.asarray(x,dtype=np.float64);return dict(median_ms=float(np.median(x)),mean_ms=float(x.mean()),p10_ms=float(np.percentile(x,10)),p90_ms=float(np.percentile(x,90)),std_ms=float(x.std()),samples=int(x.size))
def event(fn):
 a,b=torch.cuda.Event(True),torch.cuda.Event(True);a.record();fn();b.record();b.synchronize();return a.elapsed_time(b)
def structural(reference,out,depths,tw,th):
 got,offs,ids,batches,masks=cuda_lists(out[:4],tw,th);missing=extra=dup=maskbad=non_tie=tieinv=tiepairs=0;mw=math.ceil(tw/8);refmask={}
 for tile,rows in enumerate(reference):
  mt=(tile//tw//4)*mw+(tile%tw//8);bit=(tile//tw%4)*8+(tile%tw%8)
  for g in rows:refmask[(mt,g)]=refmask.get((mt,g),0)|(1<<bit)
 seen={}
 for mt in range(len(offs)-1):
  for p in range(int(offs[mt]),int(offs[mt+1])):seen[(mt,int(ids[p]))]=int(np.uint32(masks[p]))
 maskbad=sum(k not in seen or seen[k]!=v for k,v in refmask.items())+sum(k not in refmask for k in seen)
 for ref,actual in zip(reference,got):
  rs,gs=set(ref),set(actual);missing+=len(rs-gs);extra+=len(gs-rs);dup+=len(actual)-len(gs);pos={g:i for i,g in enumerate(actual)};common=[g for g in ref if g in pos]
  for i,l in enumerate(common):
   for r in common[i+1:]:
    if depths[l]==depths[r]:tiepairs+=1
    if pos[l]>pos[r]:
     if depths[l]==depths[r]:tieinv+=1
     else:non_tie+=1
 return dict(missing=missing,extra=extra,duplicate=dup,wrong_ID=missing+extra,mask_mismatch=maskbad,semantic_non_tie_inversion=non_tie,equal_depth_pair_comparisons=tiepairs,equal_depth_permutation_inversions=tieinv,N_macro_entries=int(offs[-1]),N_reconstructed_fine_pairs=sum(map(len,got)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--core-so',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ext=load(a.source,a.core_so);from gsplat.cuda._wrapper import isect_offset_encode
 timing=[];stage=[];structs={};ties={};memory={};breakeven={};f5={'room':.6717,'bicycle':.9103,'garden':.4127}
 for scene in ('room','bicycle','garden'):
  f=f2_b2(scene,0,2048,'cuda');ins=[torch.from_numpy(f[k]).cuda() for k in ('m2d','conics','depth','opacity','radii')];m2d,con,dep,opa,rad=ins;bm2d,brad,bdep,bcon=(x[None,None].contiguous() for x in (m2d,rad,dep,con));bopa=opa[None,None].contiguous();nm=math.ceil(f['tw']/8)*math.ceil(f['th']/4)
  def b2():
   try:_,ix,_=torch.ops.gsplat.intersect_tile(bm2d,brad,bdep,bcon,bopa,None,None,1,16,f['tw'],f['th'],True,False,None)
   except RuntimeError:_,ix,_=torch.ops.gsplat.intersect_tile(bm2d,brad,bdep,bcon,bopa,None,None,1,16,f['tw'],f['th'],True,False)
   return isect_offset_encode(ix,1,f['tw'],f['th'])
  def r2(pr=False):return ext.higs_train_macro_f4(*ins,f['tw'],f['th'],16,True,pr)
  def h60(pr=False):return ext.higs_train_one_touch_f4(*ins,f['tw'],f['th'],16,pr)
  def h6r(pr=False):return ext.higs_train_direct_bucket_one_touch_f4(*ins,f['tw'],f['th'],16,pr)
  result=h6r(False);torch.cuda.synchronize();ref=tile_lists(f['flat'],f['offs'],f['tw']*f['th']);st=structural(ref,result,f['depth'],f['tw'],f['th']);structs[scene]=st;ties[scene]={k:st[k] for k in ('equal_depth_pair_comparisons','equal_depth_permutation_inversions','semantic_non_tie_inversion')};(out/f'structural_{scene}.json').write_text(json.dumps(dict(scene=scene,cam=0,**st),indent=2))
  for _ in range(20):b2();r2();h60();h6r()
  torch.cuda.synchronize();b=[];r=[];g=[];d=[]
  for rep in range(5):
   for i in range(100):b.append(event(b2));r.append(event(r2));g.append(event(h60));d.append(event(h6r))
  rows=[('B2_F4',b),('R2_Macro_F4',r),('H6_0_Global_One_Touch',g),('H6_0R_Direct_Bucket',d)]
  for name,x in rows:timing.append(dict(scene=scene,variant=name,**stats(x)))
  samples=[[] for _ in range(7)]
  for rep in range(5):
   for i in range(100):
    for j,v in enumerate(h6r(True)[4].cpu().numpy()):samples[j].append(float(v))
  for name,x in zip(['candidate_bbox','candidate_scan_and_slot_allocation','exact_coverage_and_macro_count','macro_scan','valid_rebucket_scatter','segmented_payload_depth_sort','batch_metadata'],samples):stage.append(dict(scene=scene,stage=name,**stats(x)))
  ts=stats(d);bs=stats(b);pen=ts['median_ms']-bs['median_ms'];remaining=f5[scene]-pen;speed=None if remaining<=0 else f5[scene]/remaining;breakeven[scene]=dict(B2_F4_ms=bs['median_ms'],H6_0R_F4_ms=ts['median_ms'],F4_penalty_ms=pen,B2_F5_ms=f5[scene],F4_penalty_fraction_of_B2_F5=pen/f5[scene],F4_penalty_percent_of_B2_F5=100*pen/f5[scene],required_future_F5_ms_for_break_even=remaining,required_future_F5_speedup_vs_B2=speed,target_max_F4_ms=bs['median_ms']+.05*f5[scene])
  candidates=int(result[5].numel());entries=st['N_macro_entries'];persistent=12*(nm+1)+8*entries;candidate_bytes=16*candidates;macro_count_bytes=4*nm;offset_bytes=4*(nm+1);scatter_bytes=4*nm+8*entries;seg_bytes=8*entries;temporary=8*int(f['nvis'])+candidate_bytes+8*nm+16*entries;memory[scene]=dict(B2_F4='see H3-R2 memory.json',R2_Macro_F4='see H3-R2 memory.json',H6_0_Global='see H6-0 memory.json',H6_0R=dict(candidate_record_bytes=candidate_bytes,macro_counts_bytes=macro_count_bytes,offsets_bytes=offset_bytes,scatter_workspace_bytes=scatter_bytes,segmented_sort_workspace_bytes=seg_bytes,persistent_bytes=persistent,temporary_bytes=temporary,peak_bytes=persistent+temporary,formula='8*Nvisible + 16*candidates + 8*macro_count + 16*exact + persistent'))
 def write(name,rows):
  with open(out/name,'w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write('timing.csv',timing);write('stage_timing.csv',stage);(out/'memory.json').write_text(json.dumps(memory,indent=2));(out/'tie_ordering.json').write_text(json.dumps(ties,indent=2));(out/'forward_break_even.json').write_text(json.dumps(breakeven,indent=2));(out/'run_summary.json').write_text(json.dumps(dict(protocol='20 warmup; 100 measurements; 5 reps; interleaved; CUDA Events',structural=structs,break_even=breakeven),indent=2))
if __name__=='__main__':main()
