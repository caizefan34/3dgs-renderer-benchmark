#!/usr/bin/env python3
"""H6-0: isolated one-touch AccuTile construction validation on mx."""
import argparse, importlib.util, json, math, os, runpy, sys
from pathlib import Path
import numpy as np
import torch
from h3_fwd_0_oracle import f2_b2
from h3_fwd_1a_r_validate import tile_lists, cuda_lists

def load(source, core):
    sys.path.insert(0, str(Path(source)))
    spec=importlib.util.spec_from_file_location('gsplat_cuda',core); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.modules['gsplat.csrc']=m
    return runpy.run_path(str(Path(source)/'gsplat/experimental/render/kernels/cuda/build.py'))['build_and_load_experimental_gaussian_render_inference_scene']()
def stat(x):
    x=np.asarray(x,dtype=np.float64); return dict(median_ms=float(np.median(x)),mean_ms=float(x.mean()),p10_ms=float(np.percentile(x,10)),p90_ms=float(np.percentile(x,90)),std_ms=float(x.std()),samples=int(x.size))
def event(fn):
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); return a.elapsed_time(b)
def percent(x):
    x=np.asarray(x,dtype=np.float64); return dict(mean=float(x.mean()),p50=float(np.percentile(x,50)),p90=float(np.percentile(x,90)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),max=int(x.max()))
def structural(reference, out, depths, tw, th):
    got,offs,ids,batches,masks=cuda_lists(out[:4],tw,th); missing=extra=duplicate=wrong=mask_bad=non_tie=tie_inv=0
    mw=math.ceil(tw/8)
    reference_masks={}
    # Reconstruct oracle masks from B2 membership, separately from the one-touch masks.
    for tile,rows in enumerate(reference):
        mx,my=(tile%tw)//8,(tile//tw)//4; bit=(tile//tw%4)*8+(tile%tw%8)
        for g in rows: reference_masks[(my*mw+mx,g)]=reference_masks.get((my*mw+mx,g),0)|(1<<bit)
    seen={}
    for mt in range(len(offs)-1):
        for p in range(int(offs[mt]),int(offs[mt+1])):
            g=int(ids[p]); seen[(mt,g)]=int(np.uint32(masks[p]))
    for k,v in reference_masks.items():
        if k not in seen: mask_bad+=1
        elif seen[k]!=v: mask_bad+=1
    mask_bad+=sum(k not in reference_masks for k in seen)
    for ref,actual in zip(reference,got):
        rs,gs=set(ref),set(actual); missing+=len(rs-gs);extra+=len(gs-rs);duplicate+=len(actual)-len(gs)
        pos={g:i for i,g in enumerate(actual)}; common=[g for g in ref if g in pos]
        for i,l in enumerate(common):
            for r in common[i+1:]:
                if pos[l]>pos[r]:
                    if depths[l]==depths[r]: tie_inv+=1
                    else: non_tie+=1
    wrong=missing+extra
    return dict(missing=missing,extra=extra,duplicate=duplicate,wrong_ID=wrong,mask_mismatch=mask_bad,semantic_non_tie_inversion=non_tie,equal_depth_permutation_inversions=tie_inv,N_macro_entries=int(offs[-1]),N_reconstructed_fine_pairs=sum(map(len,got)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--core-so',required=True);p.add_argument('--out',required=True);a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ext=load(a.source,a.core_so); from gsplat.cuda._wrapper import isect_offset_encode
    all_time=[]; all_stage=[]; capacities={}; structurals={}; sorts=[]; memories={}
    for scene in ('room','bicycle','garden'):
        f=f2_b2(scene,0,2048,'cuda'); ins=[torch.from_numpy(f[k]).cuda() for k in ('m2d','conics','depth','opacity','radii')];m2d,con,dep,opa,rad=ins; bm2d,brad,bdep,bcon=(x[None,None].contiguous() for x in (m2d,rad,dep,con));bopa=opa[None,None].contiguous();nm=math.ceil(f['tw']/8)*math.ceil(f['th']/4)
        def b2():
            try: _,ix,_=torch.ops.gsplat.intersect_tile(bm2d,brad,bdep,bcon,bopa,None,None,1,16,f['tw'],f['th'],True,False,None)
            except RuntimeError: _,ix,_=torch.ops.gsplat.intersect_tile(bm2d,brad,bdep,bcon,bopa,None,None,1,16,f['tw'],f['th'],True,False)
            return isect_offset_encode(ix,1,f['tw'],f['th'])
        def r2(profile=False): return ext.higs_train_macro_f4(*ins,f['tw'],f['th'],16,True,profile)
        def one(profile=False): return ext.higs_train_one_touch_f4(*ins,f['tw'],f['th'],16,profile)
        final=one(False);torch.cuda.synchronize(); ref=tile_lists(f['flat'],f['offs'],f['tw']*f['th']); s=structural(ref,final,f['depth'],f['tw'],f['th']);structurals[scene]=s;(out/f'structural_{scene}.json').write_text(json.dumps(dict(scene=scene,cam=0,**s),indent=2))
        cc=final[5].cpu().numpy(); candidate=int(cc.sum()); exact=s['N_macro_entries']; capacities[scene]=dict(N_visible=int(f['nvis']),N_candidate_macro_bbox=candidate,N_exact_macro_entries=exact,candidate_exact_ratio=candidate/exact,candidate_macros_per_visible_gaussian=percent(cc),upper_bound_stream_bytes=dict(record_soa_bytes=16*candidate,valid_bytes=candidate,candidate_count_and_prefix_bytes=8*len(cc),total_raw_stream_bytes=17*candidate+8*len(cc)))
        # 20 warmup then five interleaved 100-sample repetitions.
        for _ in range(20): b2();r2();one()
        torch.cuda.synchronize(); bt=[];rt=[];ot=[]
        for _rep in range(5):
            for _ in range(100): bt.append(event(b2));rt.append(event(r2));ot.append(event(one))
        for name,vals in [('B2_F4',bt),('R2_Macro_F4',rt),('One_Touch_Macro_F4',ot)]: all_time.append(dict(scene=scene,variant=name,**stat(vals)))
        # Stage samples are deliberately outside total timing to avoid profiler-event perturbation.
        rstage=[[] for _ in range(6)];ostage=[[] for _ in range(6)]
        for _rep in range(5):
            for _ in range(100):
                for i,v in enumerate(r2(True)[4].cpu().numpy()):rstage[i].append(float(v))
                for i,v in enumerate(one(True)[4].cpu().numpy()):ostage[i].append(float(v))
        for name,groups in [('R2',rstage),('One_Touch',ostage)]:
            labels=(['count_exact_coverage','scan','fill_exact_coverage','segmented_sort','batch_metadata','entry_mask_write'] if name=='R2' else ['candidate_macro_bbox_generation','candidate_scan_and_upper_bound_allocation','exact_AccuTile_emit_mask_once','compact_valid_records','global_radix_sort_and_offsets','return_bookkeeping'])
            for label,vals in zip(labels,groups):all_stage.append(dict(scene=scene,variant=name,stage=label,**stat(vals)))
        # Both sort alternatives consume the exact same compacted one-touch stream captured above.
        rawmacro,rawdepth,rawids,rawmasks=final[6:10]
        for _ in range(20):ext.higs_one_touch_global_sort(rawmacro,rawdepth,rawids,rawmasks,nm);ext.higs_one_touch_segmented_sort(rawmacro,rawdepth,rawids,nm)
        torch.cuda.synchronize();g=[];q=[]
        for _rep in range(5):
            for _ in range(100):g.append(event(lambda:ext.higs_one_touch_global_sort(rawmacro,rawdepth,rawids,rawmasks,nm)));q.append(event(lambda:ext.higs_one_touch_segmented_sort(rawmacro,rawdepth,rawids,nm)))
        e=exact; sorts += [dict(scene=scene,representation='A_global_radix_macro_plus_depth',workspace_bytes=24*e+8*nm,total_construction_workspace_bytes=24*e+16*nm,**stat(g)),dict(scene=scene,representation='B_macro_count_offset_segmented_depth',workspace_bytes=20*e+12*nm,total_construction_workspace_bytes=20*e+16*nm,**stat(q))]
        # Logical live-tensor accounting; it excludes caching allocator reserve and includes all temporary SoA streams.
        persistent=12*(nm+1)+8*e; temporary=8*len(cc)+17*candidate+36*e+16*nm; memories[scene]=dict(B2='copied_from_R2_fresh_process_artifact',R2_Macro_F4='copied_from_R2_fresh_process_artifact',One_Touch=dict(persistent_bytes=persistent,temporary_bytes=temporary,peak_bytes=persistent+temporary,formula='12*(macro_count+1)+8*exact + 8*Nvisible + 17*candidate + 36*exact + 16*macro_count'))
    import csv
    def csvwrite(name,rows):
        with open(out/name,'w',newline='') as h:
            cols=list(rows[0]);w=csv.DictWriter(h,fieldnames=cols);w.writeheader();w.writerows(rows)
    csvwrite('one_touch_timing.csv',all_time);csvwrite('current_stage_timing.csv',all_stage);csvwrite('sort_comparison.csv',sorts)
    (out/'candidate_capacity.json').write_text(json.dumps(capacities,indent=2));(out/'memory_one_touch.json').write_text(json.dumps(memories,indent=2));
    (out/'run_summary.json').write_text(json.dumps(dict(protocol=dict(warmup=20,measurements_per_rep=100,repetitions=5,clock='CUDA Events',interleaved=True),capacities=capacities,structural=structurals,memory_one_touch=memories),indent=2))
if __name__=='__main__':main()
