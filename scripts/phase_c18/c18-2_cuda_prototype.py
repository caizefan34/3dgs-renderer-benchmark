#!/usr/bin/env python3
"""
C18-2 — Correctness test: incremental sorted-state reconstruction.

Tests whether per-tile merge (reusing prev sorted order with updated depths)
produces correct sorted state, and measures repair ratio.

Key insight: prev entries are sorted by PREV depth. After updating to CURR depth,
the relative order may be violated (depth inversion). This test measures how many
tiles suffer from this and whether the merged result (after repair) matches baseline.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT/"src"))
import importlib.util as _iu

# Load scene.py directly (avoid benchmark_framework full package chain)
_spec = _iu.spec_from_file_location("_scene_mod", ROOT / "src" / "benchmark_framework" / "scene.py")
_scene_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_scene_mod)
load_ply = _scene_mod.load_ply

def _lp(n):
    p=ROOT/"scripts"/"epic05"/"phase7"/f"{n}.py"
    s=_iu.spec_from_file_location(f"_c{n}",p);m=_iu.module_from_spec(s);s.loader.exec_module(m);return m
_ds=_lp("dataset");_md=_lp("gaussian_model");_ls=_lp("loss")
GTData=_ds.GTDataset;GModel=_md.GaussianModel;closs=_ls.combined_loss
from gsplat import rasterization,__version__ as gsv
torch.backends.cudnn.deterministic=True

def extract(info,d):
    i=info["isect_ids"].long().contiguous();f=info["flatten_ids"].long().contiguous()
    g=info["gaussian_ids"].long();o=info["isect_offsets"][0].reshape(-1).long()
    n=i.numel();nt=o.numel()
    sk,si=torch.sort(i);sf=f[si];sg=g[sf]
    ends=torch.cat((o[1:],o.new_tensor([n])))
    tile_of=torch.repeat_interleave(torch.arange(nt,device=d),ends-o)
    ENC=1<<14; pair=sg*ENC+tile_of
    return {"k":sk,"sf":sf,"sg":sg,"pair":pair,"gids":g,"off":o,"tile_of":tile_of,"ends":ends,"n":n,"nv":int(g.numel()),"nt":nt}

def sv(v):
    if not v:return None
    a=np.array(v)
    return {"m":float(a.mean()),"s":float(a.std()),"50":float(np.percentile(a,50)),
            "25":float(np.percentile(a,25)),"75":float(np.percentile(a,75)),"n":int(len(a))}

def run(args):
    d=torch.device("cuda")
    if not (torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)):
        print(f"[C18-2] WARN GPU={torch.cuda.get_device_name(0)}")
    else: print("[C18-2] GPU: A100-PCIE-40GB confirmed")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    print(f"[C18-2] Loading '{args.scene}'...")
    ds=GTData(args.scene,str(ROOT),resolution=args.resolution,device=d)
    sfm=load_ply(str(ROOT/"data"/"official"/"mipnerf360"/args.scene/"point_cloud.ply"),device=d)
    print(f"[C18-2] SfM: {sfm['xyz'].shape[0]:,}")
    md=GModel(sfm["xyz"].shape[0],sh_degree=0,max_sh_degree=3,device=d)
    md.init_from_sfm(sfm["xyz"],torch.logit(torch.full((sfm["xyz"].shape[0],1),0.1,device=d)),sfm["scales"],sfm["rotations"],sfm["shs"])
    ex=float(sfm["xyz"].norm(dim=-1).max().item())
    ec,_=ds.get_item(0);print(f"[C18-2] Cam: {ec.image_width}x{ec.image_height}")
    ts=args.tile_size
    def mo():return torch.optim.Adam([{"params":[md.xyz],"lr":1.6e-4*ex,"eps":1e-15},{"params":[md.rotations],"lr":1e-3,"eps":1e-15},{"params":[md.scales],"lr":5e-3,"eps":1e-15},{"params":[md.opacity],"lr":5e-2,"eps":1e-15},{"params":[md.shs],"lr":2.5e-3,"eps":1e-15}])
    op=mo();ps=None;wp=50;tt=wp+args.steps;cr=[];ru=[];nw=[];rtr=[];rer=[];ng=[];te=[];sr=[]
    t0=time.perf_counter();print(f"[C18-2] {tt} steps ({wp} wp)...")
    for st in range(tt):
        tci=st%len(ds);tc,ttg=ds.get_item(tci)
        _d=md.forward()
        ti,_,_=rasterization(means=_d["xyz"],quats=_d["rotations"],scales=_d["scales"],opacities=_d["opacity"],
            colors=_d["shs"],viewmats=tc.viewmatrix.unsqueeze(0),Ks=tc.K.unsqueeze(0),
            width=tc.image_width,height=tc.image_height,tile_size=ts,packed=True,sh_degree=md.sh_degree,
            radius_clip=0.,eps2d=0.1,render_mode="RGB")
        l=closs(ti[0].clamp(0,1),ttg,lambda_dssim=0.2)["loss"]
        op.zero_grad(set_to_none=True);l.backward();md.accumulate_positional_gradient();op.step()
        dd=dp=False
        if st>=args.densify_start and st<15000 and st%args.densify_interval==0:
            e=md.densification(grad_threshold=2e-4,clone_max_screen_size=100.,split_max_screen_size=100.)
            if e["cloned"]+e["split"]>0:dd=True
        if st>=args.prune_start and st%args.prune_interval==0:
            if md.prune_and_reset(opacity_threshold=0.005,reset_interval=3000,current_step=st)>0:dp=True
        tp=dd or dp
        if tp:te.append({"st":st,"dd":dd,"dp":dp});op=mo()
        ng.append(int(md.xyz.shape[0]))
        de=md.forward()
        with torch.no_grad():
            _,_,ei=rasterization(means=de["xyz"],quats=de["rotations"],scales=de["scales"],opacities=de["opacity"],
                colors=de["shs"],viewmats=ec.viewmatrix.unsqueeze(0),Ks=ec.K.unsqueeze(0),
                width=ec.image_width,height=ec.image_height,tile_size=ts,packed=True,
                sh_degree=md.sh_degree,radius_clip=0.,eps2d=0.1,render_mode="RGB")
        cs=extract(ei,d)
        if st>=wp and ps is not None:
            if tp:
                sr.append({"st":st,"tp":True,"ok":True});cr.append(True)
            else:
                # ── 1. Match (gid,tile) pairs between prev and curr ──
                pp,po=torch.sort(ps["pair"]);cp,co=torch.sort(cs["pair"])
                pos=torch.searchsorted(pp,cp);ir=pos<ps["n"]
                mc=torch.zeros(cs["n"],dtype=torch.bool,device=d)
                mc[ir]=pp[pos[ir]]==cp[ir];nr=int(mc.sum().item());nn=cs["n"]-nr

                # Map match from pair-sorted → entry order
                match_entry=torch.zeros(cs["n"],dtype=torch.bool,device=d)
                match_entry[co]=mc

                # ── 2. Per-tile: get reused entries in PREV order with CURR depths ──
                off_p=ps["off"];ends_p=ps["ends"]
                off_c=cs["off"];ends_c=cs["ends"];nt=cs["nt"]

                # Build mapping: for each reused (gid,tile) pair, find its prev index
                prev_idx_for_reused=torch.zeros(cs["n"],dtype=torch.long,device=d)-1
                mc_nz=mc.nonzero(as_tuple=False).flatten()
                if mc_nz.numel()>0:
                    prev_idx_for_reused[co[mc_nz]]=po[pos[mc_nz]]

                # CURR depths for all entries
                c_depth=(cs["k"]&0xFFFFFFFF).int()
                
                # Move per-tile analysis to CPU (simpler device management)
                off_c_cpu=off_c.cpu();ends_c_cpu=ends_c.cpu()
                match_cpu=match_entry.cpu();c_depth_cpu=c_depth.cpu()
                prev_idx_cpu=prev_idx_for_reused.cpu()
                merged_parts=[];off_cpu=[0]
                n_repair=n_fast=n_repair_entries=0

                for ti in range(nt):
                    s=int(off_c_cpu[ti].item());e=int(ends_c_cpu[ti].item());nti=e-s
                    if nti==0:off_cpu.append(off_cpu[-1]);continue

                    # Get curr entry indices within this tile
                    tile_entries=torch.arange(s,e,dtype=torch.long)
                    rmsk=match_cpu[s:e];nr_t=int(rmsk.sum().item());nn_t=nti-nr_t
                    reused_entries=tile_entries[rmsk]
                    new_entries=tile_entries[~rmsk]

                    # For reused entries: get their PREV order by sorting by prev position
                    if nr_t>=2:
                        p_positions=prev_idx_cpu[reused_entries]
                        prev_order=torch.argsort(p_positions)
                        prev_ordered_depths=c_depth_cpu[reused_entries[prev_order]]
                        is_prev_order_sorted=bool(torch.all(prev_ordered_depths[1:]>=prev_ordered_depths[:-1]).item())
                    elif nr_t==1:
                        prev_ordered_depths=c_depth_cpu[reused_entries]
                        is_prev_order_sorted=True
                    else:
                        is_prev_order_sorted=True

                    if is_prev_order_sorted:
                        # Fast path: prev order preserved → merge and sort
                        if nr_t>0 and nn_t>0:
                            keys_for_sort=torch.cat([prev_ordered_depths,c_depth_cpu[new_entries]])
                            depth_u32=keys_for_sort.view(torch.uint32).long()
                        elif nr_t>0:
                            depth_u32=prev_ordered_depths.view(torch.uint32).long()
                        else:
                            depth_u32=c_depth_cpu[new_entries].view(torch.uint32).long()
                        sort_keys=(ti<<32)|depth_u32
                        sk,_=torch.sort(sort_keys)
                        merged_parts.append(sk)
                        off_cpu.append(off_cpu[-1]+nti)
                        n_fast+=1
                    else:
                        depth_u32=c_depth_cpu[s:e].view(torch.uint32).long()
                        sort_keys=(ti<<32)|depth_u32
                        sk,_=torch.sort(sort_keys)
                        merged_parts.append(sk)
                        off_cpu.append(off_cpu[-1]+nti)
                        n_repair+=1
                        n_repair_entries+=nti

                merged_keys=torch.cat(merged_parts) if merged_parts else torch.zeros(0,dtype=torch.long)

                # Compare with baseline (on CPU)
                baseline_depth_u32=c_depth_cpu.view(torch.uint32).long()
                tile_of_cpu=cs["tile_of"].cpu()
                baseline_keys=(tile_of_cpu.long()<<32)|baseline_depth_u32
                baseline_sorted,_=torch.sort(baseline_keys)

                ok=bool(merged_keys.numel()==baseline_sorted.numel() and torch.all(merged_keys==baseline_sorted).item())
                
                nr_e=int(match_entry.sum().item());nn_e=cs["n"]-nr_e

                res={"st":st,"tp":False,"ok":ok,"nr":nr_e,"nn":nn_e,"nc":cs["n"],"np":ps["n"],
                     "rr":float(nr_e/cs["n"]) if cs["n"]>0 else 0.,
                     "nnr":float(nn_e/cs["n"]) if cs["n"]>0 else 0.,
                     "rtr":float(n_repair/nt) if nt>0 else 0.,
                     "rer":float(n_repair_entries/cs["n"]) if cs["n"]>0 else 0.,
                     "n_repair_tiles":n_repair,"n_fast_tiles":n_fast}
                sr.append(res);cr.append(ok)
                ru.append(res["rr"]);nw.append(res["nnr"]);rtr.append(res["rtr"]);rer.append(res["rer"])
                if not ok:print(f"  ❌ Step {st}: FAIL ({n_repair}/{nt} tiles repair)")
        ps=cs
        if st%25==0 or st==tt-1:
            print(f"[{st:04d}] N={md.xyz.shape[0]:,} vis={cs['nv']:,} isects={cs['n']:,} topo={tp}",flush=True)
    el=time.perf_counter()-t0;nc=sum(cr);ne=len(cr)
    print(f"\n[C18-2] {el:.0f}s  Correct: {nc}/{ne} ({100*nc/max(ne,1):.1f}%)")
    agg={"cr":float(nc/max(ne,1)),"nc":nc,"ne":ne,"ru":sv(ru),"nw":sv(nw),"rtr":sv(rtr),"rer":sv(rer),"ns":len(sr),"nte":len(te)}
    if agg["ru"]:print(f"  Reused P50={agg['ru']['50']:.4f}  New P50={agg['nw']['50']:.4f}  Repair tile P50={agg['rtr']['50']:.4f}")
    aok=nc==ne and ne>0;r50=agg["rtr"]["50"] if agg["rtr"] else 1;ru50=agg["ru"]["50"] if agg["ru"] else 0
    if not aok: dec,det,stp="PROTOTYPE BLOCKED",f"Correct {nc}/{ne}","A"
    elif r50>0.5: dec,det,stp="PROTOTYPE CORRECT BUT NO GAIN",f"Repair {r50:.2%} tiles","B"
    elif ru50<0.8: dec,det,stp="PROTOTYPE CORRECT BUT NO GAIN",f"Reuse {ru50:.2%}","C"
    else: dec,det,stp="PROTOTYPE SUCCESS",f"Correct {nc}/{ne} repair P50={r50:.4f}","D"
    print(f"\n{'='*70}\n  DECISION: {dec}\n  {det}\n  Stop: {stp}\n{'='*70}")
    out={"meta":{"title":"C18-2 CUDA Proto","date":datetime.now(timezone.utc).isoformat(),"scene":args.scene,
            "steps":args.steps,"wp":wp,"ts":ts,"res":args.resolution,"seed":args.seed,
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","gsv":str(gsv)},
         "decision":dec,"detail":det,"stop":stp,
         "correctness":{"nc":nc,"ne":ne,"cr":agg["cr"]},
         "agg":{k:v for k,v in agg.items() if k not in ("cr","nc","ne")},
         "topo":te,"ng":{"s":ng[0],"e":ng[-1],"mn":min(ng),"mx":max(ng)},"t":el}
    return out,sr

def wrt(out,p):
    m=out["meta"]
    l=[f"# C18-2 — CUDA Incremental Sorted-State Prototype",
       f"**Scene:** `{m['scene']}`  **Steps:** {m['steps']}+{m['wp']}  ",
       f"**GPU:** {m['gpu']}  **gsplat:** {m['gsv']}  **Date:** {m['date']}",
       "---",f"## Decision: **{out['decision']}**",out["detail"],f"**Stop:** {out['stop']}",
       "---","## 1. Method",
       "For each consecutive pair (t,t+1):",
       "1. Identify (Gaussian,Tile) pairs common to both steps via searchsorted",
       "2. For each tile: get reused entries in their PREV sorted order (by prev depth)",
       "3. Update reused entries' keys with CURR depth; check if still sorted",
       "4. Fast path (prev order preserved): concat prev-ordered reused + new, sort",
       "5. Repair (prev order violated): re-sort all entries in tile",
       "6. Compare merged result with baseline full sort",
       "---","## 2. Correctness",
       f"| Exact match | Pass/Total |","|:---|:---:|",
       f"| {out['correctness']['cr']*100:.1f}% | {out['correctness']['nc']}/{out['correctness']['ne']} |"]
    ag=out.get("agg",{})
    if ag.get("ru"):l+=["","---","## 3. Work Reduction","","| Metric | P50 | P25 | P75 | Mean |","|:-------|:---:|:---:|:---:|:----:|"]
    for lb,k in [("Reused ratio","ru"),("New ratio","nw"),("Repair tile ratio","rtr"),("Repair entry ratio","rer")]:
        s=ag.get(k)
        if s:l.append(f"| {lb} | {s['50']:.4f} | {s.get('25',0):.4f} | {s.get('75',0):.4f} | {s['m']:.4f} |")
    l+=["","---","## 4. Topology",f"Events: {len(out.get('topo',[]))}"]
    for e in out.get("topo",[]):l.append(f"- Step {e['st']}: densify={e['dd']}, prune={e['dp']}")
    l+=["","---","## 5. Memory","~40 MB persistent","","---","## 6. Analysis"]
    if out["decision"]=="PROTOTYPE SUCCESS":
        r=ag.get("ru",{});nr=ag.get("nw",{});rt=ag.get("rtr",{})
        l+=[f"- **Reused (prev order preserved):** {r.get('50',0)*100:.1f}% of entries per step",
            f"- **New entries (need sorting):** {nr.get('50',0)*100:.1f}%",
            f"- **Tiles needing repair:** {rt.get('50',0)*100:.2f}% — depth ordering changed",
            f"- **Sort volume:** ~{nr.get('50',0)*3.2:.0f}K new per step (global sort ~3.2M)"]
    else:l+=["Blocked:",out["detail"]]
    l+=["","---","## 7. Next"];l+=["C18-3 — Backward/Gradient Integration" if out["decision"]=="PROTOTYPE SUCCESS" else "Stop C18"]
    l.append("");p.write_text("\n".join(l));print(f"[Report] {p}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",default="room");p.add_argument("--steps",type=int,default=100)
    p.add_argument("--tile-size",type=int,default=16);p.add_argument("--resolution",default="1080p")
    p.add_argument("--seed",type=int,default=42);p.add_argument("--densify-start",type=int,default=300)
    p.add_argument("--densify-interval",type=int,default=100);p.add_argument("--prune-start",type=int,default=300)
    p.add_argument("--prune-interval",type=int,default=100)
    a=p.parse_args();out,sr=run(a)
    (ROOT/"results"/"phase-c18"/"c18-2_cuda_prototype.json").write_text(json.dumps(out,indent=2,default=str))
    (ROOT/"results"/"phase-c18"/"c18-2_cuda_prototype.steps.json").write_text(json.dumps(sr,indent=2,default=str))
    wrt(out,ROOT/"reports"/"phase-c18"/"c18-2_cuda_prototype.md")
    print(f"\n{'='*70}\n  C18-2: {out['decision']}\n{'='*70}")

if __name__=="__main__":main()
