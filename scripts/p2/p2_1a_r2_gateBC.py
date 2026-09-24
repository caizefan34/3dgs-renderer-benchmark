#!/usr/bin/env python3
"""P2-1A-R2 Gate B/C: fine-tile structural + ordering equivalence."""
import json, os, sys, math, importlib.util, argparse
import torch

CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2 = "/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"
CAM_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
MAX_LONG = 2048; TS = 16; MTW, MTH = 8, 4
CHOL_SCALE = 0.84932180028 * TS; MAX_EXTEND = 3.33
LOG2_INV_AT = math.log2(255.0)
MAX_EXTEND_CHOL_SQ = (math.log2(math.e) * 0.5) * MAX_EXTEND * MAX_EXTEND

def load_so(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def make_viewmat_K(cam, dev):
    R = torch.tensor(cam["rotation"], dtype=torch.float32, device=dev)
    pos = torch.tensor(cam["position"], dtype=torch.float32, device=dev)
    w, h = float(cam["width"]), float(cam["height"]); sf = MAX_LONG/max(w,h)
    W, H = int(round(w*sf)), int(round(h*sf))
    vm = torch.eye(4, dtype=torch.float32, device=dev); vm[:3,:3]=R; vm[:3,3]=-R@pos
    K = torch.tensor([[cam["fx"]*sf,0,W/2-0.5],[0,cam["fy"]*sf,H/2-0.5],[0,0,1]],dtype=torch.float32,device=dev)[None]
    return vm[None], K, W, H

def base_tiles_from_isect(isect_ids, flatten_ids, TW, TH):
    ids = isect_ids.cpu().numpy(); flat = flatten_ids.cpu().numpy()
    n_tiles = TW*TH; tile_n_bits = int(math.floor(math.log2(n_tiles)))+1
    mask = (1 << tile_n_bits) - 1; tids = (ids >> 32) & mask
    tiles = {}
    for i in range(ids.shape[0]):
        tiles.setdefault(int(tids[i]), []).append(int(flat[i]))
    return tiles

def native_tiles(macro_offsets, sorted_rows, means2d, conics, opac, W, H, depths):
    TW, TH = math.ceil(W/TS), math.ceil(H/TS); mc_cols = (TW+MTW-1)//MTW
    l0 = torch.sqrt(torch.clamp_min(conics[...,0],0.0))
    l1 = torch.where(l0>1e-12, conics[...,1]/l0, torch.zeros_like(l0))
    l2 = torch.sqrt(torch.clamp_min(conics[...,2]-l1*l1,0.0))
    log2_opac = torch.log2(torch.clamp_min(opac, 1e-30))
    t_rast = torch.min(torch.full_like(log2_opac, MAX_EXTEND_CHOL_SQ), log2_opac+LOG2_INV_AT)
    rl0=l0*CHOL_SCALE; rl1=l1*CHOL_SCALE; rl2=l2*CHOL_SCALE
    C = rl1*rl1+rl2*rl2; nlr = torch.where(C>0, -rl1/C, torch.zeros_like(C))
    tiles = {}; mt_depths = {}
    n_macro = macro_offsets.shape[0]-1
    for mt in range(n_macro):
        s, e = int(macro_offsets[mt]), int(macro_offsets[mt+1])
        if s==e: continue
        gids = sorted_rows[s:e]
        mt_col = mt % mc_cols; mt_row = mt // mc_cols
        mt_cx = mt_col*MTW+MTW*0.5; mt_cy = mt_row*MTH+MTH*0.5
        g = torch.tensor(gids, device=means2d.device)
        x_t = means2d[g,0]/TS-mt_cx; y_t = means2d[g,1]/TS-mt_cy
        a0=rl0[g]; a1=rl1[g]; a2=rl2[g]; tr=t_rast[g]; nr=nlr[g]; dp=depths[g].cpu().numpy()
        seen = {}
        for r in range(MTH):
            ny = y_t-(r+0.5-MTH*0.5); dy0=-0.5-ny; dy1=dy0+1.0; dy_h=torch.where(ny>=0,dy1,dy0)
            l1_dy=a1*dy_h; v_h=a2*dy_h; v_h_sq=v_h*v_h
            for c in range(MTW):
                nx = x_t-(c+0.5-MTW*0.5); dx0=-0.5-nx; l0dx0=a0*dx0; l0dx1=l0dx0+a0
                u_h=torch.minimum(torch.maximum(torch.zeros_like(l0dx0), l0dx0+l1_dy), l0dx1+l1_dy); q_h=u_h*u_h+v_h_sq
                l0dx=torch.where(nx>=0,l0dx1,l0dx0); dy_v=torch.clamp(l0dx*nr,dy0,dy1)
                u_v=a1*dy_v+l0dx; v_v=a2*dy_v; q_v=v_v*v_v+u_v*u_v
                c_hit=(torch.abs(nx)<0.5)&(torch.abs(ny)<0.5); e_hit=torch.minimum(q_h,q_v)<=tr
                hit=(c_hit|e_hit).nonzero().flatten().tolist()
                if hit:
                    tid=(mt_row*MTH+r)*TW+(mt_col*MTW+c)
                    lst = tiles.setdefault(tid, []); dlst = mt_depths.setdefault(tid, [])
                    for idx in hit:
                        key = int(gids[idx])
                        if key not in seen:
                            seen[key] = True
                            lst.append(key); dlst.append(float(dp[idx]))
    return tiles, mt_depths

def cmp_orders(base_l, nat_l, db, dn, at):
    """Non-tie inversions + tie-only differences over shared gid set, per tile.
    db/dn are dicts gid->depth for base and native."""
    set_b = set(base_l); set_n = set(nat_l); shared = set_b & set_n
    ob = [g for g in base_l if g in shared]; on = [g for g in nat_l if g in shared]
    pb = {g: i for i, g in enumerate(ob)}; pn = {g: i for i, g in enumerate(on)}
    m = len(ob); nti = 0; tie = 0; md = 0.0
    for a in range(m):
        for b in range(a+1, m):
            ga, gb = ob[a], ob[b]
            dga, dgb = db[ga], db[gb]
            if pb[ga] < pb[gb]:      # base: ga before gb
                if pn[ga] > pn[gb]:  # native: gb before ga -> inversion
                    if abs(dga-dgb) <= at:
                        tie += 1
                    else:
                        nti += 1
                        md = max(md, abs(dga-dgb))
    return {"non_tie_inversions": nti, "tie_only": tie, "max_depth_diff_reordered": md,
            "n_shared": m}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", default="room,bicycle,garden"); ap.add_argument("--out", required=True)
    args = ap.parse_args(); torch.set_grad_enabled(False); dev = torch.device('cuda')
    core = load_so(CORE, "gsplat_cuda"); r2 = load_so(R2, "experimental_gaussian_render_inference_scene_cuda")
    bg = torch.tensor([0.,0.,0.], device=dev).float(); os.makedirs(args.out, exist_ok=True)
    res = {}
    for sc in args.scenes.split(","):
        cam = json.load(open(f"{CAM_BASE}/{sc}/cameras.json"))[0]
        ck = torch.load(f"{CKPT_BASE}/a100_30k_{sc}_t16_16/a100_30k_{sc}_t16_16_latest.pt", map_location="cpu", weights_only=False)
        st = ck.get("model_state", ck)
        means=st["xyz"].to(dev).contiguous(); quats=st["rotations"].to(dev).contiguous()
        scales=torch.exp(st["scales"].to(dev)).contiguous(); opac=torch.sigmoid(st["opacity"].to(dev).flatten()).contiguous()
        sh=st["shs"].to(dev).contiguous(); vm,K,W,H=make_viewmat_K(cam,dev)
        N=means.numel()//3; visible_ids=torch.arange(N,device=dev,dtype=torch.int64)
        cam_pos=r2.higs_camera_positions_from_viewmats(vm.contiguous())
        radii,means2d,depths,conics,opac_bc,colors=r2.higs_gatherless_projected_producer(
            visible_ids,means,quats,scales,opac,sh,vm.contiguous(),K.contiguous(),cam_pos,W,H,0.3,0.01,1e10,0.0)
        TW,TH=math.ceil(W/TS),math.ceil(H/TS)
        mm2d=means2d.reshape(1,1,N,2); rr=radii.reshape(1,1,N,2); dd=depths.reshape(1,1,N)
        cc=conics.reshape(1,1,N,3); oo=opac_bc.reshape(1,1,N); col=colors.reshape(1,1,N,3)
        tp,isect_ids,flatten=torch.ops.gsplat.intersect_tile(mm2d,rr,dd,cc,oo,None,None,1,TS,TW,TH,True,False)
        base_tiles=base_tiles_from_isect(isect_ids,flatten,TW,TH)
        rbg,alpha,diag=torch.ops.experimental.higs_native_hierarchy_from_projected(
            visible_ids,radii,means2d,depths,conics,opac_bc,colors,W,H,TS,bg,True)
        macro_offsets,sorted_rows,batch_offsets,active_masks=diag[0].cpu().numpy(),diag[1].cpu().numpy(),diag[2].cpu().numpy(),diag[3].cpu().numpy()
        nat_tiles, nat_depths = native_tiles(macro_offsets,sorted_rows,means2d,conics,opac_bc,W,H,depths)
        # Gate B
        missing=0; extra=0
        for tid in nat_tiles:
            b=set(base_tiles.get(tid,[])); n=set(nat_tiles[tid]); missing+=len(b-n); extra+=len(n-b)
        # duplicates
        dup_b=sum(len(v)-len(set(v)) for v in base_tiles.values()); dup_n=0
        for v in nat_tiles.values():
            s=set(); d=0
            for x in v: d += (1 if x in s else 0); s.add(x)
            dup_n+=d
        bpc=sum(len(v) for v in base_tiles.values()); npc=sum(len(v) for v in nat_tiles.values())
        struct={"baseline_pair_count":bpc,"native_pair_count":npc,"missing":missing,"extra":extra,
                "duplicates_base":dup_b,"duplicates_native":dup_n,
                "pass_b":(missing==0 and extra==0 and dup_b==0 and dup_n==0)}
        # depth dicts by gid (F9 depth is per-gid; both BASE and native sort by this depth)
        dep_g = depths.cpu().numpy()
        def _ddict(lst): return {g: float(dep_g[g]) for g in lst}
        # Gate C on shared tiles
        non_tie_total=0; tie_total=0; max_dd=0.0; n_shared_g=0
        at = 1e-6  # fp32 reassociation tie tolerance (absolute, documented)
        for tid in set(base_tiles)&set(nat_tiles):
            o = cmp_orders(base_tiles[tid], nat_tiles[tid], _ddict(base_tiles[tid]), _ddict(nat_tiles[tid]), at)
            non_tie_total+=o["non_tie_inversions"]; tie_total+=o["tie_only"]
            max_dd=max(max_dd,o["max_depth_diff_reordered"]); n_shared_g+=o["n_shared"]
        order={"non_tie_inversions":non_tie_total,"tie_only_differences":tie_total,
               "max_depth_diff_reordered":max_dd,"n_shared_gids":n_shared_g,
               "pass_c": non_tie_total==0}
        res[sc]={"gateB":struct,"gateC":order}
    outo={"scenes":res}
    json.dump(outo, open(os.path.join(args.out,"bc_results.json"),"w"), indent=2, default=str)
    print(json.dumps(outo, indent=2, default=str))

if __name__=="__main__":
    main()