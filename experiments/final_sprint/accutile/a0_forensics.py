"""Single allowed AccuTile forensic investigation; no renderer backport."""
import csv
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from a0_measure import ALPHA, DATA, EXTEND, PHASE7, TILE, load as load_scene, quadratic_min_on_tile
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles

ROOT = Path(os.environ.get("ACCUTILE_A0_ROOT", Path(__file__).resolve().parent))
OUT = ROOT / "raw"


def bounds(mean, radius, conic, opacity, tw, th):
    mx, my = map(float, mean); rx, ry = map(float, radius); a, b, c = map(float, conic)
    aabb = [max(0, min(tw, math.floor(mx / TILE - rx / TILE))), max(0, min(th, math.floor(my / TILE - ry / TILE))),
            max(0, min(tw, math.ceil(mx / TILE + rx / TILE))), max(0, min(th, math.ceil(my / TILE + ry / TILE)))]
    t = min(EXTEND * EXTEND, 2.0 * math.log(opacity / ALPHA))
    disc = b * b - a * c
    xe, ye = math.sqrt((-t / disc) * c), math.sqrt((-t / disc) * a)
    bmin, bmax = (mx - xe, my - ye), (mx + xe, my + ye)
    rmin = [max(0, min(tw, int(bmin[0] / TILE))), max(0, min(th, int(bmin[1] / TILE)))]
    rmax = [max(0, min(tw, int(bmax[0] / TILE + 1.0))), max(0, min(th, int(bmax[1] / TILE + 1.0)))]
    # Reproduce accutile_process_tiles's selected v-range for this tile's u column.
    argmin = [my + b * xe / c, mx + b * ye / a]
    argmax = [my - b * xe / c, mx - b * ye / a]
    is_y = (rmax[1] - rmin[1]) < (rmax[0] - rmin[0])
    if is_y:
        rm0, rm1 = [rmin[1], rmin[0]], [rmax[1], rmax[0]]
        mn, mxv, amin, amax = [bmin[1], bmin[0]], [bmax[1], bmax[0]], [argmin[1], argmin[0]], [argmax[1], argmax[0]]
        u, v = None, None
    else:
        rm0, rm1, mn, mxv, amin, amax = rmin, rmax, list(bmin), list(bmax), argmin, argmax
        u, v = None, None
    return aabb, bmin, bmax, rmin, rmax, is_y, (a, b, c, disc, t, xe, ye, rm0, rm1, mn, mxv, amin, amax)


def selected_v_range(detail, tile_x, tile_y):
    a, b, c, disc, t, xe, ye, rmin, rmax, bmin, bmax, argmin, argmax = detail
    is_y = False  # replaced by caller's coordinate mapping
    # `bmin/bmax/arg*` and rect values already use AccuTile's possibly-swapped axes.
    return rmin, rmax, bmin, bmax, argmin, argmax


def range_for_tile(mean, conic, opacity, tw, th, tile_x, tile_y):
    aabb, bmin_orig, bmax_orig, snug_min, snug_max, is_y, raw = bounds(mean, [0, 0], conic, opacity, tw, th)
    # Recompute without relying on radius for the process-tiles range.
    a, b, c = map(float, conic); mx, my = map(float, mean); t = min(EXTEND * EXTEND, 2 * math.log(opacity / ALPHA)); disc = b*b-a*c
    xe, ye = math.sqrt((-t/disc)*c), math.sqrt((-t/disc)*a)
    bmin, bmax = [mx-xe, my-ye], [mx+xe, my+ye]
    argmin, argmax = [my+b*xe/c, mx+b*ye/a], [my-b*xe/c, mx-b*ye/a]
    rmin = [max(0,min(tw,int(bmin[0]/TILE))), max(0,min(th,int(bmin[1]/TILE)))]
    rmax = [max(0,min(tw,int(bmax[0]/TILE+1))), max(0,min(th,int(bmax[1]/TILE+1)))]
    is_y = (rmax[1]-rmin[1]) < (rmax[0]-rmin[0])
    if is_y:
        rmin, rmax = [rmin[1],rmin[0]], [rmax[1],rmax[0]]
        bmin, bmax, argmin, argmax = [bmin[1],bmin[0]], [bmax[1],bmax[0]], [argmin[1],argmin[0]], [argmax[1],argmax[0]]
        u, v = tile_y, tile_x
    else:
        u, v = tile_x, tile_y
    min_line, max_line = u*TILE, (u+1)*TILE
    def intersect(coord):
        h = coord - mean[1 if is_y else 0]
        coeff = a if is_y else c
        puv = mean[0 if is_y else 1]
        root = math.sqrt(max(0.0, disc*h*h + t*coeff))
        return ((-b*h-root)/coeff+puv, (-b*h+root)/coeff+puv)
    mxline = (bmax[1], bmin[1])
    mnline = intersect(min_line) if bmin[0] <= min_line else mxline
    if max_line <= bmax[0]: mxline = intersect(max_line)
    emin = bmin[1] if min_line <= argmin[1] < max_line else min(mnline[0], mxline[0])
    emax = bmax[1] if min_line <= argmax[1] < max_line else max(mnline[1], mxline[1])
    vlo = max(rmin[1], min(rmax[1], int(emin / TILE)))
    vhi = min(rmax[1], max(rmin[1], int(emax / TILE + 1)))
    return {"is_y": is_y, "u": u, "v": v, "selected_v_min": vlo, "selected_v_max_exclusive": vhi,
            "snug_min": snug_min, "snug_max": snug_max}


def classify(ratio):
    if ratio <= 1.0001: return "NUMERIC_BORDERLINE"
    if ratio <= 1.01: return "SMALL_MARGIN"
    return "MATERIAL"


def scene_cases(scene, extension):
    state, camera, viewmat, K = load_scene(scene)
    xyz, quats = state["xyz"].detach().cuda(), state["rotations"].detach().cuda()
    scales, opacity = torch.exp(state["scales"].detach()).cuda(), state["opacity"].detach().flatten().cuda().float()
    width, height = camera["width"], camera["height"]; tw, th = math.ceil(width/TILE), math.ceil(height/TILE)
    with torch.no_grad():
        projection = fully_fused_projection(xyz, None, quats, scales, viewmat, K, width, height, eps2d=.3, near_plane=.01, far_plane=1e10, radius_clip=0., packed=True, sparse_grad=False, calc_compensations=False, camera_model="pinhole", opacities=opacity)
        bi, ci, gi, _, radii, means2d, depths, conics, _ = projection
        packed_opacity, image_ids = opacity[gi].contiguous(), bi*ci
        a_tpg, a_ids, a_flat = isect_tiles(means2d,radii,depths,TILE,tw,th,sort=False,segmented=False,packed=True,n_images=1,image_ids=image_ids,gaussian_ids=gi)
        u_tpg, u_ids, u_flat = isect_tiles(means2d,radii,depths,TILE,tw,th,sort=False,segmented=False,packed=True,n_images=1,image_ids=image_ids,gaussian_ids=gi,conics=conics.float().contiguous(),opacities=packed_opacity)
        ntiles=tw*th; mask=(1<<math.ceil(math.log2(ntiles)))-1
        ak=torch.sort(a_flat.long()*ntiles+((a_ids>>32)&mask)).values; uk=torch.sort(u_flat.long()*ntiles+((u_ids>>32)&mask)).values
        pos=torch.searchsorted(uk,ak); keep=(pos<uk.numel())&(uk[pos.clamp_max(uk.numel()-1)]==ak); removed=ak[~keep]
        candidate=[]
        for start in range(0,removed.numel(),1_000_000):
            keys=removed[start:start+1_000_000]; flat=torch.div(keys,ntiles,rounding_mode="floor"); tile=keys.remainder(ntiles)
            qmin=quadratic_min_on_tile(means2d[flat],conics.float()[flat],tile,width,height,TILE)
            t=torch.minimum(torch.full_like(packed_opacity[flat],EXTEND*EXTEND),2*torch.log(packed_opacity[flat]/ALPHA))
            hit=(qmin-t)<=1e-5
            if hit.any(): candidate.append((flat[hit],tile[hit],qmin[hit],t[hit]))
        flat,tile,qmin,tlevel=(torch.cat(x) for x in zip(*candidate))
        n=flat.numel(); ma=torch.empty(n,device="cuda"); sig=torch.empty(n,device="cuda"); px=torch.empty(n,dtype=torch.int32,device="cuda"); py=torch.empty(n,dtype=torch.int32,device="cuda")
        extension.evaluate_fragments(means2d[flat].float().contiguous(),conics[flat].float().contiguous(),packed_opacity[flat].float().contiguous(),tile.long().contiguous(),width,height,TILE,ma,sig,px,py)
        valid=ma>=ALPHA
        rows=[]
        for k in valid.nonzero().flatten().cpu().tolist():
            f=int(flat[k]); tx=int(tile[k].item()%tw); ty=int(tile[k].item()//tw); mean=means2d[f].float().cpu().tolist(); conic=conics[f].float().cpu().tolist(); radius=radii[f].cpu().tolist(); opa=float(packed_opacity[f].item())
            aabb,_,_,snug_min,snug_max,_,_=bounds(mean,radius,conic,opa,tw,th)
            rng=range_for_tile(mean,conic,opa,tw,th,tx,ty)
            ratio=float(ma[k].item()/ALPHA)
            rows.append({"scene":scene,"camera":0,"gaussian_id":int(gi[f].item()),"flat_id":f,"tile_x":tx,"tile_y":ty,"mean2d":json.dumps(mean),"conic_A":conic[0],"conic_B":conic[1],"conic_C":conic[2],"opacity":opa,"radius_x":radius[0],"radius_y":radius[1],"depth":float(depths[f].item()),"aabb_bounds":json.dumps(aabb),"snugbox_bounds":json.dumps({"min":snug_min,"max":snug_max}),"accutile_selected_range":json.dumps(rng),"pixel_x":int(px[k].item()),"pixel_y":int(py[k].item()),"sigma":float(sig[k].item()),"gaussian_weight_G":float(math.exp(-float(sig[k].item()))),"effective_alpha":float(ma[k].item()),"alpha_threshold":ALPHA,"alpha_threshold_ratio":ratio,"distance_from_threshold":float(ma[k].item()-ALPHA),"margin_class":classify(ratio),"image_boundary":tx in (0,tw-1) or ty in (0,th-1),"ellipse_bbox_boundary":tx in (snug_min[0],snug_max[0]-1) or ty in (snug_min[1],snug_max[1]-1),"tile_column_boundary":rng["v"] in (rng["selected_v_min"]-1,rng["selected_v_max_exclusive"]),"tile_row_boundary":ty in (0,th-1)})
        return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ext=load(name="accutile_fragment_forensics_v1",sources=[str(ROOT/"fragment_forensics_ext.cpp"),str(ROOT/"fragment_forensics_ext.cu")],extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],build_directory=str(ROOT/"build"),verbose=True)
    rows=[]
    for scene in ("room","bicycle","garden"):
        rows.extend(scene_cases(scene,ext)); torch.cuda.empty_cache()
    path=OUT.parent/"false_negative_forensics.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/"forensics_summary.json").write_text(json.dumps({"n":len(rows),"max_ratio":max(r["alpha_threshold_ratio"] for r in rows),"material":sum(r["margin_class"]=="MATERIAL" for r in rows),"by_scene":{s:sum(r["scene"]==s for r in rows) for s in ("room","bicycle","garden")}},indent=2),encoding="utf-8")
    print(json.dumps(json.loads((OUT/"forensics_summary.json").read_text()),indent=2))

if __name__=="__main__": main()
