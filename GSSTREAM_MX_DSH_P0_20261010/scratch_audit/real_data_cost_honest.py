#!/usr/bin/env python3
"""Honest G2 representation-density on REAL data, comparing the KIT'S row-span
candidate count against the REAL per-tile footprint (radii) the rasterizer uses.

Motivation: the previous script compared row-span candidates against a per-Gaussian
full-tile-bbox pixel-visit proxy (~6.7x the true intersection count), which is not the
honest baseline. gsplat/FastGS build a per-Gaussian tile footprint from `radii` (the
conservative 3-sigma screen extent) and emit a (gaussian,tile) pair per overlapped
tile. We reconstruct that footprint here from the stored conics (identical math the
kernel uses: radius = 3*max(sigma_x,sigma_y)) and count tiles per gaussian, then
compare against the row-span candidate pixel count and the kernel's real pair count.
"""
import numpy as np, glob, os, json, math

D = r'C:\Users\36570\3dgs-renderer-benchmark\experiments\training_solver_design\TILE_PREFIX_TRANSMITTANCE_GATE_20260928'
CUTOFF = 1.0/255.0
H, W = 1066, 1600
TS = 16

def load(f):
    z = np.load(f, allow_pickle=True)
    xy = z['projected_xy'].astype(np.float64)
    co = z['conic_opacity'].astype(np.float64)
    a,b,c,op = co[:,0],co[:,1],co[:,2],co[:,3]
    return z, xy, a,b,c,op

def tile_footprint_tiles(xy, a,b,c, op, ts=TS, nsigma=3.0):
    """Tiles per gaussian using conservative screen-radius footprint (gsplat/FastGS math)."""
    det = a*c - b*b
    inv_det = 1.0/np.maximum(det,1e-30)
    var_x = c*inv_det; var_y = a*inv_det      # marginal variances of the projected gaussian
    sigma_x = np.sqrt(np.maximum(var_x,0.0)); sigma_y = np.sqrt(np.maximum(var_y,0.0))
    radius = nsigma*np.maximum(sigma_x, sigma_y)
    tx0 = np.floor((xy[:,0]-radius)/ts).astype(np.int64)
    tx1 = np.floor((xy[:,0]+radius)/ts).astype(np.int64)
    ty0 = np.floor((xy[:,1]-radius)/ts).astype(np.int64)
    ty1 = np.floor((xy[:,1]+radius)/ts).astype(np.int64)
    gx = (W-1)//ts; gy=(H-1)//ts
    tx0=np.clip(tx0,0,gx); tx1=np.clip(tx1,0,gx); ty0=np.clip(ty0,0,gy); ty1=np.clip(ty1,0,gy)
    w = tx1-tx0+1; h = ty1-ty0+1
    tiles_per_g = w*h
    total_tile_pairs = int(tiles_per_g.sum())
    return total_tile_pairs, tiles_per_g, radius

def row_span(xy, a,b,c, op, ts=TS):
    """Kit's conservative row-span candidate PIXEL count (superset, exact alpha recheck per pixel)."""
    det = a*c-b*b
    tau = 2.0*np.log(np.maximum(op,1e-12)/CUTOFF)
    maxy = np.sqrt(np.maximum(tau*a/np.maximum(det,1e-30),0.0))
    n=len(op); intervals=0; candidates=0
    x0f=np.floor((xy[:,0]-W)); x1f=np.ceil((xy[:,0]+W))  # not used
    for i in range(n):
        if op[i]<CUTOFF: continue
        y0=max(0,int(math.floor(xy[i,1]-maxy[i]-2))); y1=min(H-1,int(math.ceil(xy[i,1]+maxy[i]+2)))
        ai,bi,ci,di,xi=a[i],b[i],c[i],det[i],xy[i,0]
        cnt=0
        for y in range(y0,y1+1):
            dy=y-xi*0+ (y-xy[i,1])
            discr=ai*tau[i]-di*dy*dy
            if discr<0: continue
            sq=math.sqrt(discr)
            l=(xi+(-bi*dy-sq)/ai)-2; r=(xi+(-bi*dy+sq)/ai)+2
            ix0=max(0,int(math.floor(l))); ix1=min(W-1,int(math.ceil(r)))
            if ix0<=ix1: intervals+=1; cnt+=(ix1-ix0+1)
        candidates+=cnt
    return intervals, candidates

files = sorted(glob.glob(os.path.join(D,'native_tiles_*.npz')))
out=[]
for f in files:
    z,xy,a,b,c,op = load(f)
    # SPD + visible filter
    det=a*c-b*b
    m=(a>0)&(c>0)&(det>0)
    gids=np.unique(z['gaussian_ids'].astype(np.int64)); vis=np.zeros(len(op),bool); vis[gids]=True
    use=m&vis&(op>=CUTOFF)
    xy2,a2,b2,c2,o2 = xy[use],a[use],b[use],c[use],op[use]
    real_pairs = len(z['gaussian_ids'])
    fp_tiles, tpg, radius = tile_footprint_tiles(xy2,a2,b2,c2,o2)
    iv, cd = row_span(xy2,a2,b2,c2,o2)
    # per-gaussian row-span candidates / per-gaussian tile footprint
    rec=dict(file=os.path.basename(f), N_visible=int(use.sum()),
             real_intersect_pairs=int(real_pairs),
             footprint_tile_pairs=int(fp_tiles),
             footprint_tiles_per_g=float(tpg.mean()),
             row_intervals=int(iv), row_span_candidates=int(cd),
             row_span_per_g=float(cd/max(use.sum(),1)))
    rec['rowspan_candidates/footprint_tile_pixels']=float(cd/fp_tiles)
    rec['footprint_tiles/real_intersect']=float(fp_tiles/real_pairs)
    out.append(rec); print(rec)

import numpy as np
rs_real=np.array([r['real_intersect_pairs'] for r in out])
fp=np.array([r['footprint_tile_pairs'] for r in out])
cd=np.array([r['row_span_candidates'] for r in out])
rs2fp=np.array([r['rowspan_candidates/footprint_tile_pixels'] for r in out])
fpg=np.array([r['footprint_tiles_per_g'] for r in out])
print('\n=== HONEST G2 ratios (REAL room cohort, per-tile footprint baseline) ===')
print(f'real_intersect / footprint_tile_pairs : median {np.median(rs_real/fp):.4f}  (opacity-culled fraction)')
print(f'footprint_tiles / real_intersect      : median {np.median(fp/rs_real):.2f}  (bbox slack)')
print(f'row_span_candidates / footprint_pixels: median {np.median(rs2fp):.3f}  mean {np.mean(rs2fp):.3f}')
print(f'footprint tiles per visible gaussian   : median {np.median(fpg):.1f}')
json.dump(out, open(os.path.join(os.path.dirname(__file__),'real_cohort_cost_honest.json'),'w'), indent=2)
