#!/usr/bin/env python3
"""G2/G4/G5 cost model on REAL projected data (native_tiles_*.npz, room scene).

CPU-only. Uses real means2d/conics/opacities/colors/depths captured from the
INRIA/FastGS rasterizer, plus the real tile-intersection structure the kernel
already produced (gaussian_ids per tile, offsets). Measures:
  - actual_active_hit_count : real (gaussian,tile) intersections from kernel
  - tile_bbox_candidate_count: tile-bbox candidate proxy over the SAME visible gaussians
  - stream_candidate_count / row_interval_count : conservative row-span support of visible gaussians
This is the preregistered G2 representation-density measurement on REAL data.
"""
import numpy as np, glob, os, json, math

D = r'C:\Users\36570\3dgs-renderer-benchmark\experiments\training_solver_design\TILE_PREFIX_TRANSMITTANCE_GATE_20260928'
CUTOFF = 1.0/255.0

def spd_filter(xy, co, op, rgb, dep):
    a,b,c = co[:,0],co[:,1],co[:,2]
    det = a*c - b*b
    m = (a>0)&(c>0)&(det>0)&np.isfinite(xy).all(1)&np.isfinite(co).all(1)
    return m

def row_span_stats(xy, co, op, H, W, pad=0):
    """Conservative row-span candidate + interval counts over visible gaussians."""
    n = len(op)
    intervals = 0
    candidates = 0
    a = co[:,0]; b = co[:,1]; c = co[:,2]; det = a*c - b*b
    tau = 2.0*np.log(np.maximum(op,1e-12)/CUTOFF)
    maxy = np.sqrt(np.maximum(tau*a/np.maximum(det,1e-30),0.0))
    for i in range(n):
        if op[i] < CUTOFF: continue
        y0 = max(0, int(math.floor(xy[i,1]-maxy[i]-pad-2)))
        y1 = min(H-1, int(math.ceil(xy[i,1]+maxy[i]+pad+2)))
        cnt = 0
        for y in range(y0,y1+1):
            dy = y-xy[i,1]
            discr = a[i]*tau[i]-det[i]*dy*dy
            if discr < 0: continue
            sq = math.sqrt(discr)
            x0 = xy[i,0]-((-b[i]*dy-sq)/a[i])-pad-2
            x1 = xy[i,0]+((-b[i]*dy+sq)/a[i])+pad+2
            ix0 = max(0,int(math.floor(x0))); ix1=min(W-1,int(math.ceil(x1)))
            if ix0<=ix1:
                intervals += 1; cnt += (ix1-ix0+1)
        candidates += cnt
    return intervals, candidates

def tile_bbox_candidates(xy, co, op, H, W, ts):
    """Tile-bbox candidate pixel-visit proxy for the SAME gaussians."""
    cost = 0
    a=co[:,0]; b=co[:,1]; c=co[:,2]; det=a*c-b*b
    tau=2.0*np.log(np.maximum(op,1e-12)/CUTOFF)
    maxx=np.sqrt(np.maximum(tau*c/np.maximum(det,1e-30),0.0))
    maxy=np.sqrt(np.maximum(tau*a/np.maximum(det,1e-30),0.0))
    for i in range(len(op)):
        if op[i]<CUTOFF: continue
        lx=max(0,int(math.floor((xy[i,0]-maxx[i]-2)/ts))); rx=min((W-1)//ts,int(math.floor((xy[i,0]+maxx[i]+2)/ts)))
        ly=max(0,int(math.floor((xy[i,1]-maxy[i]-2)/ts))); ry=min((H-1)//ts,int(math.floor((xy[i,1]+maxy[i]+2)/ts)))
        if lx>rx or ly>ry: continue
        for ty in range(ly,ry+1):
            for tx in range(lx,rx+1):
                cost += min(ts,W-tx*ts)*min(ts,H-ty*ts)
    return cost

# Authoritative frame from camera_tile_manifest.json sidecar: 1600 x 1060 (W x H), mult=0.5
H, W = 1066, 1600
files = sorted(glob.glob(os.path.join(D,'native_tiles_*.npz')))
rows = []
for f in files:
    z = np.load(f, allow_pickle=True)
    xy = z['projected_xy'].astype(np.float64)
    co = z['conic_opacity'].astype(np.float64)[:, :3]
    op = z['conic_opacity'].astype(np.float64)[:, 3]
    rgb = z['rgb'].astype(np.float64)
    dep = z['depth'].astype(np.float64)
    # real intersections: gaussian_ids unique = gaussians that landed in >=1 tile
    gids = z['gaussian_ids'].astype(np.int64)
    n_intersect = len(gids)                 # real (gaussian,tile) pairs emitted by kernel
    visible = np.unique(gids)               # gaussians that survived culling into a tile
    m = spd_filter(xy, co, op, rgb, dep)
    vis_in = np.zeros(len(op), bool); vis_in[visible]=True
    use = m & vis_in                        # SPD AND actually-visible
    xy2,co2,op2 = xy[use],co[use],op[use]
    ts = 16  # authoritative tile size from camera_tile_manifest.json render.tile_size
    tb = tile_bbox_candidates(xy2,co2,op2,H,W,ts)
    iv,cd = row_span_stats(xy2,co2,op2,H,W,0)
    # per-camera: use the real tile_offsets to get per-tile counts if available
    rows.append(dict(file=os.path.basename(f), N_total=len(op), N_intersect_pairs=int(n_intersect),
                     N_spd_visible=int(use.sum()),
                     tile_bbox_candidate_visits=int(tb),
                     row_intervals=int(iv), row_span_candidates=int(cd),
                     tile_size=ts))
    print(rows[-1])

# Aggregate ratios
import numpy as np
tb = np.array([r['tile_bbox_candidate_visits'] for r in rows])
cd = np.array([r['row_span_candidates'] for r in rows])
iv = np.array([r['row_intervals'] for r in rows])
ni = np.array([r['N_intersect_pairs'] for r in rows])
print('\n=== G2 representation-density ratios (REAL room-scene cohort) ===')
print(f'active_hit/tile_bbox_candidate : median {np.median(ni/tb):.4f}  mean {np.mean(ni/tb):.4f}  (gate <=0.20)')
print(f'stream_candidate/tile_bbox     : median {np.median(cd/tb):.4f}  mean {np.mean(cd/tb):.4f}  (gate <=0.35)')
print(f'intervals per intersect pair   : median {np.median(iv/np.maximum(ni,1)):.2f}')
json.dump(rows, open(os.path.join(os.path.dirname(__file__),'real_cohort_cost.json'),'w'), indent=2)
