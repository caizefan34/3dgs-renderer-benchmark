"""Debug: check if C1 collision groups lead to different ordering."""
import numpy as np, math
np.random.seed(42)

h, w, ts = 540, 960, 16
tw = math.ceil(w/ts); th = math.ceil(h/ts)
tnb = int(math.floor(math.log2(tw*th))) + 1
n = 10000

# Generate scene
depths = np.exp(np.random.uniform(np.log(0.2), np.log(6.0), n)).astype(np.float32)
mx = (w * np.random.beta(1.5, 1.5, n)).astype(np.float32)
my = (h * np.random.beta(1.5, 1.5, n)).astype(np.float32)
r = np.clip((64.0 / (depths + 0.01)).astype(np.int32), 1, w//4)
valid = r > 0
dv = depths[valid]; mxv = mx[valid]; myv = my[valid]; rv = r[valid]
giv = np.where(valid)[0]; nv = len(dv)

tr = rv.astype(np.float32) / ts
tminx = np.maximum(0, np.floor(mxv/ts - tr).astype(np.int32))
tminy = np.maximum(0, np.floor(myv/ts - tr).astype(np.int32))
tmaxx = np.minimum(tw, np.ceil(mxv/ts + tr).astype(np.int32))
tmaxy = np.minimum(th, np.ceil(myv/ts + tr).astype(np.int32))
wt = tmaxx-tminx; ht = tmaxy-tminy
ntp = (wt*ht).astype(np.int64); total = int(np.sum(ntp))

starts = np.zeros(nv+1, dtype=np.int64); starts[1:] = np.cumsum(ntp)
nz = ntp > 0; ns = starts[:-1][nz]; nw = wt[nz]
nx0 = tminx[nz]; ny0 = tminy[nz]; nd = dv[nz]; ng = giv[nz]
pos = np.arange(total, dtype=np.int64)
gi = np.searchsorted(ns, pos, side='right') - 1
off = pos - ns[gi]; row = off // nw[gi]; col = off % nw[gi]
tile_ids = (ny0[gi] + row) * tw + (nx0[gi] + col)
depth_out = nd[gi]; gidx_out = ng[gi]

# Build keys
d32 = depth_out.view(np.uint32)
dp_hi = (d32 >> 16).astype(np.int64)
baseline_key = (np.int64(0) | tile_ids.astype(np.int64) << 32 | d32.astype(np.int64))
c1_key = (dp_hi | tile_ids.astype(np.int64) << 16 | np.int64(0))

# Sort
bo = np.argsort(baseline_key, kind='stable')
co = np.argsort(c1_key, kind='stable')

# Sample 5 tiles with collisions - compare ordering
tiles_with_col = []
for test_tile in range(tw * th):
    mask = tile_ids == test_tile
    cnt = int(np.sum(mask))
    if cnt >= 3:
        d = depth_out[mask]; g = gidx_out[mask]
        d32_t = d.view(np.uint32)
        dp_hi_t = d32_t >> 16
        if len(np.unique(dp_hi_t)) < cnt:  # has collision
            tiles_with_col.append(test_tile)
            if len(tiles_with_col) >= 5:
                break

print(f"Found {len(tiles_with_col)} tiles with collisions (sample):")
for tid in tiles_with_col:
    mask = tile_ids == tid
    cnt = int(np.sum(mask))
    d = depth_out[mask]; g = gidx_out[mask]
    
    # Get baseline order (within tile)
    b_mask = mask[bo]
    b_g_in_tile = gidx_out[bo][b_mask]
    c_mask = mask[co]
    c_g_in_tile = gidx_out[co][c_mask]
    
    same = np.array_equal(b_g_in_tile, c_g_in_tile)
    if not same:
        d_in_tile = depth_out[bo][b_mask]
        print(f"\nTile {tid}: {cnt} items, ORDERING DIFFERS")
        print(f"  Baseline order (gidx): {b_g_in_tile[:10].tolist()}{'...' if cnt>10 else ''}")
        print(f"  C1 order      (gidx): {c_g_in_tile[:10].tolist()}{'...' if cnt>10 else ''}")
        print(f"  Baseline depths: {[f'{x:.6f}' for x in d_in_tile[:10]]}{'...' if cnt>10 else ''}")
        # Check correlation
        d_sorted = d_in_tile[:cnt]
        print(f"  Depth sorted (baseline): {np.all(np.diff(d_sorted)>=0)}")
    else:
        print(f"Tile {tid}: {cnt} items, ordering SAME")
