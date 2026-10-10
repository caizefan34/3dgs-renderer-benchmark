#!/usr/bin/env python3
"""Independent geometric count reference for sampled HiGS projected Gaussians.

Conic mode: count tiles whose closed axis-aligned square intersects the
opacity-thresholded ellipse q(x,y)<=t, computed via a convex quadratic minimum
on the square's 4 edges. This is independent of gsplat's AccuTile strip scanner.
CUDA AccuTile's conservative discretization / float rounding can legitimately
differ on boundaries. A discrepancy is a diagnostic, NOT automatic proof of bug.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import numpy as np


def parse_value(text, key):
    number = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)[fF]?"
    for line in text.splitlines():
        if key not in line or line.lstrip().startswith('//'):
            continue
        if not re.search(r'\b' + re.escape(key) + r'\b', line):
            continue
        # Support C/C++ assignment and #define forms, plus 1.f / 255.f.
        key_match = re.search(r'\b' + re.escape(key) + r'\b', line)
        expression = line[key_match.end():].split('//',1)[0].strip()
        if expression.startswith('='):
            expression = expression[1:].strip()
        elif not line.lstrip().startswith('#define'):
            continue
        expression = expression.strip(' ();')
        m = re.match(r'^' + number + r'(?:\s*/\s*' + number + r')?', expression)
        if m:
            return float(m.group(1)) / float(m.group(2)) if m.group(2) else float(m.group(1))
    return None


def geometric_tile_count(mx, my, rx, ry, A, B, C, opacity, W, H, T, alpha, extend, mask=None):
    if not all(map(math.isfinite, (mx, my, rx, ry))):
        return None, 'nonfinite_projection'
    if rx <= 0 or ry <= 0:
        return 0, None
    if A is None or opacity is None:
        x0 = max(0, min(W, math.floor((mx-rx)/T)))
        x1 = max(0, min(W, math.ceil((mx+rx)/T)))
        y0 = max(0, min(H, math.floor((my-ry)/T)))
        y1 = max(0, min(H, math.ceil((my+ry)/T)))
        if x1 <= x0 or y1 <= y0:
            return 0, None
        if mask is None:
            return (x1-x0)*(y1-y0), None
        return int(np.asarray(mask[y0:y1,x0:x1], dtype=np.bool_).sum()), None
    if not all(map(math.isfinite, (A, B, C, opacity))):
        return None, 'nonfinite_conic_or_opacity'
    det = A*C - B*B
    if det <= 0 or A <= 0 or C <= 0:
        return None, 'non_positive_definite_conic'
    if opacity <= alpha:
        return 0, 'opacity_below_threshold'
    t = min(extend*extend, 2.0*math.log(opacity/alpha))
    if t <= 0:
        return 0, None
    ex = math.sqrt(t*C/det)
    ey = math.sqrt(t*A/det)
    if not (math.isfinite(ex) and math.isfinite(ey)):
        return None, 'nonfinite_ellipse_extent'
    # Enclose the entire ellipse. Extra geometric boundary tiles can differ
    # from CUDA's numerical conservative SNUGBOX/AccuTile path.
    x0 = max(0, min(W, math.floor((mx-ex)/T)))
    x1 = max(0, min(W, math.ceil((mx+ex)/T)))
    y0 = max(0, min(H, math.floor((my-ey)/T)))
    y1 = max(0, min(H, math.ceil((my+ey)/T)))
    if x1 <= x0 or y1 <= y0:
        return 0, None
    xi, yi = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    lo_x = xi*T; hi_x = (xi+1)*T
    lo_y = yi*T; hi_y = (yi+1)*T
    if lo_x.size == 0:
        return 0, None
    inside = ((lo_x <= mx) & (mx <= hi_x) & (lo_y <= my) & (my <= hi_y))
    min_q = np.full(lo_x.shape, np.inf, dtype=np.float64)
    for xedge in (lo_x, hi_x):
        dx = xedge - mx
        dy = np.clip(my - B*dx/C, lo_y, hi_y) - my
        q = A*dx*dx + 2*B*dx*dy + C*dy*dy
        min_q = np.minimum(min_q, q)
    for yedge in (lo_y, hi_y):
        dy = yedge - my
        dx = np.clip(mx - B*dy/A, lo_x, hi_x) - mx
        q = A*dx*dx + 2*B*dx*dy + C*dy*dy
        min_q = np.minimum(min_q, q)
    overlap = inside | (min_q <= t)
    if mask is not None:
        overlap &= np.asarray(mask[y0:y1,x0:x1], dtype=bool)
    return int(overlap.sum()), None


def run_oracle(capture, outdir, alpha, extend):
    import torch
    data = torch.load(capture, map_location='cpu', weights_only=False)
    md = data['meta']; W = int(md['width_in_tiles']); H = int(md['height_in_tiles']); T = int(md['tile_size'])
    def array(name):
        v = data.get(name)
        return None if v is None else v.detach().numpy()
    means = array('means2d').reshape(-1, len(data['sample_indices']), 2)
    radii = array('radii').reshape(-1, len(data['sample_indices']), 2)
    depths = array('depths').reshape(-1, len(data['sample_indices']))
    conics = array('conics')
    if conics is not None: conics=conics.reshape(-1,len(data['sample_indices']),3)
    opacities = array('opacities')
    if opacities is not None: opacities=opacities.reshape(-1,len(data['sample_indices']))
    masks = array('tile_mask')
    if masks is not None: masks = masks.reshape(-1,H,W)
    gpu = array('gpu_counts').reshape(-1,len(data['sample_indices']))
    sample_ids = array('sample_indices')
    if not (len(means) == len(gpu) == int(md['n_images'])):
        raise RuntimeError('Captured shapes disagree with metadata')
    outdir.mkdir(parents=True, exist_ok=True)
    diff_count = 0; valid = 0; invalid = {}; gpu_sum = 0; oracle_sum=0; first=None; above_physical=0; above_aabb=0
    rows=[]
    for camera in range(len(means)):
        for j, source_id in enumerate(sample_ids):
            xy = means[camera,j]; rr = radii[camera,j]
            ab = conics[camera,j] if conics is not None else [None,None,None]
            opacity = float(opacities[camera,j]) if opacities is not None else None
            mask = masks[camera] if masks is not None else None
            count, reason = geometric_tile_count(float(xy[0]), float(xy[1]),float(rr[0]),float(rr[1]),
                *[float(x) if x is not None else None for x in ab],opacity,W,H,T,alpha,extend,mask)
            g = int(gpu[camera,j]); gpu_sum += g
            if g > W*H:
                above_physical += 1
            # Independent conservative AABB with +1 pixel rounding tolerance.
            broad_upper, _ = geometric_tile_count(
                float(xy[0]), float(xy[1]), float(rr[0])+1., float(rr[1])+1.,
                None,None,None,None,W,H,T,alpha,extend,mask)
            if broad_upper is not None and g > broad_upper:
                above_aabb += 1
            if count is None:
                invalid[reason] = invalid.get(reason,0)+1
                continue
            valid += 1; oracle_sum += count
            difference=g-count
            if difference:
                diff_count += 1
                if first is None:
                    first={'camera':camera,'sample_index':int(j),'source_gaussian_id':int(source_id),
                        'gpu_count':g,'oracle_count':count,'difference':difference,
                        'mean2d':xy.tolist(),'radii':rr.tolist(),
                        'conics':ab if conics is None else np.asarray(ab).tolist(), 'opacity':opacity,
                        'depth':float(depths[camera,j])}
            rows.append({'camera':camera,'source_gaussian_id':int(source_id),'gpu_count':g,
                         'reference_geometry_count':count,'diff':difference,'projected_x':float(xy[0]),
                         'projected_y':float(xy[1]),'radius_x':float(rr[0]),'radius_y':float(rr[1]),
                         'opacity':opacity, 'depth':float(depths[camera,j]),'aabb_upper_bound_padded':broad_upper})
    with (outdir/'count_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['camera','source_gaussian_id','gpu_count','reference_geometry_count','diff'])
        w.writeheader(); w.writerows(rows)
    summary={'capture':str(capture),'method':'independent exact continuous ellipse-versus-tile overlap; not byte-equivalent CUDA AccuTile',
             'alpha_threshold':alpha,'gaussian_extend':extend,'n_cameras':len(means),'sample_n_per_camera':len(sample_ids),
             'valid_pairs':valid,'invalid_or_skipped_pairs':invalid,'mismatch_pairs':diff_count,
             'count_above_physical_tiles':above_physical,'count_above_padded_aabb_upper_bound':above_aabb,
             'gpu_sum_valid_and_skipped':gpu_sum,'reference_sum_valid':oracle_sum,
             'gpu_sample_sum_all':int(np.asarray(gpu,dtype=np.int64).sum()),
             'first_mismatch':first,
             'verdict':('PHYSICAL_COUNT_BOUND_VIOLATED' if above_physical else 'AABB_COUNT_BOUND_VIOLATED' if above_aabb else
                        'ALL_COMPARABLE_COUNTS_EQUAL' if diff_count==0 and not invalid else
                        'GEOMETRIC_DISCREPANCY_REQUIRES_CUDA_SEMANTIC_AUDIT' if diff_count else 'INCOMPLETE_INVALID_PROJECTION')}
    (outdir/'oracle_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (outdir/'first_mismatch.json').write_text(json.dumps(first,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='first_mismatch'},indent=2,ensure_ascii=False))
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--source-root',type=Path,help='Reads the actual gsplat/cuda/include/Common.h')
    ap.add_argument('--alpha',type=float)
    ap.add_argument('--extend',type=float)
    a=ap.parse_args()
    alpha=a.alpha; extend=a.extend
    if a.source_root:
        header=a.source_root/'gsplat/cuda/include/Common.h'
        if header.exists():
            s=header.read_text(encoding='utf-8')
            if alpha is None: alpha=parse_value(s, 'ALPHA_THRESHOLD')
            if extend is None: extend=parse_value(s,'GAUSSIAN_EXTEND')
    if alpha is None or extend is None:
        ap.error('Actual GAUSSIAN_EXTEND / ALPHA_THRESHOLD not resolved from Common.h. Pass --alpha and --extend explicitly after checking compiled constants.')
    if alpha<=0 or extend<=0: ap.error('Constants must be positive')
    run_oracle(a.capture,a.out_dir,alpha,extend)

if __name__=='__main__':
    main()
