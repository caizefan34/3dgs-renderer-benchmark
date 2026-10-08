#!/usr/bin/env python3
"""Find gsplat IntersectTile.cu path and show current key structure."""
import gsplat, os
p = gsplat.__path__[0]
cu = os.path.join(p, "cuda", "csrc", "IntersectTile.cu")
print("IntersectTile.cu:", cu)
print("Exists:", os.path.exists(cu))
with open(cu) as f:
    lines = f.readlines()
# Show key construction lines
for i, line in enumerate(lines):
    if any(x in line for x in ['iid_enc', 'depth_id_enc', 'depth_upper', 'isect_ids[', 'end_bit', '>> 32', '>> 16', '<< 32', '<< 16']):
        print(f"  L{i+1}: {line.rstrip()}")
print("Total lines:", len(lines))
