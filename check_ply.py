#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')
from plyfile import PlyData

ply_path = sys.argv[1]
ply = PlyData.read(ply_path)
vertex = ply['vertex']
print(f"N={len(vertex)}")
print(f"Properties: {[p.name for p in vertex.properties]}")
