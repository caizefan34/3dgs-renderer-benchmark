import gsplat, os
d = os.path.dirname(gsplat.__file__)
p = os.path.join(d, 'cuda', 'csrc', 'IntersectTile.cu')
c = open(p).read()
print('gsplat version:', gsplat.__version__)
print('C17-0 in source:', 'tile_segmented_sort_double_buffer' in c)
print('Restored to baseline:', 'tile_segmented_sort_double_buffer' not in c)
