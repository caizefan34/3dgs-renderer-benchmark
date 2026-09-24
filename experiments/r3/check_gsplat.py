"""Check if gsplat has pure torch fallback for key functions."""
import gsplat, os

base = os.path.dirname(gsplat.__file__)

# Check torch_impl
path = os.path.join(base, 'cuda', '_torch_impl.py')
with open(path) as f:
    s = f.read()
print(f'_torch_impl.py size: {len(s)} chars')
for fn in ['fully_fused_projection', 'isect_tiles', 'rasterize_to_pixels']:
    count = s.count(fn)
    print(f"  {fn}: {count} matches")

# Check _lazy_backend
path2 = os.path.join(base, '_lazy_backend.py')
with open(path2) as f:
    lb = f.read()
print(f'_lazy_backend.py size: {len(lb)} chars')
print(f"  references torch_impl: {'_torch_impl' in lb}")
print(f"  has _C cuda: {'_C' in lb}")
