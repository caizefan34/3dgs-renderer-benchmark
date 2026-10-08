import os, inspect
os.environ.setdefault('TMPDIR', '/mnt/storage_pool/liaoyuanjun/mss0_mx/tmp')
import torch, gsplat
print('gsplat', gsplat.__version__)
print('attrs:', [a for a in dir(gsplat) if not a.startswith('_')])
for name in ['rasterization','fully_fused_projection','spherical_harmonics','fully_fused_projection_with_bg','fully_fused_projection_grad','isect_tiles','estimate_normal']:
    fn = getattr(gsplat, name, None)
    if fn is None:
        print('MISSING', name); continue
    try:
        print('SIG', name, str(inspect.signature(fn)))
    except (TypeError, ValueError):
        try:
            print('SIG', name, str(fn.__doc__[:200]))
        except Exception:
            print('SIG', name, '<builtin>')
try:
    from gsplat import rendering as _r
    print('rendering attrs:', [a for a in dir(_r) if not a.startswith('_')][:30])
except Exception as e:
    print('no rendering submodule', e)
ckpt = '/mnt/storage_pool/liaoyuanjun/retiregs/RETIRE0_room_source/checkpoints/iter_15000.pt'
obj = torch.load(ckpt, map_location='cpu', weights_only=False)
print('ckpt type', type(obj))
def show(d, prefix=''):
    for k, v in d.items():
        if torch.is_tensor(v):
            print('T', prefix+k, tuple(v.shape), v.dtype)
        elif isinstance(v, dict):
            print('D', prefix+k, 'subkeys=', list(v.keys())[:25])
            if prefix.count('.') < 2:
                show(v, prefix+k+'.')
        else:
            print('O', prefix+k, type(v).__name__, str(v)[:80])
if isinstance(obj, dict):
    show(obj)
elif isinstance(obj, (list, tuple)):
    print('seq len', len(obj))
    for i, v in enumerate(obj[:3]):
        print(i, type(v), (tuple(v.shape), v.dtype) if torch.is_tensor(v) else str(v)[:80])
