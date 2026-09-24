import torch, sys
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print('Keys:', list(ckpt.keys())[:20])
print('Has optimizer:', any('optim' in k.lower() for k in ckpt.keys()))
print('Iteration:', ckpt.get('iteration', 'N/A'))
print('N_gaussians:', ckpt.get('N_gaussians', 'N/A'))
for k in ckpt.keys():
    v = ckpt[k]
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}')
    elif isinstance(v, dict):
        print(f'  {k}: dict with keys {list(v.keys())[:5]}')
    else:
        print(f'  {k}: {type(v).__name__} = {v}')
