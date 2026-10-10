import sys, os
sys.path.insert(0,"/dev/shm")
import torch
from _wv0_core import load_cameras, load_state_params, make_grad_enabled, STATE_PATHS, RV1
from _wv0_grad import dense_reference_grad
rel, scene, it = STATE_PATHS["room__iter5000"]
raw, act = load_state_params(os.path.join(RV1, rel))
cams = load_cameras(scene, resolution="540p")[:1]
g = make_grad_enabled(act)
gd, render, alpha, meta, loss = dense_reference_grad(g, cams[0], act["active_sh"])
for k,v in meta.items():
    if torch.is_tensor(v):
        print(k, tuple(v.shape), v.dtype)
    else:
        print(k, type(v).__name__, v if not isinstance(v,(list,dict)) else f"len={len(v)}")
