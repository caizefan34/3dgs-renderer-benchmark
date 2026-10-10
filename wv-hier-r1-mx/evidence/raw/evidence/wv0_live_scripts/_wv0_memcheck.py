import sys, os, time
sys.path.insert(0,"/dev/shm")
import torch
from _wv0_core import load_cameras, load_state_params, make_grad_enabled, STATE_PATHS, RV1
from _wv0_grad import dense_reference_grad
for sid in ["bicycle__iter5000","garden__iter5000"]:
    rel,scene,it = STATE_PATHS[sid]
    raw,act = load_state_params(os.path.join(RV1,rel))
    cams = load_cameras(scene, resolution="540p")[:16]
    print(f"{sid}: N={act['N']:,} cams={len(cams)} hw={cams[0]['H']}x{cams[0]['W']}", flush=True)
    g = make_grad_enabled(act)
    t0=time.time()
    gd, render, alpha, meta, loss = dense_reference_grad(g, cams[0], act["active_sh"])
    print(f"  dense render+grad cam0 OK loss={loss:.1f} {round(time.time()-t0,1)}s "
          f"radii>0={int((meta['radii'][0,:,0]>0).sum()):,}", flush=True)
    del g, gd, render, alpha, meta; torch.cuda.synchronize()
