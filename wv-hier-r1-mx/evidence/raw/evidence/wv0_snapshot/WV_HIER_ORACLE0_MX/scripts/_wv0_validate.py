import sys, os, json, time
sys.path.insert(0, "/dev/shm")
import numpy as np, torch
from _wv0_core import (load_cameras, load_state_params, make_grad_enabled,
                       STATE_PATHS, RV1, ALL_GROUPS, PARAM_GROUPS)
from _wv0_grad import dense_reference_grad, sparse_tile_grad

scene = sys.argv[1] if len(sys.argv)>1 else "room"
state_id = sys.argv[2] if len(sys.argv)>2 else "room__iter5000"
res = sys.argv[3] if len(sys.argv)>3 else "540p"
nprobe = int(sys.argv[4]) if len(sys.argv)>4 else 4

rel, scn, it = STATE_PATHS[state_id]
ckpt = os.path.join(RV1, rel)
print("state", state_id, "ckpt", ckpt, "exists", os.path.exists(ckpt))
cams = load_cameras(scene, resolution=res)
print("cams", len(cams), "first", cams[0]["W"], cams[0]["W"] and cams[0]["H"], "has_gt", "gt" in cams[0])
raw, act = load_state_params(ckpt)
print("N", act["N"], "active_sh", act["active_sh"])

# aggregate over nprobe cameras
cam_subset = cams[:nprobe]
t0=time.time()
# dense
dense_agg = {k: 0.0 for k in ALL_GROUPS}  # accumulate relative L2 later; here store sum of grads
dense_norms = {k: 0.0 for k in ALL_GROUPS}
dense_grads_per_cam = []
for ci, cam in enumerate(cam_subset):
    g = make_grad_enabled(act)
    gd, render, alpha, meta, loss = dense_reference_grad(g, cam, act["active_sh"])
    dense_grads_per_cam.append(gd)
    for k in ALL_GROUPS:
        dense_norms[k] += float(gd[k].norm())
print("dense done in", round(time.time()-t0,1),"s; loss", round(loss,4))
print("dense grad norms (sum over cams):", {k: round(v,4) for k,v in dense_norms.items()})

# sparse estimator averaged over masks for G=4, p=0.25
for G in [4, 1]:
    pi = 0.25
    sparse_agg = {k: 0.0 for k in ALL_GROUPS}
    NM = 32
    for mi in range(NM):
        for ci, cam in enumerate(cam_subset):
            g = make_grad_enabled(act)
            gs, render, alpha, meta, counters, sloss, mask = sparse_tile_grad(g, cam, act["active_sh"], G, seed=1000+mi, pi=pi)
            for k in ALL_GROUPS:
                sparse_agg[k] += float(gs[k].norm())/ (NM*len(cam_subset))
    print(f"G={G} p={pi}: mean ||ghat|| per group (avg over cams&masks):", {k: round(sparse_agg[k],5) for k in ALL_GROUPS})
print("dense mean ||g*|| per group:", {k: round(dense_norms[k]/len(cam_subset),5) for k in ALL_GROUPS})
