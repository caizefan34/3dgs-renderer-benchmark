#!/usr/bin/env python3
"""DPC0 frozen-protocol harness. Stages: cameras|baseline|determinism|cross|
correctness|directional|onestep|timing|all. EPIC-05 A100 only.
Optimizer-unit coords: gradient scaled by per-group lr from the snapshot optimizer.
"""
import sys, os, json, hashlib, importlib.util, math, datetime, time
import numpy as np
import torch
import sys as _sys; _sys.path.insert(0,"/root/epic05_n1"); import n1_core as C

DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")
CELLS = [("room",10000),("room",15000),("bicycle",10000),("bicycle",15000)]
GRAD_ORDER = C.PARAM_NAMES  # _xyz,_shs,_scaling,_rotation,_opacity
GROUP_LR = {"_xyz":"xyz","_shs":"shs","_scaling":"scaling","_rotation":"rotation","_opacity":"opacity"}

def nowz(): return datetime.datetime.utcnow().isoformat()+"Z"
def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def load_dpc():
    C.bootstrap_c0()
    exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
    spec = importlib.util.spec_from_file_location("dpc0_gaussian_render_inference_scene_cuda", DPC_SO)
    dpc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dpc)
    return exp, dpc, exp.higs_rasterize_backward

def get_lrs(model):
    lr = {}
    for g in model.optimizer.param_groups:
        lr[g["name"]] = float(g["lr"])
    return lr

def get_cameras(scene):
    ds = C.load_scene_dataset(scene)
    n = len(ds)
    eval_idx = sorted(range(0, n, max(1, n//10)))
    holdout = eval_idx[:8]
    eset = set(eval_idx)
    train_all = [i for i in range(n) if i not in eset]
    # 8 midpoint-stratified source cameras (contiguous strata, midpoint of each)
    source = []
    for s in range(8):
        lo = s*len(train_all)//8; hi = (s+1)*len(train_all)//8
        if hi <= lo: hi = min(lo+1, len(train_all))
        stratum = train_all[lo:hi]
        source.append(stratum[len(stratum)//2])
    # 3 deterministic peer-train (first train cams not in source)
    sset = set(source)
    peer = [i for i in train_all if i not in sset][:3]
    return ds, source, peer, holdout, n

def opt_vec(arrs, lrs):
    # optimizer-unit coords: each group's grad scaled by its lr, concatenated
    out = []
    for nm in GRAD_ORDER:
        a = arrs[nm].astype(np.float64).ravel()
        out.append(a * lrs[GROUP_LR[nm]])
    return np.concatenate(out)

def pair_metrics(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(np.dot(a,b)/(na*nb+1e-30))
    rel = float(np.linalg.norm(a-b)/(na+1e-30))
    return cos, rel

def render_loss(model, cam, gt, fused):
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    torch.cuda.synchronize()
    L = fused(img, gt)
    return L

def grads_of(L, params):
    g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    out = {}
    for nm, gr in zip(GRAD_ORDER, g):
        out[nm] = None if gr is None else gr.detach().to("cpu", torch.float32).numpy()
    return out

def hash_grads(arrs):
    h = hashlib.sha256()
    for nm in GRAD_ORDER:
        h.update(b"NULL" if arrs[nm] is None else np.ascontiguousarray(arrs[nm]).tobytes())
    return h.hexdigest()

# ---------------------------------------------------------------- stages
def stage_cameras(scene):
    ds, source, peer, holdout, n = get_cameras(scene)
    return {"scene":scene,"n_cameras":n,"source":source,"peer":peer,"holdout":holdout}

def run_cell(scene, it, gpu, exp, dpc, orig_bwd, stage, n_det=32, n_ens=8):
    torch.cuda.set_device(gpu); torch.manual_seed(0)
    ckpt = C.load_checkpoint(scene, it); model = C.build_model(ckpt); C.scene_reset()
    ds, source, peer, holdout, n = get_cameras(scene)
    lrs = get_lrs(model)
    params = [getattr(model, nm) for nm in GRAD_ORDER]
    fused = C.make_fused_loss()
    res = {"cell":f"{scene}@{it}","gpu":gpu,"lrs":lrs,"n_cameras":n,
           "source":source,"peer":peer,"holdout":holdout}
    cam_objs = {i: ds.get_item(i) for i in set(source)|set(peer)|set(holdout)}

    if stage in ("baseline","all"):
        # DPC-B0: native C0 backward 8x per source -> noise diagnostic
        b0 = {}
        for ci in source:
            cam, gt = cam_objs[ci]
            hs = []
            exp.higs_rasterize_backward = orig_bwd
            for r in range(n_ens):
                L = render_loss(model, cam, gt, fused)
                g = grads_of(L, params)
                for nm in GRAD_ORDER:
                    getattr(model,nm).grad = None if g[nm] is None else torch.from_numpy(g[nm]).cuda()
                hs.append(hash_grads(g))
                torch.cuda.synchronize()
            u = sorted(set(hs))
            # pairwise relL2 across the 8 samples (concatenated optimizer-unit)
            vs = [opt_vec(g, lrs) for g in [grads_of(render_loss(model,cam,gt,fused), params) for _ in range(1)]]
            b0[str(ci)] = {"n_repeat":n_ens,"n_unique":len(u),"unique":len(u)==1,"hashes":hs}
        res["baseline"] = b0

    if stage in ("determinism","all"):
        det = {}
        exp.higs_rasterize_backward = dpc.higs_rasterize_backward
        for ci in source:
            cam, gt = cam_objs[ci]
            L = render_loss(model, cam, gt, fused)  # render ONCE (fixed inputs)
            torch.cuda.synchronize()
            hs = []
            for r in range(n_det):
                g = grads_of(L, params)
                for nm in GRAD_ORDER:
                    getattr(model,nm).grad = None if g[nm] is None else torch.from_numpy(g[nm]).cuda()
                hs.append(hash_grads(g))
                torch.cuda.synchronize()
            u = sorted(set(hs))
            det[str(ci)] = {"n_repeat":n_det,"n_unique":len(u),"deterministic":len(u)==1,"hash":hs[0]}
        res["determinism"] = det

    if stage in ("correctness","all"):
        corr = {}
        exp.higs_rasterize_backward = orig_bwd
        for ci in source:
            cam, gt = cam_objs[ci]
            # C0 ensemble: 8 independent native C0 backward (fresh render each)
            ens = []
            for r in range(n_ens):
                L = render_loss(model, cam, gt, fused)
                g = grads_of(L, params)
                for nm in GRAD_ORDER:
                    getattr(model,nm).grad = None if g[nm] is None else torch.from_numpy(g[nm]).cuda()
                ens.append({nm:(g[nm].astype(np.float64) if g[nm] is not None else None) for nm in GRAD_ORDER})
                torch.cuda.synchronize()
            # fp64 mean
            mean = {}
            for nm in GRAD_ORDER:
                stack = np.stack([e[nm] for e in ens if e[nm] is not None])
                mean[nm] = stack.mean(axis=0)
            # DPC: render once, backward (deterministic)
            exp.higs_rasterize_backward = dpc.higs_rasterize_backward
            L = render_loss(model, cam, gt, fused)
            gd = grads_of(L, params)
            torch.cuda.synchronize()
            exp.higs_rasterize_backward = orig_bwd
            # compare in optimizer-unit coords
            a = opt_vec(gd, lrs); b = opt_vec(mean, lrs)
            cos, rel = pair_metrics(a, b)
            corr[str(ci)] = {"cosine":cos,"relL2":rel,
                             "c0_ensemble_norm":float(np.linalg.norm(b)),
                             "dpc_norm":float(np.linalg.norm(a))}
        res["correctness"] = corr
    return res

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv)>1 else "cameras"
    scene = sys.argv[2] if len(sys.argv)>2 else "room"
    it = int(sys.argv[3]) if len(sys.argv)>3 else 10000
    gpu = int(sys.argv[4]) if len(sys.argv)>4 else 0
    exp, dpc, orig_bwd = load_dpc()
    if stage == "cameras":
        print(json.dumps(stage_cameras(scene), indent=2))
    else:
        r = run_cell(scene, it, gpu, exp, dpc, orig_bwd, stage)
        print(json.dumps(r, indent=2, default=str))
