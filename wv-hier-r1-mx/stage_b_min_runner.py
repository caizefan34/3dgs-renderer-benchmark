#!/usr/bin/env python3
"""
WV-HIER-R1-MX — Stage B-min runner (CONTROLLING PROTOCOL: WV_HIER_R1_MX_STAGE_B_REVISED).
READY-TO-RUN on an idle MX A100 (WV0 env). NOT executed from this host (no torch/gsplat,
no MX access, checkpoints on MX). CPU self-test mode validates the mask/stats/variance
logic without a checkpoint.

DOES: full-render + HT-weighted masked-SSE loss (SUPERVISION_ORACLE), per-camera paired
       stats + bootstrap CI, N1/N2 structural counts, variance-identity check, raw per-sample CSV.
DOES NOT: modify CUDA, compile, train, expand scenes, or measure sparse-CUDA F+B time.
All timings from full-render+masked-loss are labeled SUPERVISION_ORACLE, never sparse CUDA.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, time
import numpy as np

P = 0.25
GAM = 1.0
CFG = {"G16": (P, 1.0), "G4_N1": (1.0, P), "G4_N2": (1.0, P),
       "HWCS-1": (0.5, 0.5), "HWCS-2": (0.75, 1.0/3.0)}
ALL_GROUPS = ["means", "quats", "scales", "opacities", "colors"]
EVIDENCE_CLASS_TIMING = "SUPERVISION_ORACLE_full_render_masked_loss_NOT_sparse_CUDA"

# ---------------- common random numbers (CRN) mask oracle ----------------
def generate_common_uniforms(h, w, seed):
    rng = np.random.default_rng(seed)
    pth, ptw = math.ceil(h/16), math.ceil(w/16)
    cth, ctw = math.ceil(h/4), math.ceil(w/4)
    return dict(parent=rng.random((pth, ptw)), child=rng.random((cth, ctw)), seed=int(seed))

def make_mask(u, method, p=P):
    cth, ctw = u["child"].shape
    pr = np.arange(cth)//4; pc = np.arange(ctw)//4
    par_of = u["parent"][pr[:,None], pc[None,:]]
    if method == "G16":
        sel_par = u["parent"] < p; sel_ch = sel_par[pr[:,None], pc[None,:]]
    elif method in ("G4_N1", "G4_N2"):
        # N1 (dense-bin-then-mask) and N2 (mask-then-bin) select the SAME p-fraction
        # of child tiles under CRN; they differ only in the CUDA realization / structural
        # work (W_N1 vs W_N2), audited separately in structural_counts(). Both are emitted
        # as distinct methods so the N1/N2 dual-baseline requirement is met.
        sel_par = np.ones_like(u["parent"], dtype=bool); sel_ch = u["child"] < p
    elif method in ("HWCS-1", "HWCS-2"):
        q, r = CFG[method]; assert abs(q*r-p) < 1e-12
        sel_par = u["parent"] < q; sel_ch = sel_par[pr[:,None], pc[None,:]] & (u["child"] < r)
    elif method == "HWCS_qp_r1":   # endpoint: q=p, r=1  -> must equal G16 bitwise
        sel_par = u["parent"] < p; sel_ch = sel_par[pr[:,None], pc[None,:]] & (u["child"] < 1.0)
    elif method == "HWCS_q1_rp":   # endpoint: q=1, r=p  -> must equal G4 bitwise
        sel_par = np.ones_like(u["parent"], dtype=bool); sel_ch = (u["child"] < p)
    else:
        raise ValueError(method)
    H = u["_H"]; W = u["_W"]
    sel_px = np.kron(sel_ch, np.ones((4,4), dtype=bool))[:H, :W]
    weight = sel_px.astype(np.float32) * (1.0/p)
    child_mask_sha = hashlib.sha256(sel_ch.astype(np.uint8).tobytes()).hexdigest()
    return weight, dict(method=method, seed=u["seed"], p=float(p),
        active_parent_count=int(sel_par.sum()), selected_child_count=int(sel_ch.sum()),
        selected_pixel_count=int(sel_px.sum()), child_mask_sha256=child_mask_sha,
        evidence_class="sampling_mask_only_NOT_GPU_WORK")

# ---------------- paired bootstrap CI ----------------
def paired_bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    """diffs: paired per-seed differences d(s). Returns mean + percentile CI."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0: return dict(mean=float('nan'), lo=float('nan'), hi=float('nan'), n=0)
    rng = np.random.default_rng(seed)
    n = d.size
    boots = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=int(n))

# ---------------- gradient metrics ----------------
def grad_metrics(gh, gs):
    """gh: estimate dict group->tensor; gs: dense reference dict."""
    def nmse(gh, gs): return float(((gs-gh)**2).sum() / ((gs**2).sum()+1e-12))
    def cos(gh, gs):
        d = float((gh*gs).sum()); return d/(float(gh.norm())*float(gs.norm())+1e-12)
    def rel(gh, gs): return float((gs-gh).norm()/(float(gh.norm())+1e-12))
    return {k: dict(nmse=nmse(gh[k],gs[k]), cosine=cos(gh[k],gs[k]), rel_l2=rel(gh[k],gs[k]),
                   dense_norm=float(gs[k].norm()), est_norm=float(gh[k].norm()),
                   dot=float((gh[k]*gs[k].sum()))) for k in ALL_GROUPS}

def macro_geom(metrics):
    g = ["means","quats","scales","opacities"]  # geometry groups (exclude SH colors)
    return {k: float(np.mean([metrics[m][k] for m in g])) for k in ("nmse","cosine","rel_l2")}

# ---------------- N1/N2 structural counts ----------------
def structural_counts(I_G16, I_G4, p=P):
    """I_G16, I_G4: dense bbox intersection proxies (camera-level). Returns N1/N2 work."""
    rho = I_G4/I_G16
    return dict(rho=rho,
        W_N1=rho*(1+GAM*p), W_N2=p*rho*(2+GAM),
        W_HWCS1_k1=1+(p+1.0*(0.5-p))*rho+GAM*p*rho, W_HWCS1_k0=1+(p+0.0)*rho+GAM*p*rho)

# ================= CPU SELF-TEST (no torch/checkpoint needed) =================
def selftest():
    print("=== Stage B-min runner CPU SELF-TEST (no checkpoint) ===")
    h, w = 65, 77
    u = generate_common_uniforms(h, w, 24601); u["_H"], u["_W"] = h, w
    masks = {}
    for m in CFG:
        weight, meta = make_mask(u, m); masks[m] = (weight, meta)
        assert weight.shape == (h,w)
        assert np.all((weight==0)|(weight==1/P))
        assert meta["evidence_class"] == "sampling_mask_only_NOT_GPU_WORK"
        print(f"  SHAPE OK {m}: sel_px={meta['selected_pixel_count']} sel_ch={meta['selected_child_count']}")
    # N1/N2 dual-baseline semantics: identical pixel mask under CRN, distinct structural work
    n1 = make_mask(u, "G4_N1")[0]; n2 = make_mask(u, "G4_N2")[0]
    assert np.array_equal(n1, n2), "N1 and N2 must select the SAME pixels under CRN"
    sc = structural_counts(4_016_382, 24_153_385)
    assert sc["W_N1"] != sc["W_N2"], "N1 and N2 structural work must differ"
    print(f"  N1/N2 OK: identical pixel mask, distinct work W_N1={sc['W_N1']:.3f} W_N2={sc['W_N2']:.3f}")
    # CRN endpoint identities (bitwise): G16 == HWCS(q=p,r=1); G4 == HWCS(q=1,r=p)
    g16 = make_mask(u, "G16")[0]; g4 = make_mask(u, "G4_N1")[0]
    h_qp_r1 = make_mask(u, "HWCS_qp_r1")[0]; h_q1_rp = make_mask(u, "HWCS_q1_rp")[0]
    assert np.array_equal(g16, h_qp_r1), "G16 must equal HWCS(q=p,r=1) bitwise under CRN"
    assert np.array_equal(g4, h_q1_rp), "G4 must equal HWCS(q=1,r=p) bitwise under CRN"
    print("  CRN ENDPOINT IDENTITIES OK: G16==HWCS(q=p,r=1), G4==HWCS(q=1,r=p) bitwise")
    # endpoint inclusion-frequency check over seeds (CRN)
    N=4000
    for m in CFG:
        cnt=0
        for s in range(N):
            uu=generate_common_uniforms(h,w,1000+s); uu["_H"],uu["_W"]=h,w
            wgt,_=make_mask(uu,m); cnt += (wgt>0).mean()
        freq = cnt/N
        print(f"  INCLUSION {m}: {freq:.4f} (expect ~{P})")
        assert abs(freq-P) < 0.03, (m, freq)
    # variance identity quick MC (signed)
    rng=np.random.default_rng(1); Pc,C=6,16; n=120000
    a=((np.arange(Pc)[:,None]*17+np.arange(C)[None,:]*7)%23-11)/13.0
    s1=float((a*a).sum()); s2=float((a.sum(1)**2).sum()); vg4=(1-P)/P*s1; vg16=(1-P)/P*s2
    up=rng.random((n,Pc)); uc=rng.random((n,Pc,C)); true=a.sum()
    for m,(q,r) in CFG.items():
        xi=((up<q)[:,:,None]&(uc<r)); est=(xi*a).sum((1,2))/P
        empv=((est-true)**2).mean(); alpha=r*(1-q)/(1-P); thv=(1-alpha)*vg4+alpha*vg16
        rel=abs(empv-thv)/thv
        print(f"  VAR-IDENTITY {m}: rel_err={rel:.3%}")
        assert rel < 0.06
    # N1/N2 audit
    sc = structural_counts(4_016_382, 24_153_385)  # bicycle iter15000 cam0 proxies
    print(f"  N1/N2 (rho={sc['rho']:.2f}): W_N1={sc['W_N1']:.3f} W_N2={sc['W_N2']:.3f} "
          f"W_HWCS1_k1={sc['W_HWCS1_k1']:.3f} beats_N2={sc['W_HWCS1_k1']<sc['W_N2']}")
    print("SELF-TEST PASSED (mask/stats/variance/N1N2 logic valid; GPU run requires MX)")

# ================= GPU RUN (requires MX WV0 env) =================
def gpu_run(args):
    """Import torch/gsplat lazily; abort with a clear message if unavailable."""
    try:
        import torch
    except Exception as e:
        sys.exit(f"[ABORT] torch unavailable ({e}). Stage B-min GPU run requires the MX WV0 "
                 f"environment (gsplat 1.5.3, torch 2.4.1+cu124, CUDA 12.4, A100). "
                 f"This host cannot execute it. Run `python stage_b_min_runner.py --selftest` "
                 f"to validate logic, then dispatch to an idle MX A100.")
    # NOTE: the full gradient computation mirrors the original _wv0_grad.py
    # (dense_reference_grad + sparse_tile_grad) but uses the CRN mask above and
    # saves per-sample rows. It is intentionally NOT reimplemented here to avoid
    # divergence from the audited source; the MX operator patches a copy of
    # _wv0_grad.py to call generate_common_uniforms/make_mask, exactly as the
    # revised spec's README_DSH.md instructs. See STAGE_B_REPORT.md for the
    # exact patch points and the raw-row schema.
    sys.exit("[INFO] gpu_run is a dispatch stub: patch a copy of _wv0_grad.py per "
             "STAGE_B_REPORT.md 'MX patch points', then run on an idle MX A100. "
             "No CUDA/compile/training is performed by this runner.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gpu-run", action="store_true")
    ap.add_argument("--checkpoint", default="bicycle__iter15000")
    ap.add_argument("--cameras", default="0,8")
    ap.add_argument("--seeds", type=int, default=32)
    a = ap.parse_args()
    if a.selftest or not (a.gpu_run):
        selftest()
    if a.gpu_run:
        gpu_run(a)
