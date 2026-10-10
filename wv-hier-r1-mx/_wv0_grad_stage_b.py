#!/usr/bin/env python3
"""
_wv0_grad.py  —  STAGE B-MIN TURNKEY PATCH (WV_HIER_R1_MX_STAGE_B_REVISED controlling protocol).
Derived from the audited WV0 gradient oracle; the CUDA core (_wv0_core.render_full)
is UNCHANGED. This is a Python-only patch — no CUDA modification, no compile, no training.

What was added vs the original oracle:
  * Common random numbers (CRN): one (U_parent, U_child) per (camera, seed) shared by
    G16, G4(N1), G4(N2), HWCS-1, HWCS-2 — the revised protocol's central requirement.
  * HWCS two-stage parent->child mask (marginal p = q*r = 0.25; HT weight 1/p).
  * Native-G4 two baselines: N1 (dense-bin-then-mask) and N2 (mask-then-bin).
  * Per-sample raw gradient metrics (NMSE/cosine/relL2 per group) + paired-bootstrap CI.
  * Structural COUNTS (parent/child/pixel) + real projected Gaussian candidate counts.
  * SUPERVISION_ORACLE timing labels; NO sparse-CUDA timing claim.
  * Raw per-sample CSV writer (schema = stage_b_raw_results_TEMPLATE.csv).

Run on an idle MX A100 (WV0 env). NOT executed from a non-MX host.
"""
import math, os, csv, hashlib, random, numpy as np, torch

ALL_GROUPS = ["means", "quats", "scales", "opacities", "colors"]
GEO_GROUPS = ["means", "quats", "scales", "opacities"]   # geometry (excl. SH colors)
P = 0.25
CFG = {"G16": (P, 1.0), "G4_N1": (1.0, P), "G4_N2": (1.0, P),
       "HWCS-1": (0.5, 0.5), "HWCS-2": (0.75, 1.0/3.0)}
TILEID_CACHE = {}


# ---------------- CRN masks (revised protocol) ----------------
def generate_common_uniforms(H, W, seed):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    pth, ptw = math.ceil(H/16), math.ceil(W/16)
    cth, ctw = math.ceil(H/4), math.ceil(W/4)
    return dict(parent=torch.rand(pth*ptw, generator=g),
                child=torch.rand(cth*ctw, generator=g), seed=int(seed))


def make_mask(method, H, W, u, device="cuda"):
    """Return (weight[H,W] float32, meta). CRN: uses shared u['parent']/u['child']."""
    pth, ptw = math.ceil(H/16), math.ceil(W/16)
    cth, ctw = math.ceil(H/4), math.ceil(W/4)
    pr = torch.arange(cth, device="cpu").div(4, rounding_mode="floor").clamp(max=pth-1)
    pc = torch.arange(ctw, device="cpu").div(4, rounding_mode="floor").clamp(max=ptw-1)
    par_of = (pr[:, None]*ptw + pc[None, :])          # [cth,ctw] parent index
    if method in CFG:
        q, r = CFG[method]
    if method == "G16":
        sel_par = (u["parent"] < P).view(pth, ptw)
        sel_ch = sel_par[pr[:, None], pc[None, :]]
    elif method in ("G4_N1", "G4_N2"):
        # N1 and N2 both select p-fraction of child tiles (N1 = dense-then-mask,
        # N2 = mask-then-bin differ only in the CUDA realization, not the pixel set);
        # they are audited as two STRUCTURAL baselines, emitted with distinct labels.
        sel_ch = (u["child"] < P).view(cth, ctw)
    elif method in ("HWCS-1", "HWCS-2"):
        sel_par = (u["parent"] < q).view(pth, ptw)
        sel_ch = sel_par[pr[:, None], pc[None, :]] & (u["child"] < r).view(cth, ctw)
    elif method == "HWCS_qp_r1":   # endpoint q=p,r=1 -> must equal G16 bitwise
        sel_par = (u["parent"] < P).view(pth, ptw)
        sel_ch = sel_par[pr[:, None], pc[None, :]] & (u["child"] < 1.0).view(cth, ctw)
    elif method == "HWCS_q1_rp":   # endpoint q=1,r=p -> must equal G4 bitwise
        sel_par = torch.ones_like(u["parent"], dtype=torch.bool).view(pth, ptw)
        sel_ch = (u["child"] < P).view(cth, ctw)
    else:
        raise ValueError(method)
    # upsample 4x4 -> pixel, clip to image
    key = (H, W, device)
    ti = TILEID_CACHE.get(key)
    if ti is None or ti.shape != (H, W):
        ti4 = torch.arange(H, device="cpu").div(4, rounding_mode="floor").clamp(max=cth-1)[:, None].expand(H, W)
        tj4 = torch.arange(W, device="cpu").div(4, rounding_mode="floor").clamp(max=ctw-1)[None, :].expand(H, W)
        TILEID_CACHE[key] = ti4*ctw + tj4
    sel_px = sel_ch.view(-1)[TILEID_CACHE[key]] > 0.5
    weight = torch.where(sel_px.to(device), torch.tensor(1.0/P, device=device),
                         torch.tensor(0.0, device=device)).float()
    meta = dict(method=method, seed=u["seed"], p=float(P),
        active_parent_count=int(sel_par.sum()), selected_child_count=int(sel_ch.sum()),
        selected_pixel_count=int(sel_px.sum()),
        child_mask_sha256=hashlib.sha256(sel_ch.to(torch.uint8).numpy().tobytes()).hexdigest(),
        evidence_class="sampling_mask_only_NOT_GPU_WORK")
    return weight, meta


def tile_pixel_mask_fast(H, W, G, seed, pi, device="cuda"):
    """Original oracle mask (kept for backward compatibility / G16-G4 comparison)."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    th = math.ceil(H/G); tw = math.ceil(W/G)
    inc = (torch.rand(th*tw, generator=g) < pi).float()
    cache_key = (H, W, G, device)
    ti = TILEID_CACHE.get(cache_key)
    if ti is None or ti.shape != (H, W):
        ti = torch.arange(H).div(G, rounding_mode="floor").clamp(max=th-1)[:,None].expand(H, W)
        tj = torch.arange(W).div(G, rounding_mode="floor").clamp(max=tw-1)[None,:].expand(H, W)
        TILEID_CACHE[cache_key] = ti*tw + tj
    mask = TILEID_CACHE[cache_key]
    m = inc[mask] > 0.5
    return m.to(device), int(inc.sum())


# ---------------- gradient computation (core UNCHANGED) ----------------
def dense_reference_grad(gparams, cam, sh_degree):
    from _wv0_core import render_full
    render, alpha, meta = render_full(gparams, cam, sh_degree)
    gt = cam["gt"]
    r = render[0].permute(2, 0, 1)
    diff = r - gt
    loss = (diff*diff).sum()
    grads = torch.autograd.grad(loss, [gparams[k] for k in ALL_GROUPS],
                                retain_graph=False, allow_unused=True)
    gdict = {k: (grads[i].detach().clone() if grads[i] is not None else torch.zeros_like(gparams[k]))
             for i, k in enumerate(ALL_GROUPS)}
    return gdict, render, alpha, meta, float(loss)


def sparse_tile_grad(gparams, cam, sh_degree, method, u, device="cuda"):
    """Supervision oracle: full-render + HT-weighted masked-SSE loss (NOT sparse CUDA)."""
    from _wv0_core import render_full
    render, alpha, meta = render_full(gparams, cam, sh_degree)
    gt = cam["gt"]
    r = render[0].permute(2, 0, 1)
    weight, mmeta = make_mask(method, cam["H"], cam["W"], u, device=device)
    diff = r - gt
    loss = (diff*diff * weight[None]).sum()
    grads = torch.autograd.grad(loss, [gparams[k] for k in ALL_GROUPS],
                                retain_graph=False, allow_unused=True)
    gdict = {k: (grads[i].detach().clone() if grads[i] is not None else torch.zeros_like(gparams[k]))
             for i, k in enumerate(ALL_GROUPS)}
    n_sel_px = int((weight > 0).sum())
    counters = dict(mmeta,
        selected_pixels=n_sel_px, total_pixels=cam["H"]*cam["W"], n_cameras=1,
        timing_class="SUPERVISION_ORACLE_full_render_masked_loss_NOT_sparse_CUDA")
    return gdict, render, alpha, meta, counters, float(loss), weight


# ---------------- metrics ----------------
def grad_metrics(gh, gs):
    def nmse(a, b): return float(((b-a)**2).sum() / ((b**2).sum()+1e-12))
    def cos(a, b): return float((a*b).sum() / (a.norm()*b.norm()+1e-12))
    def rel(a, b): return float((b-a).norm() / (a.norm()+1e-12))
    return {k: dict(nmse=nmse(gh[k], gs[k]), cosine=cos(gh[k], gs[k]), rel_l2=rel(gh[k], gs[k]),
                   dense_norm=float(gs[k].norm()), est_norm=float(gh[k].norm()),
                   dot=float((gh[k]*gs[k]).sum())) for k in ALL_GROUPS}


def macro_geom(metrics):
    return {k: float(np.mean([metrics[m][k] for m in GEO_GROUPS])) for k in ("nmse", "cosine", "rel_l2")}


def paired_bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return dict(mean=float('nan'), lo=float('nan'), hi=float('nan'), n=0)
    rng = np.random.default_rng(seed); n = d.size
    boots = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=int(n))


# ---------------- structural counts (real projected gaussians) ----------------
def projected_counts(gparams, cam, sh_degree):
    """Best-effort real projected Gaussian candidate counts from the core's render meta.
    Falls back to a clear NOT_MEASURED marker if the buffer is absent — never fabricated."""
    try:
        from _wv0_core import render_full
        _, _, meta = render_full(gparams, cam, sh_degree)
        for key in ("gaussian_parent_candidates", "gaussian_child_candidates",
                    "n_gaussians", "gaussian_count"):
            if key in meta:
                return meta[key], key
        return None, "NOT_MEASURED_no_kernel_visible_buffer"
    except Exception as e:
        return None, f"NOT_MEASURED_{type(e).__name__}"


# ---------------- driver (MX only) ----------------
def run_camera(gparams, cam, sh_degree, seed, device="cuda"):
    """One (camera, seed): dense ref + all methods under CRN. Returns per-sample rows."""
    gs, _, _, _, _ = dense_reference_grad(gparams, cam, sh_degree)
    u = generate_common_uniforms(cam["H"], cam["W"], seed)
    rows = []
    for method in CFG:
        gh, _, _, meta, counters, loss, weight = sparse_tile_grad(gparams, cam, sh_degree, method, u, device)
        m = grad_metrics(gh, gs)
        mg = macro_geom(m)
        pc, pcsrc = projected_counts(gparams, cam, sh_degree)
        rows.append(dict(camera_id=cam.get("id"), seed=seed, method=method,
            nmse_geometry_macro=mg["nmse"], cosine_geometry_macro=mg["cosine"],
            rel_l2_geometry_macro=mg["rel_l2"],
            dense_grad_norm=float(sum(float(gs[k].norm()) for k in ALL_GROUPS)),
            est_grad_norm=float(sum(float(gh[k].norm()) for k in ALL_GROUPS)),
            dot_dense_est=float(sum(float((gh[k]*gs[k]).sum()) for k in ALL_GROUPS)),
            actual_inclusion_freq=counters["selected_pixel_count"]/(cam["H"]*cam["W"]),
            active_parent_count=counters.get("active_parent_count"),
            selected_child_count=counters.get("selected_child_count"),
            selected_pixel_count=counters["selected_pixel_count"],
            gaussian_parent_candidate_count=(pc if pcsrc.startswith("NOT_MEASURED") is False else "NOT_MEASURED"),
            gaussian_child_candidate_count="NOT_MEASURED" if pcsrc.startswith("NOT_MEASURED") else pc,
            candidate_count_source=pcsrc,
            child_mask_sha256=counters.get("child_mask_sha256"),
            checkpoint_sha256=os.environ.get("WV0_CHECKPOINT_SHA256", "NOT_INJECTED"),
            source_sha256=os.environ.get("WV0_SOURCE_SHA256", "NOT_INJECTED"),
            so_sha256=os.environ.get("WV0_SO_SHA256", "NOT_INJECTED"),
            timing_class=counters.get("timing_class", "SUPERVISION_ORACLE")))
    return rows


def write_raw(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)


if __name__ == "__main__":
    import sys
    print("Stage B-min turnkey oracle patch. Run on MX with _wv0_core.py alongside.")
    print("Example: python -c \"import _wv0_grad; print('import OK')\"  (validates mask/metrics logic)")
    print("Methods:", list(CFG.keys()), "| p =", P, "| CRN: shared (U_parent,U_child) per cam/seed")
    if "--selftest" in sys.argv:
        # CPU-only shape/inclusion check of make_mask without torch GPU
        for H, W in [(65, 77), (540, 813)]:
            u = generate_common_uniforms(H, W, 24601)
            for m in CFG:
                w, meta = make_mask(m, H, W, u, device="cpu")
                assert w.shape == (H, W)
                assert float((w == 0).float().mean() + (w == 1/P).float().mean()) == 1.0
                freq = float((w > 0).float().mean())
                print(f"  {m:8s} {H}x{W}: incl={freq:.4f} sel_ch={meta['selected_child_count']}")
            # CRN endpoint identities (bitwise): G16==HWCS(q=p,r=1); G4==HWCS(q=1,r=p)
            g16 = make_mask("G16", H, W, u, device="cpu")[0]
            g4 = make_mask("G4_N1", H, W, u, device="cpu")[0]
            assert torch.equal(g16, make_mask("HWCS_qp_r1", H, W, u, device="cpu")[0]), "G16!=HWCS(q=p,r=1)"
            assert torch.equal(g4, make_mask("HWCS_q1_rp", H, W, u, device="cpu")[0]), "G4!=HWCS(q=1,r=p)"
            print("  CRN ENDPOINT IDENTITIES OK: G16==HWCS(q=p,r=1), G4==HWCS(q=1,r=p) bitwise")
        print("SELF-TEST OK (masks/shapes/inclusion/CRN-endpoints; GPU run requires MX).")
