import sys, os, json, time, statistics, random
P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
FUSED = P + "/FUSED_HOTSPOT0"
for p in [P+"/pylibs", P+"/methods_overlay", P+"/ATOM0",
          "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/src",
          "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/scripts",
          FUSED+"/scripts"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import FasterGSFusedCudaBackend._C_atom0 as CATOM
import hotspot  # FUSED-HOTSPOT0 harness (build, make_iter)

CHUNK_N_BUCKETS = 5_000_000
WARMUP = 30
SAMPLES = 20

def build_model(config, ckpt, preload=0):
    return hotspot.build(config, ckpt, preload)

def view_settings(trainer, view, g, adam_step_count=300001):
    from Methods.FasterGSFused.Renderer import extract_settings
    bg = view.camera.background_color
    s = extract_settings(view, g.active_sh_bases, bg, g.lr_means, adam_step_count)
    return s, bg

def call_forward(g, settings, bg):
    out = CATOM.forward(
        g.means.contiguous(), g.raw_scales.contiguous(), g.raw_rotations.contiguous(),
        g.raw_opacities.contiguous(), g.sh_coefficients_0.contiguous(), g.sh_coefficients_rest.contiguous(),
        *settings.as_tuple())
    image, pb, tb, ib, bb, n_instances, n_buckets, sel = out
    return dict(image=image, pb=pb, tb=tb, ib=ib, bb=bb,
                n_instances=int(n_instances), n_buckets=int(n_buckets), sel=int(sel),
                width=settings.width, height=settings.height)

def get_instance_prim(fd):
    return CATOM.get_instance_primitive_indices(fd['ib'], fd['n_instances'],
                                                 fd['width'], fd['height'], fd['sel'])

def time_variant(fd, settings, bg, n_prim, variant, cf, cn, grad_image,
                 grad_mean2d, grad_conic, grad_opacity, grad_colors,
                 private_grad, partial_out, capture):
    EMPTY = torch.empty(0, device='cuda')
    gm = grad_mean2d if variant == 0 else EMPTY
    gc = grad_conic if variant == 0 else EMPTY
    go = grad_opacity if variant == 0 else EMPTY
    gcol = grad_colors if variant == 0 else EMPTY
    priv = private_grad if variant > 0 else EMPTY
    po = partial_out if capture else EMPTY
    ms, _ = CATOM.backward_atom0(
        grad_image, fd['image'], bg, fd['pb'], fd['tb'], fd['ib'], fd['bb'],
        n_prim, settings.width, settings.height, fd['n_instances'], fd['n_buckets'], fd['sel'],
        variant, cf, cn, gm, gc, go, gcol, priv, po, 1 if capture else 0)
    return ms

def measure_abc(fd, settings, bg, n_prim, seed=1234, chunk_n_buckets=CHUNK_N_BUCKETS,
                warmup=WARMUP, samples=SAMPLES, capture_equiv=False, tag=""):
    rng = random.Random(seed)
    width, height = settings.width, settings.height
    gen = torch.Generator(device='cuda').manual_seed(seed)
    grad_image = torch.randn(fd['image'].shape, generator=gen, device='cuda')
    n_buckets = fd['n_buckets']

    grad_mean2d = torch.zeros(n_prim, 2, device='cuda')
    grad_conic = torch.zeros(3 * n_prim, device='cuda')
    grad_opacity = torch.zeros(n_prim, 1, device='cuda')
    grad_colors = torch.zeros(n_prim, 3, device='cuda')
    chunk_cap = min(chunk_n_buckets, n_buckets) * 32
    private_grad = torch.zeros(chunk_cap, 9, device='cuda')
    partial_out = torch.zeros(chunk_cap, 9, device='cuda')

    n_buckets = fd['n_buckets']
    # T_A_full: production-equivalent variant 0 over ALL buckets (authoritative ref)
    ta_full = time_variant(fd, settings, bg, n_prim, 0, 0, n_buckets, grad_image,
                           grad_mean2d, grad_conic, grad_opacity, grad_colors,
                           None, None, False)

    chunks = [(s, min(chunk_n_buckets, n_buckets - s)) for s in range(0, n_buckets, chunk_n_buckets)]
    T = {0: 0.0, 1: 0.0, 2: 0.0}
    raw = []
    for ci, (cf, cn) in enumerate(chunks):
        for v in (0, 1, 2):
            for _ in range(warmup):
                time_variant(fd, settings, bg, n_prim, v, cf, cn, grad_image,
                             grad_mean2d, grad_conic, grad_opacity, grad_colors,
                             private_grad, partial_out, False)
        torch.cuda.synchronize()
        order = [0, 1, 2]; rng.shuffle(order)
        samp = {0: [], 1: [], 2: []}
        for v in order:
            for _ in range(samples):
                samp[v].append(time_variant(fd, settings, bg, n_prim, v, cf, cn, grad_image,
                                            grad_mean2d, grad_conic, grad_opacity, grad_colors,
                                            private_grad, partial_out, False))
        torch.cuda.synchronize()
        for v in (0, 1, 2):
            T[v] += statistics.median(samp[v])
        raw.append({"chunk_index": ci, "chunk_first_bucket": cf, "chunk_n_buckets": cn,
                    "A": samp[0], "B": samp[1], "C": samp[2]})
        print(f"  [{tag}] chunk {ci}/{len(chunks)} fb={cf} nb={cn} "
              f"A={statistics.median(samp[0]):.4f} B={statistics.median(samp[1]):.4f} "
              f"C={statistics.median(samp[2]):.4f} ms", flush=True)

    equiv = None
    if capture_equiv and len(chunks) > 0:
        cf, cn = chunks[0]
        cap = {0: None, 1: None, 2: None}
        for v in (0, 1, 2):
            po = torch.zeros(chunk_cap, 9, device='cuda')
            time_variant(fd, settings, bg, n_prim, v, cf, cn, grad_image,
                         grad_mean2d, grad_conic, grad_opacity, grad_colors,
                         private_grad, po, True)
            torch.cuda.synchronize()
            cap[v] = po
        eq_ab = torch.equal(cap[0], cap[1])
        eq_ac = torch.equal(cap[0], cap[2])
        eq_bc = torch.equal(cap[1], cap[2])
        maxdiff = (cap[0] - cap[1]).abs().max().item()
        equiv = {"partial_A_eq_B": bool(eq_ab), "partial_A_eq_C": bool(eq_ac),
                 "partial_B_eq_C": bool(eq_bc), "max_abs_diff_A_B": float(maxdiff),
                 "chunk_first_bucket": cf, "chunk_n_buckets": cn}
        print(f"  EQUIV chunk0: A==B {eq_ab} A==C {eq_ac} B==C {eq_bc} maxdiff {maxdiff}", flush=True)

    return {"T_A_full_ms": ta_full, "T_A_ms": T[0], "T_B_ms": T[1], "T_C_ms": T[2],
            "n_chunks": len(chunks), "chunk_n_buckets": chunk_n_buckets,
            "raw_chunks": raw, "equiv": equiv}
