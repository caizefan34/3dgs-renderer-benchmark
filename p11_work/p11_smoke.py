#!/usr/bin/env python3
"""P11 smoke test: exercise every API surface the gates use, cheaply."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))

    ident = C.bootstrap_gmain()
    print("bootstrap OK:", ident["so_sha256"][:16])
    ns = C.import_recipe()
    print("recipe imports OK; fused_ssim at", ns["fused_ssim"].__file__)

    ssim_fn = C.SepSSIM(device="cuda")
    print("SepSSIM OK")

    scene = "room"
    dataset = C.load_dataset(scene)
    model, extent, meta = C.load_model_for_dataset(
        scene, dataset,
        ckpt_path=os.path.join(C.FULL30K, "gmain03_s42_room", "ckpt_30000.pt"))
    print(f"model OK: N={meta['state_N']:,} iter={meta['state_iteration']} "
          f"sh={meta['active_sh_degree']} extent={extent:.3f}")

    cam, gt = dataset.get_item(0)
    print(f"camera OK: {cam.image_width}x{cam.image_height}")

    r0, image, radii_full, means2d = C.render_with_meta(
        model, cam, model.active_sh_degree)
    print(f"render OK: r0 {tuple(r0.shape)} image {tuple(image.shape)} "
          f"radii {tuple(radii_full.shape)} means2d {tuple(means2d.shape)}")
    assert hasattr(means2d, "absgrad"), "absgrad not attached"

    # loss paths (shared r0 graph -> retain_graph for the successive grads)
    loss_sep = C.sep_total_loss(image, gt, ssim_fn)
    g_sep = torch.autograd.grad(loss_sep, r0, retain_graph=True)[0]
    print(f"sep loss OK: {float(loss_sep):.6f} grad {tuple(g_sep.shape)} "
          f"finite={bool(torch.isfinite(g_sep).all())}")

    loss_f = C.fused_total_loss(image, gt, padding="same")
    g_f = torch.autograd.grad(loss_f, r0, retain_graph=True)[0]
    print(f"fused loss OK: {float(loss_f):.6f} finite={bool(torch.isfinite(g_f).all())}")
    d = (g_sep - g_f)
    print(f"grad diff: max_abs={float(d.abs().max()):.3e} "
          f"rel_l2={float(d.norm()/g_sep.norm()):.3e}")

    # renderer backward with external adjoint (means2d is a sibling output of
    # the op, not an ancestor of r0 -> excluded; inputs must be the EXACT
    # tensors handed to rasterization, so use return_inputs=True)
    v = g_sep.detach().contiguous()
    r0, image, radii_full, means2d, inputs = C.render_with_meta(
        model, cam, model.active_sh_degree, return_inputs=True)
    grads = torch.autograd.grad(r0, inputs, grad_outputs=v)
    print("renderer backward (ext adjoint) OK:",
          [tuple(g.shape) for g in grads])

    # contribution utilities
    from gsplat import rasterization
    with torch.no_grad():
        r, ra, m = rasterization(
            means=model.get_xyz, quats=model.get_rotation,
            scales=model.get_scaling, opacities=model.get_opacity,
            colors=model.get_features,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=model.active_sh_degree,
            radius_clip=0.0, eps2d=C.EPS2D, render_mode="RGB", absgrad=False)
        nc, alphas_util = ns["rasterize_num_contributing_gaussians"](
            m["means2d"], m["conics"], m["opacities"], m["isect_offsets"],
            m["flatten_ids"], cam.image_width, cam.image_height, 16)
        ad = (ra[0, :, :, 0] - alphas_util[0]).abs()
        print(f"num_contributing OK: total={int(nc.sum()):,} "
              f"alpha_max_diff={float(ad.max()):.3e} (FP-reassociation scale) "
              f"tile_hw={m['tile_height']}x{m['tile_width']}")
        print(f"isects={int(m['flatten_ids'].numel()):,} "
              f"offsets_last={int(m['isect_offsets'].flatten()[-1]):,}")

    # faithful full iteration at 30K state (post-15K: no densify chain)
    model.optimizer.zero_grad(set_to_none=True)
    r0, image, radii_full, means2d = C.render_with_meta(
        model, cam, model.active_sh_degree)
    L1 = F.l1_loss(image, gt)
    dssim = ssim_fn(image, gt)
    loss = 0.8 * L1 + 0.2 * dssim
    loss.backward()
    radii = radii_full[0]
    vis = (radii > 0).any(dim=-1)
    n_vis = int(vis.sum())
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"full iteration OK: loss={float(loss):.4f} n_visible={n_vis:,}")

    # quick timing sanity (20 iters)
    img = image.detach().requires_grad_(True)
    r = C.bench_events(lambda: torch.autograd.grad(
        C.sep_total_loss(img, gt, ssim_fn), img), warmup=5, iters=20, reps=2)
    print(f"sep combined fwd+bwd timing sanity: {r['median_of_medians_ms']:.2f} ms")
    img2 = image.detach().requires_grad_(True)
    r2 = C.bench_events(lambda: torch.autograd.grad(
        C.fused_total_loss(img2, gt), img2), warmup=5, iters=20, reps=2)
    print(f"fused combined fwd+bwd timing sanity: {r2['median_of_medians_ms']:.2f} ms")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
