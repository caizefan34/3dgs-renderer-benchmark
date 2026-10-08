"""Two resolutions sharing projection and SH, isolated from frozen baselines.

The proxy uses only fine-visible Gaussians. It is deliberately a new proxy,
not a claim of equivalence to an independently projected coarse camera.
"""

import math

import torch


def rescale_conics(conics, visible, factor=2, eps2d=.3):
    # Inactive projection outputs need not be initialized: sanitize before math.
    safe = conics.new_tensor([1., 0., 1.])
    a, b, c = torch.where(visible[..., None], conics, safe).unbind(-1)
    k = eps2d*(factor*factor-1)
    det = (a*c-b*b).clamp_min(0)
    denom = 1+k*(a+c)+k*k*det
    return factor*factor*torch.stack((a+k*det, b, c+k*det), -1)/denom[..., None]


def render_shared(model, cam, tile_mask=None, factor=2, eps2d=.3):
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode,
        spherical_harmonics, rasterize_to_pixels,
    )
    means, quats, scales = model.get_xyz, model.get_rotation, model.get_scaling
    opacities, features = model.get_opacity, model.get_features
    viewmats = cam.viewmatrix[None]
    radii, projected, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmats, cam.K[None], cam.image_width, cam.image_height,
        eps2d=eps2d, packed=False, opacities=opacities)
    opacity = opacities[None].contiguous()
    visible = (radii > 0).all(-1)
    colors = (spherical_harmonics(model.active_sh_degree, means, viewmats, features,
                                 masks=visible)+.5).relu()

    def raster(xy, cc, width, height, mask):
        tw, th = math.ceil(width/16), math.ceil(height/16)
        # In this pinned AccuTile path radii provide only a visibility gate;
        # its actual ellipse bounds are recomputed from conics and opacity.
        _, ids, flat = isect_tiles(xy, radii, depths, 16, tw, th,
                                  packed=False, n_images=1, conics=cc, opacities=opacity)
        offsets = isect_offset_encode(ids, 1, tw, th)
        image, _ = rasterize_to_pixels(xy, cc, colors, opacity, width, height,
                                      16, offsets, flat, masks=mask, packed=False, absgrad=False)
        return image[0].clamp(0, 1)

    fine = raster(projected, conics, cam.image_width, cam.image_height, tile_mask)
    coarse_conics = rescale_conics(conics, visible, factor, eps2d)
    coarse_xy = torch.where(visible[..., None], projected, 0.)/factor
    coarse = raster(coarse_xy, coarse_conics, cam.image_width//factor, cam.image_height//factor, None)
    return fine, coarse
