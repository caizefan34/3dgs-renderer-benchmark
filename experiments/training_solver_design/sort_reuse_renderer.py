"""Diagnostic cached-list counterfactual; caches are valid only at fixed parameters."""

import math

import torch


def prepare(model, cam, eps2d=.3):
    from gsplat.cuda._wrapper import fully_fused_projection, spherical_harmonics
    means, quats, scales = model.get_xyz, model.get_rotation, model.get_scaling
    opacity, features = model.get_opacity, model.get_features
    viewmats = cam.viewmatrix[None]
    radii, xy, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmats, cam.K[None], cam.image_width, cam.image_height,
        eps2d=eps2d, packed=False, opacities=opacity)
    visible = (radii > 0).all(-1)
    colors = (spherical_harmonics(model.active_sh_degree, means, viewmats, features,
                                 masks=visible)+.5).relu()
    return {'radii': radii, 'xy': xy, 'depths': depths, 'conics': conics,
            'opacity': opacity[None].contiguous(), 'colors': colors,
            'coarse_xy': torch.where(visible[..., None], xy, 0.)/2,
            'coarse_conics': conics*4}


def intersections(data, width, height, tile=16, coarse=False):
    from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode
    tw, th = math.ceil(width/tile), math.ceil(height/tile)
    xy = data['coarse_xy'] if coarse else data['xy']
    conics = data['coarse_conics'] if coarse else data['conics']
    counts, ids, flat = isect_tiles(xy, data['radii'], data['depths'], tile, tw, th,
                                  packed=False, n_images=1, conics=conics, opacities=data['opacity'])
    offsets = isect_offset_encode(ids, 1, tw, th)
    return counts, ids, flat, offsets


def render_shared(model, cam, tile_mask=None, eps2d=.3, coarse_list=None):
    from gsplat.cuda._wrapper import rasterize_to_pixels
    data = prepare(model, cam, eps2d)
    fine_list = intersections(data, cam.image_width, cam.image_height)
    if coarse_list is None:
        coarse_list = intersections(data, cam.image_width//2, cam.image_height//2, coarse=True)

    def raster(xy, cc, width, height, mask, lists):
        image, _ = rasterize_to_pixels(xy, cc, data['colors'], data['opacity'], width, height,
                                      16, lists[3], lists[2], masks=mask, packed=False, absgrad=False)
        return image[0].clamp(0, 1)

    fine = raster(data['xy'], data['conics'], cam.image_width, cam.image_height, tile_mask, fine_list)
    coarse = raster(data['coarse_xy'], data['coarse_conics'], cam.image_width//2, cam.image_height//2,
                    None, coarse_list)
    return fine, coarse
