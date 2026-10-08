"""Isolated composition of existing gsplat kernels with a tile mask.

Keeps full-camera projection and intersections, avoiding crop-camera changes.
No new CUDA, approximate Gaussian culling, or modifications to frozen sources.
"""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def core_tile_mask(labels, selected, height, width, halo=5, tile_size=16):
    rows, columns = labels.shape
    assert height % rows == 0 and width % columns == 0
    core = (labels == selected).repeat_interleave(height//rows, 0).repeat_interleave(width//columns, 1)
    covered = F.max_pool2d(core.float()[None, None], 2*halo+1, stride=1, padding=halo)[0, 0] > 0
    tile_h, tile_w = math.ceil(height/tile_size), math.ceil(width/tile_size)
    padded = F.pad(covered, (0, tile_w*tile_size-width, 0, tile_h*tile_size-height))
    tiles = padded.reshape(tile_h, tile_size, tile_w, tile_size).any(dim=3).any(dim=1)
    return core, covered, tiles[None].contiguous()


def render_masked(model, cam, tile_mask=None, eps2d=.3):
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
    tile_w, tile_h = math.ceil(cam.image_width/16), math.ceil(cam.image_height/16)
    _, ids, flat = isect_tiles(projected, radii, depths, 16, tile_w, tile_h,
                              packed=False, n_images=1, conics=conics, opacities=opacity)
    offsets = isect_offset_encode(ids, 1, tile_w, tile_h)
    colors = (spherical_harmonics(model.active_sh_degree, means, viewmats, features,
                                  masks=(radii > 0).all(-1))+.5).relu()
    image, _ = rasterize_to_pixels(projected, conics, colors, opacity,
                                   cam.image_width, cam.image_height, 16, offsets, flat,
                                   masks=tile_mask, packed=False, absgrad=False)
    return image[0].clamp(0, 1)
