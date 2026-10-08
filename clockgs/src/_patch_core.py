import ast
p = 'clockgs/src/core.py'
s = open(p, encoding='utf-8').read()

s = s.replace('from gaussian_renderer import render\n', '')

anchor = 'from utils.image_utils import psnr\n'
local_render = '''from utils.image_utils import psnr

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0,
           separate_sh=False, override_color=None, use_trained_exp=False):
    # Local renderer matching the installed (older) Inria rasterizer:
    # no antialiasing field; forward returns (rendered_image, radii).
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=pipe.debug)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = rotations = cov3D_precomp = None
    shs = colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            from utils.sh_utils import eval_sh
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    rendered_image, radii = rasterizer(
        means3D=means3D, means2D=means2D, shs=shs, colors_precomp=colors_precomp,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)
    rendered_image = rendered_image.clamp(0, 1)
    return {"render": rendered_image, "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero(), "radii": radii}
'''
assert anchor in s, 'anchor not found'
s = s.replace(anchor, local_render, 1)
open(p, 'w', encoding='utf-8').write(s)
ast.parse(s)
print('patched + syntax OK; local render present:', 'def render(viewpoint_camera' in s)
