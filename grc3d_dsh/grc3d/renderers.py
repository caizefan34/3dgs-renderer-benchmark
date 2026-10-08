"""Small differentiable 3D oracle and a public gsplat diagnostic adapter."""
import math
import torch
import torch.nn.functional as F


def rotation(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def sh_colors(sh, directions, degree):
    """Real SH through degree 3, Graphdeco/gsplat coefficient ordering."""
    x, y, z = F.normalize(directions, dim=-1).unbind(-1)
    basis = [torch.full_like(x, .28209479177387814)]
    if degree >= 1:
        c = .4886025119029199
        basis += [-c*y, c*z, -c*x]
    if degree >= 2:
        basis += [1.0925484305920792*x*y, -1.0925484305920792*y*z,
                  .31539156525252005*(2*z*z-x*x-y*y),
                  -1.0925484305920792*x*z, .5462742152960396*(x*x-y*y)]
    if degree >= 3:
        basis += [-.5900435899266435*y*(3*x*x-y*y), 2.890611442640554*x*y*z,
                  -.4570457994644658*y*(4*z*z-x*x-y*y),
                  .3731763325901154*z*(2*z*z-3*x*x-3*y*y),
                  -.4570457994644658*x*(4*z*z-x*x-y*y),
                  1.445305721320277*z*(x*x-y*y), -.5900435899266435*x*(x*x-3*y*y)]
    return (torch.stack(basis, -1)[..., None]*sh[:, :(degree+1)**2]).sum(1).add(.5).clamp_min(0)


def resized_camera(camera, height, width):
    k = camera["K"].clone()
    k[0] *= width/camera["width"]
    k[1] *= height/camera["height"]
    return k


def render_reference(params, camera, height, width, degree=3, eps2d=.3):
    """O(NHW), pinhole Jacobian projection, full support, depth alpha composition.

    No gsplat tile culling/early exit/alpha threshold, covariance FOV clamp,
    antialiasing or absgrad. This is a math oracle, never a gsplat-equivalent benchmark.
    Intended for tiny CPU scenes, black background only.
    """
    means = params["means"]
    view, k = camera["viewmat"], resized_camera(camera, height, width)
    xyz = means@view[:3, :3].T+view[:3, 3]
    z = xyz[:, 2]
    if (z <= .01).any():
        raise ValueError("Oracle requires all Gaussians in front of camera")
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    uv = torch.stack((fx*xyz[:, 0]/z+cx, fy*xyz[:, 1]/z+cy), -1)
    zeros = torch.zeros_like(z)
    jac = torch.stack((fx/z, zeros, -fx*xyz[:, 0]/z.square(),
                       zeros, fy/z, -fy*xyz[:, 1]/z.square()), -1).reshape(-1, 2, 3)
    r = rotation(params["quats"])
    cov = (r*params["log_scales"].exp().square()[:, None, :])@r.transpose(-1, -2)
    cv = view[:3, :3]@cov@view[:3, :3].T
    cov2 = jac@cv@jac.transpose(-1, -2)+eps2d*torch.eye(2, device=z.device, dtype=z.dtype)
    inv = torch.linalg.inv(cov2)
    yy, xx = torch.meshgrid(torch.arange(height, dtype=z.dtype, device=z.device)+.5,
                            torch.arange(width, dtype=z.dtype, device=z.device)+.5, indexing="ij")
    delta = torch.stack((xx, yy), -1)[None]-uv[:, None, None, :]
    power = torch.einsum("nhwi,nij,nhwj->nhw", delta, inv, delta)
    alpha = (params["opacity_logits"].sigmoid()[:, None, None]*torch.exp(-.5*power)).clamp_max(.99)
    center = -(view[:3, :3].T@view[:3, 3])
    colors = sh_colors(torch.cat((params["sh0"], params["shN"]), 1), means-center, degree)
    order = torch.argsort(z.detach())
    trans = torch.ones_like(xx); result = torch.zeros(height, width, 3, dtype=z.dtype, device=z.device)
    for i in order.tolist():
        result = result + (trans*alpha[i])[..., None]*colors[i]
        trans = trans*(1-alpha[i])
    return result.clamp(0, 1)


def render_gsplat(params, camera, height, width, degree=3, eps2d=.3):
    from gsplat import rasterization
    out, _, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=params["log_scales"].exp(),
        opacities=params["opacity_logits"].sigmoid(), colors=torch.cat((params["sh0"], params["shN"]), 1),
        viewmats=camera["viewmat"][None], Ks=resized_camera(camera, height, width)[None],
        width=width, height=height, sh_degree=degree, tile_size=16, packed=False,
        eps2d=eps2d, radius_clip=0., render_mode="RGB", rasterize_mode="classic", absgrad=False)
    return out[0].clamp(0, 1)


def synthetic(scale=.18, drift=.003, seed=7, height=24, width=32, degree=3):
    gen = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.rand(s, generator=gen, dtype=torch.float64)
    normal = lambda *s: torch.randn(s, generator=gen, dtype=torch.float64)
    n = 6
    means = torch.cat(((rand(n, 2)-.5)*1.6, 2+rand(n, 1)), 1)
    p = {"means": means, "log_scales": math.log(scale)+normal(n, 3)*.2,
         "quats": torch.cat((torch.ones(n, 1, dtype=torch.float64), normal(n, 3)*.15), 1),
         "opacity_logits": normal(n)*.3+.2,
         "sh0": normal(n, 1, 3)*.5, "shN": normal(n, (degree+1)**2-1, 3)*.08}
    anchor = {k: v+normal(*v.shape)*drift for k, v in p.items()}
    truth = {k: v+normal(*v.shape)*.015 for k, v in p.items()}
    camera = {"width": width, "height": height, "viewmat": torch.eye(4, dtype=torch.float64),
              "K": torch.tensor([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], dtype=torch.float64)}
    target = render_reference(truth, camera, height, width, degree).detach()
    return {k: v.requires_grad_() for k, v in p.items()}, anchor, camera, target
