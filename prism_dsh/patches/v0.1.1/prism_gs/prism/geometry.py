"""Differentiable pinhole geometry and a chunkable per-Gaussian consensus solver.

Coordinates: world-to-camera matrices, wxyz quaternions, pixel centers.
Screen features are the renderer's native conic form [u,v,a,b,c]; the FOV clamp
follows gsplat's pinhole projection; CUDA parity is a GPU gate.
No claim of closed-form nonlinear optimization or ADMM convergence is made.

v0.1.1 P1: the screen transport carries the renderer's native conic instead of a
log-Cholesky re-parameterization, so reading features back and re-rendering is
bitwise (M0 zero-displacement gate) and no inverse roundtrip is involved.
v0.1.1 P3: covariance() delegates to the pinned renderer's own conversion on
CUDA so a covars-route render matches the quats route bitwise; the pure-torch
expression is a CPU fallback and is not renderer-parity.
"""
import torch
import torch.nn.functional as F


def quat_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def covariance(q, log_scales):
    """Quaternion -> world covariance in the renderer's own value convention.

    On CUDA this delegates to the pinned renderer's device function on the same
    normalized quaternions the renderer receives, so 'covars=' routes reproduce
    the quats route exactly. The pure-torch path is the CPU fallback.
    """
    if q.is_cuda:
        from gsplat.cuda._wrapper import quat_scale_to_covar_preci
        six, _ = quat_scale_to_covar_preci(
            F.normalize(q, dim=-1), log_scales.exp(), True, False, True)
        # Renderer triu layout [00,01,02,11,12,22] mirrored to a full matrix.
        return torch.stack((six[..., 0], six[..., 1], six[..., 2],
                            six[..., 1], six[..., 3], six[..., 4],
                            six[..., 2], six[..., 4], six[..., 5]),
                           -1).reshape(six.shape[:-1]+(3, 3))
    r = quat_matrix(q)
    return (r * log_scales.exp().square()[..., None, :]) @ r.transpose(-1, -2)


def pack_geometry(means, cov):
    l = torch.linalg.cholesky(cov)
    return torch.cat((means, torch.stack((l[...,0,0].log(), l[...,1,0],
        l[...,1,1].log(), l[...,2,0], l[...,2,1], l[...,2,2].log()), -1)), -1)


def unpack_geometry(x):
    a,b,c,d,e,f = x[...,3:].unbind(-1)
    o = torch.zeros_like(a)
    l = torch.stack((a.exp(),o,o,b,c.exp(),o,d,e,f.exp()), -1).reshape(x.shape[:-1]+(3,3))
    return x[...,:3], l @ l.transpose(-1,-2)


def screen_pack(uv, conic):
    """Native conic transport (P1): concatenation only, no hidden roundtrip."""
    return torch.cat((uv, conic), -1)


def screen_unpack(z):
    return z[..., :2], z[..., 2:]


def cov_to_conic(c):
    """2x2 inverse with the fused kernel's elementwise arithmetic (glm::inverse)."""
    det = c[..., 0, 0]*c[..., 1, 1]-c[..., 0, 1]*c[..., 1, 0]
    return torch.stack((c[..., 1, 1]/det, -c[..., 0, 1]/det, c[..., 0, 0]/det), -1)


def conic_to_cov(c):
    a, b, d = c.unbind(-1)
    det = a*d-b*b
    return torch.stack((d/det, -b/det, -b/det, a/det), -1).reshape(c.shape[:-1]+(2, 2))


def project(x, views, ks, width, height, eps2d=.3):
    """x [N,9] -> features [V,N,5] in the renderer's native conic form, depth [V,N].
    Does NOT decide visibility.

    Caller must supply a valid camera/visibility mask; near-plane-invalid rows
    are excluded before fitting. Covariance uses gsplat's asymmetric FOV clamp;
    the conic is the same 2x2 inverse the fused kernel stores (P1).
    """
    means, cov = unpack_geometry(x)
    r, t = views[:,:3,:3], views[:,:3,3]
    mc = torch.einsum('vij,nj->vni', r, means) + t[:,None]
    cv = r[:,None] @ cov[None] @ r[:,None].transpose(-1,-2)
    tx,ty,tz = mc.unbind(-1)
    safe_z = torch.where(tz.abs() < 1e-8, torch.full_like(tz,1e-8), tz)
    fx,fy,cx,cy = [ks[:,i,j,None] for i,j in [(0,0),(1,1),(0,2),(1,2)]]
    uv = torch.stack((fx*tx/safe_z+cx, fy*ty/safe_z+cy), -1)
    nx = torch.clamp(tx/safe_z, min=-(cx+.15*width)/fx, max=(width-cx+.15*width)/fx)
    ny = torch.clamp(ty/safe_z, min=-(cy+.15*height)/fy, max=(height-cy+.15*height)/fy)
    o = torch.zeros_like(tz)
    j = torch.stack((fx/safe_z,o,-fx*nx/safe_z,o,fy/safe_z,-fy*ny/safe_z),-1).reshape(*tz.shape,2,3)
    c2 = j @ cv @ j.transpose(-1,-2) + eps2d*torch.eye(2,device=x.device,dtype=x.dtype)
    return screen_pack(uv,cov_to_conic(c2)), tz


def jacobian(x, views, ks, width, height, eps2d=.3):
    """[V,N,5,9], exploiting independence across Gaussian rows; no N x N graph."""
    x = x.detach().requires_grad_(True)
    z,_ = project(x,views,ks,width,height,eps2d)
    rows=[]
    for v in range(len(views)):
        rows.append(torch.stack([torch.autograd.grad(z[v,:,k].sum(),x,
            retain_graph=True)[0] for k in range(5)],1))
    return z.detach(), torch.stack(rows).detach()


def decompose(j, d, weights, damping=1e-4, parameter_scale=None):
    """Weighted demand decomposition. Returns delta, residual, singular values.

    The damped residual includes regularization/conditioning effects; it is NOT
    evidence of representation failure. Singular values must accompany it.
    """
    v,n,k,p = j.shape
    a = (j*weights[...,None].sqrt()).permute(1,0,2,3).reshape(n,v*k,p)
    b = (d*weights.sqrt()).permute(1,0,2).reshape(n,v*k)
    if parameter_scale is None:
        parameter_scale = torch.ones(n,p,device=j.device,dtype=j.dtype)
    a_scaled = a*parameter_scale[:,None]
    eye = torch.eye(p,device=j.device,dtype=j.dtype)
    h = a_scaled.transpose(-1,-2)@a_scaled
    rhs = (a_scaled.transpose(-1,-2)@b[...,None]).squeeze(-1)
    step = torch.linalg.solve(h+damping*eye,rhs[...,None]).squeeze(-1)
    delta = step*parameter_scale
    residual = (b-(a@delta[...,None]).squeeze(-1)).reshape(n,v,k).permute(1,0,2)
    return delta, residual, torch.linalg.svdvals(a_scaled)


def fit_consensus(x0, target, views, ks, width, height, weights, steps=3,
                  damping=1e-3, eps2d=.3, max_step=.25):
    """Damped GN + per-row backtracking on the actual projection objective.

    This is a diagnostic solver, not an optimized CUDA implementation. It
    records rejected proposals. An outer real-image acceptance test is required.
    """
    x=x0.detach().clone()
    _,cov=unpack_geometry(x0)
    size=cov.diagonal(dim1=-2,dim2=-1).mean(-1).sqrt().clamp_min(1e-6)
    scale=torch.stack((size,size,size,torch.ones_like(size),size,
        torch.ones_like(size),size,size,torch.ones_like(size)),-1)
    history=[]
    def cost(z): return ((z-target).square()*weights).sum((0,2))
    for _ in range(steps):
        z,j=jacobian(x,views,ks,width,height,eps2d)
        delta,residual,singular=decompose(j,target-z,weights,damping,scale)
        relative=(delta/scale).norm(dim=-1).clamp_min(1e-12)
        delta=delta*(max_step/relative).clamp_max(1)[:,None]
        old=cost(z); accepted=torch.zeros_like(old,dtype=torch.bool)
        before=x.clone(); trials=0
        for alpha in (1.,.5,.25,.125):
            proposal=before+alpha*delta
            with torch.no_grad():
                zp,depth=project(proposal,views,ks,width,height,eps2d)
                cp=cost(zp)
                required=(weights.sum(-1)>0)
                valid=((depth>.01)|~required).all(0)
                ok=(~accepted)&valid&torch.isfinite(cp)&(cp<old)
                x=torch.where(ok[:,None],proposal,x)
                accepted|=ok
            trials+=1
            if bool(accepted.all()): break
        history.append({'objective_before':float(old.sum()),
            'objective_after':float(cost(project(x,views,ks,width,height,eps2d)[0]).sum()),
            'accepted_rows':int(accepted.sum()),'line_search_trials':trials,
            'weighted_residual_energy':float(residual.square().sum()),
            'min_singular_value':float(singular[:,-1].min())})
    return x.detach(),history
