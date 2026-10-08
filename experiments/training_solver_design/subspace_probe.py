"""Synthetic 3DGS experiment for full-loss gradients + sampled subspace curvature.

The CPU autograd Jacobian is an oracle, not the planned CUDA implementation.
No runtime comparison here represents end-to-end training speed.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import lsq_linear

from appearance_oracle import fixture, render


SIZE = 18
DTYPE = torch.float64


def photometric(pred, target):
    x = pred.reshape(-1, SIZE, SIZE, 3).permute(0, 3, 1, 2)
    y = target.reshape_as(pred).reshape(-1, SIZE, SIZE, 3).permute(0, 3, 1, 2)
    k = torch.exp(-0.5*(torch.arange(-5, 6, dtype=pred.dtype)/1.5)**2)
    k = k/k.sum()
    kernel = (k[:, None]*k[None, :]).expand(3, 1, 11, 11).contiguous()
    mx, my = F.conv2d(x, kernel, groups=3), F.conv2d(y, kernel, groups=3)
    vx = F.conv2d(x*x, kernel, groups=3)-mx*mx
    vy = F.conv2d(y*y, kernel, groups=3)-my*my
    cov = F.conv2d(x*y, kernel, groups=3)-mx*my
    ssim = ((2*mx*my+0.01**2)*(2*cov+0.03**2))/((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))
    return 0.8*(pred-target).abs().mean()+0.2*(1-ssim.mean()), ssim.mean()


def score(pred, target):
    loss, ssim = photometric(pred, target)
    return {'loss': loss.item(), 'ssim': ssim.item(), 'psnr': (-10*torch.log10((pred-target).square().mean())).item()}


def schur(h, g):
    geometry = torch.arange(0, h.shape[0], 3)
    appearance = torch.tensor([i for i in range(h.shape[0]) if i % 3], dtype=torch.long)
    aa = h[appearance[:, None], appearance]
    gg = h[geometry[:, None], geometry]
    ga = h[geometry[:, None], appearance]
    dg = torch.linalg.solve(gg-ga @ torch.linalg.solve(aa, ga.T), -g[geometry]+ga @ torch.linalg.solve(aa, g[appearance]))
    da = torch.linalg.solve(aa, -g[appearance]-ga.T @ dg)
    result = torch.empty_like(g)
    result[geometry], result[appearance] = dg, da
    return result


def run(seed, mode, constraint):
    means, scales, opacity, colors, cameras = fixture(seed)
    rng = torch.Generator().manual_seed(seed+1000)
    target = render(means, scales, opacity, colors, cameras, size=SIZE).detach()
    validation_camera = torch.tensor([[0.55, -0.05, 0.0]], dtype=DTYPE)
    validation_target = render(means, scales, opacity, colors, validation_camera, size=SIZE).detach()
    nmeans = means.numel()
    start_means = means+0.055*torch.randn(means.shape, generator=rng, dtype=DTYPE)
    start_colors = colors+0.06*torch.randn(colors.shape, generator=rng, dtype=DTYPE)
    theta = torch.cat((start_means.flatten(), start_colors.flatten())).requires_grad_()

    def model(t, cams=cameras):
        return render(t[:nmeans].reshape_as(means), scales, opacity, t[nmeans:].reshape_as(colors), cams, size=SIZE)

    loss, _ = photometric(model(theta), target)
    gradient = torch.autograd.grad(loss, theta)[0]
    # A precisely defined first Adam step, no momentum state from earlier steps.
    adam = -0.005*gradient/(gradient.abs()+1e-8)
    selected = 8
    basis = torch.zeros(theta.numel(), selected*3, dtype=DTYPE)
    for gaussian in range(selected):
        basis[3*gaussian:3*gaussian+3, 3*gaussian] = adam[3*gaussian:3*gaussian+3]
        color_start = nmeans+12*gaussian
        basis[color_start:color_start+3, 3*gaussian+1] = adam[color_start:color_start+3]
        basis[color_start+3:color_start+12, 3*gaussian+2] = adam[color_start+3:color_start+12]
    selected_adam = basis @ torch.ones(selected*3, dtype=DTYPE)
    unselected_adam = adam-selected_adam
    started = time.perf_counter()
    j = torch.autograd.functional.jacobian(
        lambda s: model(theta.detach()+basis @ s).flatten(), torch.zeros(selected*3, dtype=DTYPE), vectorize=True, strategy='forward-mode',
    )
    jacobian_seconds = time.perf_counter()-started
    sampled = torch.randperm(j.shape[0], generator=rng)[:j.shape[0]//4]
    if mode == 'absolute_mse':
        h_data = j[sampled].T @ j[sampled]/len(sampled)
        h = h_data+0.003*torch.eye(h_data.shape[0], dtype=DTYPE)
    else:
        residual = (model(theta.detach())-target).detach().flatten()
        weights = 0.8/residual[sampled].abs().clamp_min(0.01)
        h_data = j[sampled].T @ (weights[:, None]*j[sampled])/len(sampled)
        scale = h_data.diag().clamp_min(h_data.diag().mean()*0.01+1e-15)
        h = h_data+0.3*torch.diag(scale)
    g = basis.T @ gradient  # full L1+SSIM gradient, NOT sampled MSE gradient
    image_leaf = model(theta.detach()).detach().requires_grad_()
    image_adjoint = torch.autograd.grad(photometric(image_leaf, target)[0], image_leaf)[0]
    projected = j.T @ image_adjoint.flatten()
    gradient_rel_error = ((g-projected).norm()/g.norm().clamp_min(1e-15)).item()
    if gradient_rel_error > 1e-7:
        raise RuntimeError(f'Directional Jacobian disagrees with full loss gradient: {gradient_rel_error}')
    coupled = schur(h, g)
    dense = torch.linalg.solve(h, -g)
    difference = (coupled-dense).abs().max().item()
    independent = torch.cat([torch.linalg.solve(h[i:i+3, i:i+3], -g[i:i+3]) for i in range(0, len(g), 3)])
    diagonal = -g/h.diag()
    kkt_checks = []
    if constraint == 'box':
        def bounded(matrix):
            chol = torch.linalg.cholesky(matrix)
            result = lsq_linear(chol.T.numpy(), torch.linalg.solve_triangular(chol, -g[:, None], upper=False).flatten().numpy(),
                                bounds=(-4.0, 4.0), method='bvls', tol=1e-12, max_iter=200)
            if not result.success:
                raise RuntimeError(result.message)
            solution = torch.from_numpy(result.x)
            derivative = matrix @ solution+g
            violation = derivative.abs()
            lower, upper = solution <= -4+1e-8, solution >= 4-1e-8
            violation[lower] = (-derivative[lower]).clamp_min(0)
            violation[upper] = derivative[upper].clamp_min(0)
            relative = (violation.max()/g.abs().max().clamp_min(1e-15)).item()
            if relative > 1e-6:
                raise RuntimeError(f'Bounded QP failed KKT check: {relative}')
            kkt_checks.append(relative)
            return solution
        coupled = bounded(h)
        independent_h = torch.block_diag(*[h[i:i+3, i:i+3] for i in range(0, len(g), 3)])
        independent = bounded(independent_h)
        diagonal = (-g/h.diag()).clamp(-4, 4)
    with torch.no_grad():
        baseline_theta = theta+adam
        baseline_loss = photometric(model(baseline_theta), target)[0].item()
        methods = {'initial': {'fit': score(model(theta), target), 'validation': score(model(theta, validation_camera), validation_target)},
                   'adam': {'fit': score(model(baseline_theta), target), 'validation': score(model(baseline_theta, validation_camera), validation_target)}}
        # Same simple three-trial allowance: distinguish coupling from larger LR.
        scaled_adam = [(photometric(model(theta+s*adam), target)[0].item(), s) for s in (1.0, 2.0, 4.0)]
        _, chosen_scale = min(scaled_adam)
        scaled_theta = theta+chosen_scale*adam
        methods['scaled_adam'] = {'fit': score(model(scaled_theta), target),
                                  'validation': score(model(scaled_theta, validation_camera), validation_target),
                                  'chosen_scale': chosen_scale}
        selected_trials = [(photometric(model(theta+unselected_adam+s*selected_adam), target)[0].item(), s) for s in (1.0, 2.0, 4.0)]
        _, selected_scale = min(selected_trials)
        selected_theta = theta+unselected_adam+selected_scale*selected_adam
        methods['scaled_selected_adam'] = {'fit': score(model(selected_theta), target),
                                           'validation': score(model(selected_theta, validation_camera), validation_target),
                                           'chosen_scale': selected_scale}
        coupled_name = 'bounded_coupled' if constraint == 'box' else 'coupled_schur'
        for name, step in [('diagonal', diagonal), ('per_gaussian', independent), (coupled_name, coupled)]:
            step = step*min(1.0, 4.0/max(step.abs().max().item(), 1e-12))
            accepted_theta, accepted_scale = baseline_theta, 0.0
            attempted = []
            for scale in (1.0, 0.5, 0.25):
                trial = theta+unselected_adam+basis @ (scale*step)
                trial_loss = photometric(model(trial), target)[0].item()
                attempted.append({'scale': scale, 'loss': trial_loss})
                if trial_loss < baseline_loss:
                    accepted_theta, accepted_scale = trial, scale
                    break
            methods[name] = {'fit': score(model(accepted_theta), target),
                             'validation': score(model(accepted_theta, validation_camera), validation_target),
                             'accepted_scale': accepted_scale, 'trial_losses': attempted,
                             'coefficient_max_abs': step.abs().max().item()}
    return {'seed': seed, 'methods': methods, 'schur_vs_dense_max_error': difference,
            'full_scalar_residuals': j.shape[0], 'curvature_scalar_samples': len(sampled),
            'subspace_dimensions': h.shape[0], 'oracle_jacobian_seconds_cpu': jacobian_seconds,
            'gradient_source': 'full original 0.8 L1 + 0.2 valid-window D-SSIM',
            'curvature_source': f'{mode}; uniform 25 percent RGB-response rows; not exact SSIM Hessian',
            'mean_response_curvature_diagonal': h_data.diag().mean().item(),
            'projected_full_gradient_rel_error': gradient_rel_error,
            'constraint': constraint, 'bounded_kkt_relative_violations': kkt_checks,
            'bounded_solver': 'SciPy BVLS CPU oracle; Schur comparison covers the unconstrained system only' if constraint == 'box' else None}


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['absolute_mse', 'scaled_irls'], default='scaled_irls')
    parser.add_argument('--constraint', choices=['ray', 'box'], default='box')
    parser.add_argument('--output', default=str(Path(__file__).with_name('subspace_bounded_evidence.json')))
    args = parser.parse_args()
    output = {'scope': 'SYNTHETIC SINGLE-STEP DIAGNOSTIC; fixed scales/opacity, means+SH1 only; no production throughput claim',
              'mode': args.mode, 'damping': 0.003 if args.mode == 'absolute_mse' else '0.3 * floored curvature diagonal',
              'adam_lr': 0.005, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'constraint': args.constraint, 'seeds': [run(i, args.mode, args.constraint) for i in range(5)]}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    for row in output['seeds']:
        print(json.dumps({'seed': row['seed'], 'schur_error': row['schur_vs_dense_max_error'],
                          'fit_loss': {m: v['fit']['loss'] for m, v in row['methods'].items()},
                          'validation_loss': {m: v['validation']['loss'] for m, v in row['methods'].items()}}))
