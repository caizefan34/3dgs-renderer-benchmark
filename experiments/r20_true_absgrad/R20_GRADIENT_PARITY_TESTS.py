#!/usr/bin/env python3
"""Status: IMPLEMENTED; CPU checks executable; frozen MX parity NOT_EXECUTED.

Run --cpu for local semantics/API tests. Run --gpu with TWO isolated project
roots for frozen native vs R20 parity. Workers never modify source/checkpoints.
Real-scene input is a read-only torch fixture; see the build instructions.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
STD_SHA = '40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50'
HIGS_SHA = 'b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3'
FIELDS = ('xyz', 'rotation', 'scaling', 'opacity', 'shs')
THRESHOLD = 0.0008


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def metrics(reference, candidate):
    a, b = reference.double(), candidate.double()
    if a.shape != b.shape:
        raise AssertionError(f'shape mismatch {a.shape} != {b.shape}')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError('nonfinite values')
    d = b - a
    if a.numel() == 0:
        return dict(mae=0., rmse=0., max_abs=0., relative_l2=0., cosine=1.)
    an, bn = a.norm().item(), b.norm().item()
    cosine = (a.flatten().dot(b.flatten()).item() / (an * bn)
              if an and bn else (1. if not an and not bn else 0.))
    return {'mae': d.abs().mean().item(), 'rmse': d.square().mean().sqrt().item(),
            'max_abs': d.abs().max().item(), 'relative_l2': d.norm().item()/max(an, 1e-12),
            'cosine': cosine, 'relative_element_mean': (d.abs()/a.abs().clamp_min(1e-12)).mean().item()}


def score(absgrad, width, height):
    # Pixel-coordinate gradients -> derivatives w.r.t. normalized [-1,1]
    # coordinates. Norm is AFTER componentwise pixel absolute accumulation.
    scaled = absgrad * absgrad.new_tensor([width/2., height/2.])
    return scaled.norm(dim=-1).sum(dim=0)  # single view matches reference_v1


def candidates(capture, fixture):
    grads = score(capture['absgrad'], fixture['width'], fixture['height'])
    high = grads >= THRESHOLD
    scale_max = fixture['scaling'].exp().max(dim=-1).values
    extent = fixture.get('extent', 20.)
    small = scale_max <= 0.01 * extent
    opacity = fixture['opacity'].sigmoid().reshape(-1)
    radii = capture['radii'].reshape(-1, len(scale_max), 2).max(dim=0).values.max(dim=-1).values
    # Parent candidates on an IDENTICAL fixed topology. Not post-split children.
    prune = opacity < 0.005
    if fixture.get('max_screen_size'):
        prune = prune | (radii > fixture['max_screen_size']) | (scale_max > .1*extent)
    return {'threshold': high, 'clone': high & small, 'split': high & ~small,
            'prune_parent': prune, 'scores': grads}


def synthetic_fixtures():
    torch.manual_seed(42)
    names = ['same_direction', 'cancellation', 'single_gaussian', 'overlap',
             'tile_boundary', 'invisible_mixed', 'all_invisible', 'multi_camera',
             'alpha_loss', 'alpha_cap', 'threshold_low', 'threshold_high', 'sh3_chain']
    fixtures = {}
    for name in names:
        n = 7 if name in ('overlap', 'invisible_mixed', 'sh3_chain') else 1
        w, h = (33, 19) if name == 'tile_boundary' else (32, 32)
        xyz = torch.zeros(n, 3); xyz[:, 2] = 3.
        if n > 1:
            xyz[:, :2] = torch.randn(n, 2)*.08
            xyz[:, 2] += torch.arange(n)*.03
        if name == 'invisible_mixed':
            xyz[-2:, 2] = -3.
        if name == 'all_invisible':
            xyz[:, 2] = -3.
        rotation = torch.tensor([[1., .03, -.02, .01]]).repeat(n, 1)
        if name in ('same_direction', 'cancellation', 'threshold_low', 'threshold_high'):
            rotation = torch.tensor([[1., 0., 0., 0.]])
        scaling = torch.full((n, 3), .15).log()
        if n > 1:
            scaling[1] = torch.tensor([.3, .2, .25]).log()
        opacity = torch.full((n,), torch.logit(torch.tensor(.4)))
        if name == 'alpha_cap':
            scaling.fill_(2.)
            opacity.fill_(12.)
        degree = 3 if name == 'sh3_chain' else 0
        shs = torch.zeros(n, (degree+1)**2, 3)
        shs[:, 0, :] = (.7-.5)/.28209479177387814
        if degree:
            shs[:, 1:, :] = torch.randn(n, 15, 3)*.03
        views = torch.eye(4)[None]
        if name == 'multi_camera':
            views = views.repeat(2, 1, 1); views[1, 0, 3] = .07
        c = len(views)
        ks = torch.tensor([[40., 0., w/2.], [0., 40., h/2.], [0., 0., 1.]])[None].repeat(c, 1, 1)
        vc = torch.randn(c, h, w, 3)/max(w*h*3, 1)
        va = torch.randn(c, h, w, 1)/max(w*h, 1)*.1
        if name in ('same_direction', 'cancellation', 'threshold_low', 'threshold_high'):
            vc.zero_(); va.zero_()
            xs = (w//2-2, w//2-1) if name == 'same_direction' else (w//2-1, w//2)
            for x in xs:
                vc[0, h//2-1, x, 0] = 1.
            if name.startswith('threshold_'):
                cov = (40*.15/3.)**2 + .1
                vis = torch.exp(torch.tensor(-.25/cov)).item()
                norm = .4*.7*vis/cov * ((w/2.)**2+(h/2.)**2)**.5
                vc *= THRESHOLD/norm * (.99 if name.endswith('low') else 1.01)
        if name == 'alpha_loss':
            vc.zero_(); va.fill_(1./(h*w))
        fixtures[name] = dict(xyz=xyz, rotation=rotation, scaling=scaling, opacity=opacity,
                              shs=shs, viewmats=views, Ks=ks, width=w, height=h,
                              sh_degree=degree, v_rgb=vc, v_alpha=va, eps2d=.1, extent=20.)
    return fixtures


def activate(fixture, device='cuda'):
    raw = {k: fixture[k].to(device).detach().clone().requires_grad_(True) for k in FIELDS}
    act = dict(means=raw['xyz'], quats=F.normalize(raw['rotation'], dim=-1),
               scales=raw['scaling'].exp(), opacities=raw['opacity'].sigmoid().reshape(-1),
               colors=raw['shs'])
    return raw, act


def render(fixture, act, backend, absgrad, use_higs_culling=True):
    common = dict(width=fixture['width'], height=fixture['height'],
                  sh_degree=fixture['sh_degree'], tile_size=16, eps2d=fixture.get('eps2d', .1),
                  near_plane=.01, far_plane=1e10, radius_clip=0., render_mode='RGB')
    views, ks = fixture['viewmats'].cuda(), fixture['Ks'].cuda()
    if backend == 'std':
        from gsplat import rasterization
        rgb, alpha, info = rasterization(**act, **common, viewmats=views, Ks=ks,
                                        packed=False, absgrad=absgrad)
        m2d, radii = info['means2d'], info['radii']
    else:
        from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
        kwargs = {'absgrad': absgrad} if backend != 'legacy' else {}
        out = rasterize_gaussian_higs_frozen(**act, **common, viewmats=views[None], Ks=ks[None],
                    freeze_topology=False, enable_culling=True, use_higs_culling=use_higs_culling,
                    backward_mode='higs_native', **kwargs)
        if out['metadata']['backward_backend'] != 'higs_native':
            raise AssertionError('HiGS silently fell back to recomputation')
        info = out['densification_info']
        m2d, radii = info['means2d'], info['radii']
        rgb = out['frame'].reshape(len(views), fixture['height'], fixture['width'], 3)
        alpha = out['alpha'].reshape(len(views), fixture['height'], fixture['width'], 1)
    m2d.retain_grad()
    return rgb, alpha, m2d, radii


def backward_loss(fixture, rgb, alpha):
    # Identical upstream cotangents. No extra mean, camera factor, AMP or clamp.
    return (rgb*fixture['v_rgb'].cuda()).sum() + (alpha*fixture['v_alpha'].cuda()).sum()


def capture(fixture, backend, absgrad, use_higs_culling=True):
    raw, act = activate(fixture)
    rgb, alpha, m2d, radii = render(fixture, act, backend, absgrad, use_higs_culling)
    backward_loss(fixture, rgb, alpha).backward()
    torch.cuda.synchronize()
    signed = m2d.grad
    if signed is None:
        raise AssertionError('missing signed means2d gradient')
    result = dict(rgb=rgb.detach().cpu(), alpha=alpha.detach().cpu(),
                  signed=signed.detach().cpu(), radii=radii.detach().cpu())
    ag = getattr(m2d, 'absgrad', None)
    if absgrad:
        if ag is None or ag.shape != m2d.shape or ag.dtype != torch.float32:
            raise AssertionError('missing/wrong absgrad contract')
        if ag.requires_grad or not torch.isfinite(ag).all() or (ag < 0).any():
            raise AssertionError('invalid absgrad tensor')
        result['absgrad'] = ag.detach().cpu()
    elif ag is not None:
        raise AssertionError('absgrad=False unexpectedly exposes absgrad')
    for k in FIELDS:
        if raw[k].grad is None:
            raise AssertionError(f'missing raw gradient {k}')
        result['raw_'+k] = raw[k].grad.detach().cpu()
    return result


def worker(args):
    root = Path(args.root).resolve()
    std_file = root/'gsplat/csrc.so'
    if not std_file.is_file() or sha(std_file) != STD_SHA:
        raise RuntimeError('frozen std_ll binary SHA256 mismatch/missing')
    sys.path.insert(0, str(root))
    import gsplat
    if root not in Path(gsplat.__file__).resolve().parents:
        raise RuntimeError('wrong gsplat import root')
    if args.backend in ('higs', 'legacy'):
        from gsplat.experimental.render.kernels import _backend
        ext = _backend._C
        if args.backend == 'higs' and getattr(ext, 'higs_true_absgrad_version', 0) != 1:
            raise RuntimeError('R20 extension missing')
        if args.backend == 'legacy' and sha(Path(ext.__file__)) != HIGS_SHA:
            raise RuntimeError('frozen HiGS binary mismatch')
    fixtures = torch.load(args.fixtures, map_location='cpu', weights_only=True)
    results = {}
    for name, fixture in fixtures.items():
        print(name, flush=True)
        results[name] = capture(fixture, args.backend, args.absgrad, not args.projection_culling)
    torch.save({'status': 'EXECUTED', 'results': results,
                'identity': dict(root=str(root), std_sha=sha(std_file), torch=str(torch.__version__),
                                 device=torch.cuda.get_device_name(), px=os.getenv('HIGS_PX_RUNTIME', '2'))}, args.output)


def compare(ref, candidate, fixtures, atol=2e-6, rtol=.002):
    report, passed = {}, True
    for name, f in fixtures.items():
        a, b = ref[name], candidate[name]
        fields = ['rgb', 'alpha', 'signed', *('raw_'+k for k in FIELDS)]
        if 'absgrad' in a:
            fields.append('absgrad')
        row = {}
        for field in fields:
            row[field] = metrics(a[field], b[field])
            row[field]['pass'] = bool(torch.allclose(a[field], b[field], atol=atol, rtol=rtol))
            passed &= row[field]['pass']
        row['radii_equal'] = bool(torch.equal(a['radii'].reshape(-1, len(f['xyz']), 2),
                                            b['radii'].reshape(-1, len(f['xyz']), 2)))
        passed &= row['radii_equal']
        if 'absgrad' in a:
            ca, cb = candidates(a, f), candidates(b, f)
            row['decisions'] = {}
            for key in ('threshold', 'clone', 'split', 'prune_parent'):
                eq = ca[key] == cb[key]
                row['decisions'][key] = dict(agreement=eq.double().mean().item(),
                    mismatched_ids=(~eq).nonzero().flatten().tolist(),
                    reference_count=ca[key].sum().item(), candidate_count=cb[key].sum().item())
                passed &= bool(eq.all())
            k = min(1000, len(f['xyz']))
            # Stable Gaussian-id tie breaking for zero/tied scores.
            ia = set(torch.argsort(ca['scores'], descending=True, stable=True)[:k].tolist())
            ib = set(torch.argsort(cb['scores'], descending=True, stable=True)[:k].tolist())
            row['top_k'] = dict(k=k, overlap=len(ia & ib)/k)
            if k < len(f['xyz']):
                passed &= row['top_k']['overlap'] >= .99
            if name.startswith('threshold_'):
                expected = name.endswith('high')
                row['threshold_crossing_pass'] = bool(ca['threshold'][0]) == expected and bool(cb['threshold'][0]) == expected
                passed &= row['threshold_crossing_pass']
            if name == 'cancellation':
                row['cancellation_pass'] = bool(a['absgrad'][0, 0, 0] > a['signed'][0, 0, 0].abs()*100)
                passed &= row['cancellation_pass']
            if name == 'all_invisible':
                row['zero_absgrad_pass'] = bool((a['absgrad'] == 0).all() and (b['absgrad'] == 0).all())
                passed &= row['zero_absgrad_pass']
        report[name] = row
    return bool(passed), report


def gpu_suite(args):
    if not args.reference_root or not args.candidate_root:
        write_json(args.output, dict(status='NOT_EXECUTED', reason='two isolated MX roots required'))
        return 2
    out = Path(args.output).resolve().parent; out.mkdir(parents=True, exist_ok=True)
    fixtures = synthetic_fixtures()
    if args.real_fixture:
        real = torch.load(args.real_fixture, map_location='cpu', weights_only=True)
        fixtures['real_scene'] = real
    fixture_path = out/'fixtures.pt'; torch.save(fixtures, fixture_path)
    captures = {}
    for tag, backend, root, absgrad in (
        ('reference', 'std', args.reference_root, True),
        ('candidate_true', 'higs', args.candidate_root, True),
        ('candidate_false', 'higs', args.candidate_root, False),
        ('legacy_false', 'legacy', args.reference_root, False)):
        dest = out/(tag+'.pt')
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--root', root,
               '--backend', backend, '--fixtures', str(fixture_path), '--output', str(dest)]
        if absgrad: cmd.append('--absgrad')
        if args.projection_culling: cmd.append('--projection-culling')
        env = os.environ.copy(); env.pop('PYTHONPATH', None)
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out/(tag+'.log')).write_text(p.stdout, encoding='utf-8')
        if p.returncode:
            write_json(args.output, dict(status='FAIL', worker=tag, exit_code=p.returncode))
            return 1
        captures[tag] = torch.load(dest, map_location='cpu', weights_only=True)
    ok, comparison = compare(captures['reference']['results'], captures['candidate_true']['results'], fixtures, args.atol, args.rtol)
    same, unchanged = compare(captures['legacy_false']['results'], captures['candidate_false']['results'], fixtures, args.atol, args.rtol)
    # Absgrad is a side channel: signed/parameter gradients and forward must agree ON/OFF.
    side = {}
    for name in fixtures:
        a, b = captures['candidate_true']['results'][name], captures['candidate_false']['results'][name]
        side[name] = {k: bool(torch.allclose(a[k], b[k], atol=args.atol, rtol=args.rtol))
                      for k in ('rgb', 'alpha', 'signed', *('raw_'+x for x in FIELDS))}
    ok &= same and all(all(x.values()) for x in side.values())
    status = ('PASS' if args.real_fixture else 'PASS_SYNTHETIC_REAL_NOT_EXECUTED') if ok else 'FAIL'
    write_json(args.output, dict(status=status, comparison=comparison, legacy_false=unchanged,
        side_channel=side, identity={k: v['identity'] for k, v in captures.items()},
        fixture_sha256=sha(fixture_path), real_scene='EXECUTED' if args.real_fixture else 'NOT_EXECUTED',
        full_training='NOT_EXECUTED', tolerances=dict(atol=args.atol, rtol=args.rtol)))
    return 0 if ok else 1


class LocalChecks(unittest.TestCase):
    def test_componentwise_cancellation(self):
        pixels = torch.tensor([[2., -3.], [-2., 3.]])
        self.assertTrue(torch.equal(pixels.sum(0).abs(), torch.zeros(2)))
        self.assertTrue(torch.equal(pixels.abs().sum(0), torch.tensor([4., 6.])))
        orthogonal = torch.tensor([[2., 0.], [0., 3.]])
        self.assertNotEqual(orthogonal.norm(dim=-1).sum().item(), orthogonal.abs().sum(0).norm().item())

    def test_px_cancellation(self):
        # Cancellation WITHIN one thread's two pixels, not just across warp lanes.
        pixels = torch.tensor([[[1., 2.], [-1., -2.]], [[3., 4.], [-3., -4.]]])
        self.assertTrue((pixels.sum(1).abs().sum(0) == 0).all())
        self.assertTrue(torch.equal(pixels.abs().sum((0, 1)), torch.tensor([8., 12.])))

    def test_metrics_zero_and_error(self):
        z = torch.zeros(3, 2)
        self.assertEqual(metrics(z, z)['cosine'], 1.)
        self.assertEqual(metrics(z, torch.ones_like(z))['cosine'], 0.)
        with self.assertRaises(AssertionError): metrics(z, torch.ones(2))
        with self.assertRaises(AssertionError): metrics(z, z + float('nan'))

    def test_threshold_unchanged(self):
        f = synthetic_fixtures()['single_gaussian']
        cap = {'absgrad': torch.tensor([[[THRESHOLD/16., 0.]]]), 'radii': torch.ones(1, 1, 2)}
        self.assertTrue(candidates(cap, f)['clone'][0])
        cap['absgrad'] *= .99
        self.assertFalse(candidates(cap, f)['threshold'][0])

    def test_capture_safe_serialization(self):
        import io
        stream = io.BytesIO()
        data = {'identity': {'torch': str(torch.__version__)}, 'results': {'signed': torch.zeros(1, 1, 2)}}
        torch.save(data, stream); stream.seek(0)
        restored = torch.load(stream, weights_only=True)
        self.assertTrue(torch.equal(restored['results']['signed'], data['results']['signed']))

    def test_wrapper_scatter_and_arity(self):
        # Execute the actual modified wrapper backward with a CPU mock extension.
        import types
        source = (HERE/'source/gsplat/experimental/render/functional/gaussian_inference.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_HigsAutogradFunction')
        funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_native_backward', '_assemble_grads')]
        native = funcs[0]
        native.body = [n for n in native.body if not isinstance(n, ast.ImportFrom)]
        mini = ast.Module(body=[ast.ClassDef(name='_HigsAutogradFunction', bases=[], keywords=[], body=funcs, decorator_list=[])], type_ignores=[])
        ast.fix_missing_locations(mini)
        c = 2; visible = torch.tensor([0, 2]); m = 3
        proxy = torch.zeros(c, m, 2)
        signed = torch.arange(c*2*2).reshape(-1, 2).float()
        absolute = signed+10
        backend = types.SimpleNamespace(higs_rasterize_backward_with_absgrad=lambda **kw: (*[torch.empty(0)]*5, torch.zeros(3), signed, absolute),
                                        higs_rasterize_backward=lambda **kw: (*[torch.empty(0)]*5, torch.zeros(3), signed))
        ns = {'torch': torch, '_inference_backend': types.SimpleNamespace(_C=backend), '_render_mode_channel_count': lambda _: 3}
        exec(compile(mini, '<actual-wrapper-backward>', 'exec'), ns)
        fn = ns['_HigsAutogradFunction']
        expected_arity = len(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward').args.args)-1
        saved = [torch.zeros(m, 3), torch.zeros(m, 4), torch.zeros(m, 3), torch.zeros(m), torch.zeros(m, 3), proxy,
                 torch.eye(4)[None, None], torch.eye(3)[None, None], visible,
                 *[torch.empty(0)]*9, None, *[torch.empty(0)]*4, None, None]
        self.assertEqual(len(saved), 25)
        ctx = types.SimpleNamespace(saved_tensors=saved, C=c, height=1, width=1, render_mode='RGB',
                 N_visible=2, sampled_tile_ratio=1., tile_size=16, eps2d=.1, camera_model='pinhole',
                 sh_degree=None, needs_input_grad=[True]*expected_arity, absgrad=True)
        result = fn._native_backward(ctx, torch.ones(c, 1, 1, 3), None)
        self.assertEqual(len(result), expected_arity)
        self.assertTrue(torch.equal(proxy.absgrad[:, visible], absolute.reshape(c, 2, 2)))
        self.assertTrue((proxy.absgrad[:, 1] == 0).all())
        self.assertTrue(torch.equal(result[5][:, visible], signed.reshape(c, 2, 2)))
        ctx.N_visible = 0
        fn._native_backward(ctx, None, None)
        self.assertTrue((proxy.absgrad == 0).all())
        ctx.absgrad = False; ctx.N_visible = 2
        del proxy.absgrad
        fn._native_backward(ctx, None, None)
        self.assertFalse(hasattr(proxy, 'absgrad'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu', action='store_true'); p.add_argument('--gpu', action='store_true')
    p.add_argument('--worker', action='store_true'); p.add_argument('--root')
    p.add_argument('--backend', choices=['std', 'higs', 'legacy']); p.add_argument('--absgrad', action='store_true')
    p.add_argument('--fixtures'); p.add_argument('--reference-root'); p.add_argument('--candidate-root')
    p.add_argument('--real-fixture'); p.add_argument('--projection-culling', action='store_true')
    p.add_argument('--atol', type=float, default=2e-6); p.add_argument('--rtol', type=float, default=.002)
    p.add_argument('--output', default=str(HERE/'R20_TEST_RESULTS.json'))
    args = p.parse_args()
    if args.worker:
        worker(args); return 0
    if args.gpu:
        return gpu_suite(args)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LocalChecks))
    write_json(args.output, dict(status='PASS' if result.wasSuccessful() else 'FAIL', cpu_tests=result.testsRun,
                cuda_frozen_parity='NOT_EXECUTED', mx_compile='NOT_EXECUTED', full_training='NOT_EXECUTED'))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
