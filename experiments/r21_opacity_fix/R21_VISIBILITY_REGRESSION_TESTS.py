#!/usr/bin/env python3
"""Status: IMPLEMENTED. CPU uses actual wrapper code with instrumented operators.
GPU mode requires both frozen MX binaries; its results are never implied by CPU PASS.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
import weakref

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REL = "gsplat/experimental/render/functional/gaussian_inference.py"
SOURCE = HERE / "source" / REL
R20 = REPO / "experiments/r20_true_absgrad"
sys.path.insert(0, str(R20))
import R20_GRADIENT_PARITY_TESTS as parity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def actual_namespace(path=None):
    """Execute selected AST nodes verbatim; replace only external CUDA operators."""
    path = SOURCE if path is None else path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_cull_gaussians", "_cull_gaussians_batched", "_cull_visible_cached",
             "_gather_visible_native", "_union_visible_mask_native", "_expand_background",
             "_render_mode_channel_count", "_render_mode_color_channels", "_render_mode_has_depth"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef)
                             and n.name == "_HigsAutogradFunction"))
    cls.bases = []
    nodes.append(cls)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    comp = types.SimpleNamespace(name="NONE")
    ns = dict(torch=torch, math=math, weakref=weakref, Tensor=torch.Tensor,
              __package__="r21test.functional", _normalize_sh_compression_mode=lambda _: comp,
              _get_sh_compression_none=lambda: comp, _higs_backend_available=lambda: True)
    exec(compile(module, str(path), "exec"), ns)
    return ns


class CapturedProjection(Exception):
    pass


class WrapperContracts(unittest.TestCase):
    def setUp(self):
        self.ns = actual_namespace()
        self.calls = []
        # Predetermined per-camera radii, not a substitute projection formula.
        self.radii = torch.tensor([[[[2, 3], [0, 0], [0, 0], [0, 0]],
                                    [[0, 0], [0, 0], [4, 5], [0, 0]]]], dtype=torch.int32)
        def project(**kw):
            self.calls.append(kw)
            self.assertIsNotNone(kw.get("opacities"), "projection omitted activated opacity")
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(kw["opacities"].shape, kw["means"].shape[:-1])
            if getattr(self, "stop_first", False): raise CapturedProjection()
            if len(self.calls) == 2 and getattr(self, "stop_compact", False):
                raise CapturedProjection()
            return self.radii, None, None, None, None
        wrapper = types.ModuleType("gsplat.cuda._wrapper")
        wrapper.fully_fused_projection = project
        wrapper.isect_tiles = wrapper.isect_offset_encode = wrapper._make_lazy_cuda_func = None
        rendering = types.ModuleType("gsplat.rendering")
        rendering._maybe_evaluate_sh = None
        backend = types.SimpleNamespace(_C=types.SimpleNamespace(higs_true_absgrad_version=1))
        self.modules = patch.dict(sys.modules, {"gsplat.cuda._wrapper": wrapper,
            "gsplat.rendering": rendering, "r21test.kernels": types.SimpleNamespace(_backend=backend)})
        self.modules.start(); self.addCleanup(self.modules.stop)
        self.means = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        self.quats = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
        self.scales = torch.ones(4, 3)
        self.opacity = torch.tensor([.21, .00001, .83, .004], requires_grad=True)
        self.views = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
        self.ks = torch.eye(3)[None, None].repeat(1, 2, 1, 1)
        self.handle = types.SimpleNamespace(_cull_cache={}, _fwd_count=0)

    def cull(self, cached=False, **kw):
        args = (self.means, self.quats, self.scales, self.views, self.ks, 32, 32)
        opts = dict(eps2d=.1, opacities=self.opacity); opts.update(kw)
        if cached:
            self.handle._fwd_count += 1
            return self.ns['_cull_visible_cached'](self.handle, 8, *args, **opts)
        return self.ns['_cull_gaussians_batched'](*args, **opts)

    def forward(self, absgrad, enable_culling=True, backward_mode='higs_native'):
        self.stop_compact = enable_culling
        ctx = types.SimpleNamespace(save_for_backward=lambda *xs: setattr(self, "saved", xs))
        return self.ns['_HigsAutogradFunction'].forward(ctx, self.means, self.quats,
            self.scales, self.opacity, torch.ones(4, 3), torch.zeros(2, 4, 2), self.views,
            self.ks, 32, 32, None, 16, .01, 1e10, 0., .1, None, "RGB", "pinhole",
            enable_culling, True, backward_mode=backward_mode, absgrad=absgrad)

    def test_master_activated_opacity_and_camera_union(self):
        ids, mask, ratio = self.cull()
        self.assertEqual(ids.tolist(), [0, 2]); self.assertEqual(ratio, .5)
        self.assertEqual(mask.tolist(), [True, False, True, False])
        self.assertTrue(torch.equal(self.calls[0]['opacities'], self.opacity[None]))
        self.assertEqual(self.calls[0]['eps2d'], .1)
        self.assertEqual(self.calls[0]['opacities'].dtype, torch.float32)

    def test_actual_forward_compact_index_absgrad_and_px(self):
        for ag in (False, True):
            for px in (0, 1, 2, 4):
                with self.subTest(absgrad=ag, px=px), patch.dict(os.environ, HIGS_PX_RUNTIME=str(px)):
                    self.calls.clear()
                    with self.assertRaises(CapturedProjection): self.forward(ag)
                    a, b = self.calls
                    self.assertTrue(torch.equal(b['opacities'], a['opacities'][:, [0, 2]]))
                    for k in ('means', 'quats', 'scales'):
                        self.assertTrue(torch.equal(b[k], a[k][:, [0, 2]]))
                    for k in ('eps2d', 'near_plane', 'far_plane', 'radius_clip', 'camera_model'):
                        self.assertEqual(a[k], b[k])
        # PX changes CUDA dispatch only; this test covers Python plumbing, not PX CUDA parity.

    def test_cache_only_reuses_identical_inputs(self):
        self.cull(True); self.cull(True); self.assertEqual(len(self.calls), 1)
        with torch.no_grad(): self.opacity.fill_(.01)
        self.cull(True); self.assertEqual(len(self.calls), 2)
        self.opacity = self.opacity.detach().clone()
        self.cull(True); self.assertEqual(len(self.calls), 3)

    def test_cache_camera_geometry_and_bounds_changes(self):
        self.cull(True)
        for t in (self.means, self.quats, self.scales, self.views, self.ks):
            before = len(self.calls); t.add_(.01); self.cull(True)
            self.assertEqual(len(self.calls), before + 1)
        self.cull(True, eps2d=.2); self.assertEqual(self.calls[-1]['eps2d'], .2)

    def test_cache_does_not_retain_activation_graph(self):
        self.cull(True); ref = weakref.ref(self.opacity)
        self.calls.clear(); self.opacity = None; gc.collect()
        self.assertIsNone(ref())

    def test_inference_mode_opacity_disables_cache(self):
        with torch.inference_mode(): self.opacity = torch.ones(4)
        self.cull(True); self.cull(True); self.assertEqual(len(self.calls), 2)

    def test_dynamic_count_and_empty_visibility(self):
        for n in (7, 1, 0, 3):
            self.means = torch.zeros(n, 3); self.quats = torch.ones(n, 4)
            self.scales = torch.ones(n, 3); self.opacity = torch.full((n,), .01)
            self.radii = torch.zeros(1, 2, n, 2, dtype=torch.int32)
            ids = self.cull(True)
            self.assertEqual(ids.numel(), 0)
            self.assertEqual(self.calls[-1]['opacities'].shape, (1, n))

    def test_empty_actual_forward(self):
        self.radii.zero_()
        for ag in (False, True):
            self.calls.clear(); rgb, alpha = self.forward(ag)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(torch.count_nonzero(rgb).item(), 0)
            self.assertEqual(torch.count_nonzero(alpha).item(), 0)

    def test_culling_disabled_still_passes_render_opacity(self):
        self.stop_first = True
        with self.assertRaises(CapturedProjection): self.forward(True, enable_culling=False)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(torch.equal(self.calls[0]['opacities'], self.opacity[None]))

    def test_recompute_fallback_receives_same_compact_opacity(self):
        def rasterization(**kw):
            self.assertTrue(torch.equal(kw['opacities'], self.opacity[None, [0, 2]]))
            self.assertEqual(kw['eps2d'], .1)
            raise CapturedProjection()
        sys.modules['gsplat.rendering'].rasterization = rasterization
        with self.assertRaises(CapturedProjection): self.forward(False, backward_mode='gsplat_recompute')

    def test_every_projection_and_cull_call_forwards_opacity(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        targets = {'fully_fused_projection', '_cull_gaussians_batched', '_cull_visible_cached'}
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id in targets]
        self.assertEqual(len(calls), 6)
        for call in calls:
            self.assertTrue(any(k.arg == 'opacities' and not isinstance(k.value, ast.Constant)
                                for k in call.keywords), f'line {call.lineno}')

    def test_backward_and_public_api_unchanged(self):
        def definitions(path):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_HigsAutogradFunction')
            result = {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name not in ('forward', '_native_forward_capture')}
            result.update({n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name.startswith('rasterize_gaussian_')})
            return result
        self.assertEqual(definitions(SOURCE), definitions(R20/'source'/REL))

    def test_real_bicycle_room_master_compact_opacity(self):
        records = json.loads((HERE/'evidence/KNOWN_MISMATCH_FIXTURES.json').read_text())
        for scene, record in records.items():
            with self.subTest(scene=scene):
                self.calls.clear()
                f = known_fixture(record)
                _, act = parity.activate(f, 'cpu')
                self.means, self.quats, self.scales = act['means'], act['quats'], act['scales']
                self.opacity = act['opacities']
                self.views, self.ks = f['viewmats'][None], f['Ks'][None]
                self.radii = torch.tensor(record['radii_std'], dtype=torch.int32)[None]
                ids, mask, _ = self.cull()
                self.assertEqual(mask.tolist(), record['std_visibility'])
                gathered = self.ns['_gather_visible_native'](self.means, self.quats, self.scales,
                    self.opacity, act['colors'], ids)
                self.assertTrue(torch.equal(gathered[3], self.opacity[ids]))
                self.assertEqual(sum(a != b for a,b in zip(record['std_visibility'], record['r20_visibility'])), 32)

    def test_loaded_binary_identity_rejects_other_path_and_wrong_bytes(self):
        import R21_PREPARE_CANDIDATE as prepare
        root = (HERE/'unused_runtime_fixture').resolve()
        module = types.SimpleNamespace(__file__=str(root/prepare.MAIN))
        with patch.dict(sys.modules, {'gsplat.csrc':module}), \
             patch.object(prepare, 'sha', return_value=prepare.STD_SHA), patch.object(sys, 'path', list(sys.path)):
            result = prepare.load_verified_binaries(root, higs=False)
            self.assertEqual(result['gsplat.csrc']['sha256'], prepare.STD_SHA)
            module.__file__ = str(root/'jit-cache/csrc.so')
            with self.assertRaisesRegex(ValueError, 'Loaded binary differs'):
                prepare.load_verified_binaries(root, higs=False)
            module.__file__ = str(root/prepare.MAIN)
            with patch.object(prepare, 'sha', return_value='different'):
                with self.assertRaisesRegex(ValueError, 'Binary bytes differ'):
                    prepare.load_verified_binaries(root, higs=False)

    def test_loaded_binary_import_failure_does_not_fallback(self):
        import R21_PREPARE_CANDIDATE as prepare
        with patch.object(prepare.importlib.util, 'module_from_spec', side_effect=ImportError('ABI mismatch')) as imp, \
             patch.object(prepare.importlib.util, 'spec_from_file_location',
                          return_value=types.SimpleNamespace(loader=object())), \
             patch.object(prepare, 'sha', return_value=prepare.STD_SHA), \
             patch.object(sys, 'path', list(sys.path)), patch.dict(sys.modules):
            sys.modules.pop('gsplat.csrc', None)
            with self.assertRaisesRegex(ImportError, 'ABI mismatch'):
                prepare.load_verified_binaries(HERE, higs=True)
            imp.assert_called_once()
            self.assertNotIn('gsplat.csrc', sys.modules)


def opacity_fixtures():
    cases = parity.synthetic_fixtures()
    base = cases['overlap']
    for name in ('near_zero_opacity', 'image_boundary', 'near_plane', 'large_radius', 'opacity_reset'):
        f = copy.deepcopy(base)
        if name == 'near_zero_opacity':
            f['opacity'] = torch.logit(torch.tensor([1e-8, .001, 1/255, .004, .01, .1, .99]))
        if name == 'image_boundary':
            f['xyz'][:, 0] = torch.tensor([-1.65, -1.4, -1.25, 1.25, 1.4, 1.65, 0.])
            f['opacity'] = torch.logit(torch.tensor([.005, .01, .1, .1, .01, .005, .9]))
        if name == 'near_plane': f['xyz'][:, 2] = torch.tensor([-.01, .001, .009, .01, .011, .1, 3.])
        if name == 'large_radius': f['scaling'].fill_(math.log(3.))
        if name == 'opacity_reset': f['opacity'].fill_(torch.logit(torch.tensor(.01)).item())
        cases[name] = f
    records = json.loads((HERE/'evidence/KNOWN_MISMATCH_FIXTURES.json').read_text())
    cases.update({'known_'+scene:known_fixture(record) for scene,record in records.items()})
    return cases


def known_fixture(record):
    f = {k:torch.tensor(v, dtype=torch.float32) for k,v in record['raw'].items()}
    cam = record['camera']; f['opacity'] = f['opacity'].reshape(-1)
    f.update(viewmats=torch.tensor(cam['viewmatrix'], dtype=torch.float32)[None],
             Ks=torch.tensor(cam['K'], dtype=torch.float32)[None], width=cam['width'], height=cam['height'],
             sh_degree=record['sh_degree'], eps2d=.1, extent=record['extent'],
             stable_ids=torch.tensor(record['stable_ids'], dtype=torch.int64))
    gen = torch.Generator().manual_seed(42)
    f['v_rgb'] = torch.randn(1, f['height'], f['width'], 3, generator=gen)/(f['height']*f['width']*3)
    f['v_alpha'] = torch.randn(1, f['height'], f['width'], 1, generator=gen)/(f['height']*f['width'])*.1
    return f


def state_fixture(path):
    """Use a read-only R21 actual state, preserving master/stable row order."""
    sys.path.insert(0, str(REPO/'experiments/r21_densification'))
    from R21_STATE import load_state
    state = load_state(path)
    if state['phase'] != 'pre_topology': raise ValueError('Use pre-topology same-state inputs')
    cam = state['camera']; raw = state['raw']
    f = {k: raw[k] for k in parity.FIELDS}
    f['opacity'] = f['opacity'].reshape(-1)
    f.update(viewmats=cam['viewmatrix'].reshape(1, 4, 4), Ks=cam['K'].reshape(1, 3, 3),
             width=cam['width'], height=cam['height'], sh_degree=state['active_sh_degree'],
             eps2d=.1, extent=state['scene_extent'], stable_ids=state['stable_ids'])
    gen = torch.Generator().manual_seed(42)
    f['v_rgb'] = torch.randn(1, f['height'], f['width'], 3, generator=gen)/(f['height']*f['width']*3)
    f['v_alpha'] = torch.randn(1, f['height'], f['width'], 1, generator=gen)/(f['height']*f['width'])*.1
    return f


def worker(args):
    root = Path(args.root).resolve()
    if sha(root/'gsplat/csrc.so') != parity.STD_SHA: raise ValueError('Wrong frozen main binary')
    if args.backend == 'higs':
        from R21_PREPARE_CANDIDATE import verify_candidate
        verify_candidate(root)
    from R21_PREPARE_CANDIDATE import load_verified_binaries
    loaded_binaries = load_verified_binaries(root, higs=args.backend == 'higs')
    sys.path.insert(0, str(root))
    import gsplat
    if root not in Path(gsplat.__file__).resolve().parents: raise ValueError('Wrong gsplat import root')
    from gsplat.cuda._wrapper import fully_fused_projection
    if args.backend == 'higs':
        from gsplat.experimental.render.kernels import _backend
        if getattr(_backend._C, 'higs_true_absgrad_version', 0) != 1: raise ValueError('True AbsGrad capability missing')
    fixtures = torch.load(args.fixtures, map_location='cpu', weights_only=True)
    results, projection = {}, {}
    for name, f in fixtures.items():
        print(name, flush=True)
        results[name] = parity.capture(f, args.backend, args.absgrad)
        _, act = parity.activate(f)
        kw = {k: act[k] for k in ('means', 'quats', 'scales')}
        kw.update(covars=None, viewmats=f['viewmats'].cuda(), Ks=f['Ks'].cuda(),
                  width=f['width'], height=f['height'], eps2d=.1, packed=False)
        with torch.no_grad():
            on = fully_fused_projection(**kw, opacities=act['opacities'])[0].cpu()
            off = fully_fused_projection(**kw, opacities=None)[0].cpu()  # diagnostic ablation only
        projection[name] = dict(on=on, off=off)
    dynamic = {}
    if args.backend == 'higs':
        from gsplat.experimental.render.functional import gaussian_inference as gi
        original = parity.render
        def dynamic_render(f, act, backend, ag, use_higs_culling=True):
            out = gi.rasterize_gaussian_higs_dynamic(**act, sh_degree=f['sh_degree'],
                viewmats=f['viewmats'].cuda(), Ks=f['Ks'].cuda(), width=f['width'], height=f['height'],
                eps2d=.1, absgrad=ag, cull_refresh_interval=8)
            info = out['densification_info']; m2d = info['means2d']; m2d.retain_grad()
            c = len(f['viewmats'])
            return (out['frame'].reshape(c,f['height'],f['width'],3),
                    out['alpha'].reshape(c,f['height'],f['width'],1), m2d, info['radii'])
        parity.render = dynamic_render
        try:
            previous_n = None
            for name in ('overlap', 'near_zero_opacity', 'opacity_reset', 'single_gaussian', 'all_invisible'):
                f = fixtures[name]
                if previous_n is not None and previous_n != len(f['xyz']): gi._HIGS_DYNAMIC_SCENE.mark_dirty()
                dynamic[name] = parity.capture(f, 'higs', args.absgrad)
                previous_n = len(f['xyz'])
        finally:
            parity.render = original
    torch.save(dict(results=results, projection=projection, dynamic=dynamic, root=str(root),
                    loaded_binaries=loaded_binaries,
                    device=torch.cuda.get_device_name(), px=os.getenv('HIGS_PX_RUNTIME')), args.output)


def gpu_suite(args):
    if not args.reference_root or not args.candidate_root:
        write_json(args.output, dict(status='NOT_EXECUTED', reason='Frozen reference and isolated candidate roots required'))
        return 2
    out = Path(args.output).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    if Path(args.output).exists(): raise FileExistsError('Use a new result path')
    cases = opacity_fixtures()
    for p in args.state: cases['state_'+Path(p).parent.name+'_'+Path(p).stem] = state_fixture(p)
    for p in args.fixture: cases['fixture_'+Path(p).stem] = torch.load(p, map_location='cpu', weights_only=True)
    if args.cameras == 4:
        for f in cases.values():
            c = len(f['viewmats']); ids = torch.arange(4) % c
            for k in ('viewmats', 'Ks', 'v_rgb', 'v_alpha'): f[k] = f[k][ids].clone()
            # Same Gaussian state, four reproducible distinct camera translations.
            f['viewmats'][:, 0, 3] += torch.arange(4)*.013
    else:
        for name,f in cases.items():
            if name != 'multi_camera':
                for k in ('viewmats', 'Ks', 'v_rgb', 'v_alpha'): f[k] = f[k][:1].clone()
    fixture_path = out/'fixtures.pt'; torch.save(cases, fixture_path)
    captures = {}
    for tag, backend, root, ag in [('std_true', 'std', args.reference_root, True),
                                 ('std_false', 'std', args.reference_root, False),
                                 ('higs_true', 'higs', args.candidate_root, True),
                                 ('higs_false', 'higs', args.candidate_root, False)]:
        dest = out/(tag+'.pt')
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--root', root,
               '--backend', backend, '--fixtures', str(fixture_path), '--output', str(dest)]
        if ag: cmd.append('--absgrad')
        env = os.environ.copy(); env.pop('PYTHONPATH', None)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        (out/(tag+'.log')).write_text(p.stdout+p.stderr, encoding='utf-8')
        if p.returncode:
            write_json(args.output, dict(status='FAIL', worker=tag, exit_code=p.returncode)); return 1
        captures[tag] = torch.load(dest, map_location='cpu', weights_only=True)
    # R20's named threshold/cancellation assertions assume its original single camera.
    comparison_cases = {('B4_'+k if args.cameras == 4 else k): v for k,v in cases.items()}
    def results(tag):
        return {('B4_'+k if args.cameras == 4 else k):v for k,v in captures[tag]['results'].items()}
    ok, true_report = parity.compare(results('std_true'), results('higs_true'), comparison_cases)
    false_ok, false_report = parity.compare(results('std_false'), results('higs_false'), comparison_cases)
    # Compare side channel ON/OFF without treating absent absgrad as an error.
    a = copy.copy(results('higs_true'))
    a = {k:{field:value for field,value in v.items() if field != 'absgrad'} for k,v in a.items()}
    side_ok, side_report = parity.compare(a, results('higs_false'), comparison_cases)
    dynamic_report = {}
    for tag in ('higs_true', 'higs_false'):
        sub = captures[tag]['dynamic']
        reference = {name:captures[tag.replace('higs','std')]['results'][name] for name in sub}
        dynamic_ok, dynamic_report[tag] = parity.compare(reference, sub, {name:cases[name] for name in sub})
        ok &= dynamic_ok
    visibility = {}
    for name, f in cases.items():
        n = len(f['xyz']); a = captures['std_true']['results'][name]['radii'].reshape(-1, n, 2)
        b = captures['higs_true']['results'][name]['radii'].reshape_as(a)
        av, bv = (a>0).all(-1).any(0), (b>0).all(-1).any(0)
        mismatch = (av != bv).nonzero().flatten()
        p = captures['std_true']['projection'][name]
        exact_projection = torch.equal(a, p['on'].reshape_as(a))
        ok &= not len(mismatch) and exact_projection
        visibility[name] = dict(id_set_equal=not len(mismatch), mismatch_rows=mismatch.tolist(),
            radii_mismatch_components=int((a!=b).sum()), native_projection_equal=exact_projection,
            radii_on_off_changed_components=int((p['on']!=p['off']).sum()),
            radii_delta_histogram={str(int(x)):int(c) for x,c in zip(*torch.unique(b-a, return_counts=True))})
    passed = bool(ok and false_ok and side_ok)
    write_json(args.output, dict(status='PASS_SCOPED' if passed else 'FAIL',
        scope='same-state supplied fixtures; full P4 historical/training gates remain separate',
        fixture_sha256=sha(fixture_path), cameras=args.cameras, px=os.getenv('HIGS_PX_RUNTIME', '2'),
        loaded_binaries={tag:c['loaded_binaries'] for tag,c in captures.items()},
        true_absgrad=true_report, signed_mode=false_report, side_channel=side_report,
        dynamic_topology=dynamic_report, visibility=visibility,
        history_replay='NOT_EXECUTED', training='NOT_EXECUTED'))
    return 0 if passed else 1


def main():
    global SOURCE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu', action='store_true'); ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--worker', action='store_true'); ap.add_argument('--source', type=Path)
    ap.add_argument('--root'); ap.add_argument('--backend', choices=('std','higs'))
    ap.add_argument('--fixtures'); ap.add_argument('--absgrad', action='store_true')
    ap.add_argument('--reference-root'); ap.add_argument('--candidate-root')
    ap.add_argument('--fixture', action='append', default=[]); ap.add_argument('--state', action='append', default=[])
    ap.add_argument('--cameras', type=int, choices=(1,4), default=1)
    ap.add_argument('--output', type=Path, default=HERE/'R21_LOCAL_VALIDATION.json')
    args = ap.parse_args()
    if args.source: SOURCE = args.source.resolve()
    if args.worker: worker(args); return 0
    if args.gpu: return gpu_suite(args)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WrapperContracts))
    print(log.getvalue())
    write_json(args.output, dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), torch=str(torch.__version__),
        scope='Actual Python wrapper AST; instrumented projection operators; CPU only',
        source_sha256=sha(SOURCE), log=log.getvalue(), cuda_parity='NOT_EXECUTED',
        mx_p4='NOT_EXECUTED', training_quality='NOT_EXECUTED'))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
