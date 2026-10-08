#!/usr/bin/env python3
"""SAT-AUDIT0 harness: audit the ~275-288ms non-kernel/framework time on
bicycle/garden in the frozen FasterGSFused profile.

Modes (SAME baseline + SAME iteration semantics):
  A  clean uninstrumented production (PRELOADING_LEVEL=0, reproduces FP1/FP2)
  B  preloaded-input production   (PRELOADING_LEVEL=2 VRAM / 1 RAM)
  C  minimal timeline trace       (torch.profiler chrome trace)
  D  controlled sync audit
  E  CPU profile (cProfile + rgb-alone timing)
Diagnostic-only. Does NOT modify production algorithm code.
"""
import sys, os, json, argparse, statistics, time, hashlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import utils  # noqa
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Optim.Samplers.DatasetSamplers import DatasetSampler

_C = None
extract_settings = None


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def reconstruct_moments(g, value=1e-4):
    refs = {
        'moments_means': g.means, 'moments_sh_coefficients_0': g._sh_coefficients_0,
        'moments_sh_coefficients_rest': g._sh_coefficients_rest, 'moments_opacities': g._opacities,
        'moments_scales': g._scales, 'moments_rotations': g._rotations,
    }
    for name, ref in refs.items():
        setattr(g, name, torch.full(tuple(ref.shape) + (2,), value, dtype=torch.float32, device='cuda'))


def build(config_path, checkpoint, preload_level):
    global _C, extract_settings
    Framework.setup(config_path=config_path, require_custom_config=True)
    trainer = MI.get_training_instance(method='FasterGSFused', checkpoint=None)
    from Methods.FasterGSFused.Renderer import extract_settings as _es
    extract_settings = _es
    import sys as _sys
    _rzmod = None
    for name, mod in _sys.modules.items():
        if name.endswith('FasterGSFusedCudaBackend.torch_bindings.rasterization'):
            _rzmod = mod
            break
    if _rzmod is None or not hasattr(_rzmod, '_C'):
        raise RuntimeError('could not locate loaded rasterization backend (_C)')
    _C = _rzmod._C
    dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
                            path=Framework.config.DATASET.PATH)
    dataset.train()
    ckpt = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model = trainer.model
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.num_iterations_trained = ckpt.get('num_iterations_trained', model.num_iterations_trained)
    g = model.gaussians
    if g.means is None:
        for k, v in ckpt['model_state_dict'].items():
            parts = k.split('.'); obj = model
            for part in parts[:-1]: obj = getattr(obj, part)
            setattr(obj, parts[-1], v)
    reconstruct_moments(g)
    camera_centers = torch.stack([v.position for v in dataset])
    radius = (1.1 * torch.max(torch.linalg.norm(
        camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
    g.training_setup(trainer, radius)
    trainer.DATA.PRELOADING_LEVEL = preload_level
    try:
        trainer._prepare_dataset(None, dataset)
    except Exception as e:
        print(f'WARNING: _prepare_dataset failed: {e}')
    train_ds = dataset.train()
    sampler = DatasetSampler(dataset=train_ds, random=True)
    autograd_dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cuda'))
    model.train()
    return trainer, dataset, train_ds, sampler, autograd_dummy, g


def make_iter(renderer, loss_fn):
    def _iter(train_ds, sampler, autograd_dummy, g, iteration, ev=None, cpu=None):
        def rec(k):
            if ev is not None: ev[k][0].record()
            if cpu is not None: cpu[k][0] = time.perf_counter()
        def end(k):
            if ev is not None: ev[k][1].record()
            if cpu is not None: cpu[k][1] = time.perf_counter()
        rec('sample'); view = sampler.get(dataset=train_ds)['view']; end('sample')
        rec('bg'); bg_color = view.camera.background_color; end('bg')
        rec('rgb'); rgb_gt = view.rgb; end('rgb')
        rec('forward')
        image, ad = renderer.render_image_training(
            view=view, update_densification_info=False, bg_color=bg_color,
            adam_step_count=iteration + 1, autograd_dummy=autograd_dummy)
        end('forward')
        rec('loss'); loss = loss_fn(image, rgb_gt) + 0.0 * autograd_dummy; end('loss')
        rec('backward'); loss.backward(); end('backward')
        return float(loss.detach().item()), int(g.means.shape[0]), view.camera_index, rgb_gt
    return _iter


def ev_pair():
    return [torch.cuda.Event(True), torch.cuda.Event(True)]


def summarize(xs):
    xs = sorted(xs); n = len(xs)
    return {'n': n, 'median_ms': statistics.median(xs), 'mean_ms': statistics.fmean(xs),
            'min_ms': xs[0], 'max_ms': xs[-1],
            'p05_ms': xs[max(0, int(0.05 * n))], 'p95_ms': xs[min(n - 1, int(0.95 * n))],
            'std_ms': statistics.pstdev(xs), 'raw_ms': xs}


def run_timed(_iter, train_ds, sampler, ad, g, warmup, iters, base_it):
    torch.cuda.synchronize()
    for i in range(warmup):
        _iter(train_ds, sampler, ad, g, base_it + i)
    torch.cuda.synchronize()
    seg = {k: [] for k in ['sample', 'bg', 'rgb', 'forward', 'loss', 'backward', 'total']}
    cpu_seg = {k: [] for k in ['sample', 'bg', 'rgb', 'forward', 'loss', 'backward']}
    nG = cam = None
    for i in range(iters):
        it = base_it + 100000 + i
        ev = {k: ev_pair() for k in seg}
        cpu = {k: [0.0, 0.0] for k in cpu_seg}
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record()
        _, nG, cam, _ = _iter(train_ds, sampler, ad, g, it, ev=ev, cpu=cpu)
        t1.record()
        torch.cuda.synchronize()
        seg['total'].append(t0.elapsed_time(t1))
        for k in seg:
            if k == 'total': continue
            seg[k].append(ev[k][0].elapsed_time(ev[k][1]))
        for k in cpu_seg:
            cpu_seg[k].append((cpu[k][1] - cpu[k][0]) * 1000.0)
    return seg, cpu_seg, nG, cam


def gpu_pci():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=pci.bus_id',
                                        '--format=csv,noheader'], text=True).strip().splitlines()[0].strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--mode', required=True,
                    choices=['clean', 'preload', 'preload_ram', 'timeline', 'sync', 'cpu'])
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    gpu_uuid = gpu_pci()

    if args.mode == 'clean':
        trainer, dataset, train_ds, sampler, ad, g = build(args.config, args.checkpoint, 0)
        _iter = make_iter(trainer.renderer, trainer.loss)
        seg, cpu_seg, nG, cam = run_timed(_iter, train_ds, sampler, ad, g, args.warmup, args.iters, 200000)
        med_total = summarize(seg['total'])['median_ms']
        med_segs = {k: summarize(v)['median_ms'] for k, v in seg.items() if k != 'total'}
        cpu_med = {k: summarize(v)['median_ms'] for k, v in cpu_seg.items()}
        out = {'schema': 'SAT_AUDIT0_CLEAN_TIMING', 'scene': args.scene,
               'mode': 'A_clean_uninstrumented_PL0', 'preloading_level': 0,
               'checkpoint': args.checkpoint, 'n_gaussians': nG, 'camera_index_example': cam,
               'warmup': args.warmup, 'iters': args.iters, 'gpu_pci_bus_id': gpu_uuid,
               'median_total_ms': round(med_total, 4),
               'segment_medians_ms': {k: round(v, 4) for k, v in med_segs.items()},
               'cpu_perf_segment_medians_ms': {k: round(v, 4) for k, v in cpu_med.items()},
               'sum_segment_medians_ms': round(sum(med_segs[k] for k in ['sample','bg','rgb','forward','loss','backward']), 4),
               'unaccounted_gap_ms': round(med_total - sum(med_segs[k] for k in ['sample','bg','rgb','forward','loss','backward']), 4),
               'cuda_event_segments_ms': {k: summarize(v) for k, v in seg.items()},
               'cpu_perf_segments_ms': {k: summarize(v) for k, v in cpu_seg.items()}}
        out['gap_fraction'] = round(out['unaccounted_gap_ms'] / med_total, 4) if med_total else None
        json.dump(out, open(args.out, 'w'), indent=2)
        print('WROTE', args.out)
        print('  median_total=%.3f sum_segments=%.3f gap=%.3f (rgb cuda=%.3f cpu_rgb=%.3f)' % (
            med_total, out['sum_segment_medians_ms'], out['unaccounted_gap_ms'], med_segs['rgb'], cpu_med['rgb']))

    elif args.mode in ('preload', 'preload_ram'):
        level = 2 if args.mode == 'preload' else 1
        trainer, dataset, train_ds, sampler, ad, g = build(args.config, args.checkpoint, level)
        _iter = make_iter(trainer.renderer, trainer.loss)
        seg, cpu_seg, nG, cam = run_timed(_iter, train_ds, sampler, ad, g, args.warmup, args.iters, 200000)
        med_total = summarize(seg['total'])['median_ms']
        med_segs = {k: summarize(v)['median_ms'] for k, v in seg.items() if k != 'total'}
        out = {'schema': 'SAT_AUDIT0_PRELOAD_TEST', 'scene': args.scene,
               'mode': f'B_preloaded_level{level}', 'preloading_level': level,
               'checkpoint': args.checkpoint, 'n_gaussians': nG,
               'warmup': args.warmup, 'iters': args.iters, 'gpu_pci_bus_id': gpu_uuid,
               'median_total_ms': round(med_total, 4),
               'segment_medians_ms': {k: round(v, 4) for k, v in med_segs.items()},
               'cpu_perf_segment_medians_ms': {k: round(summarize(v)['median_ms'], 4) for k, v in cpu_seg.items()}}
        json.dump(out, open(args.out, 'w'), indent=2)
        print('WROTE', args.out, 'median_total=%.3f' % med_total)

    elif args.mode == 'timeline':
        trainer, dataset, train_ds, sampler, ad, g = build(args.config, args.checkpoint, 0)
        from torch.profiler import profile, ProfilerActivity
        for i in range(args.warmup):
            view = sampler.get(dataset=train_ds)['view']
            _ = view.rgb
        torch.cuda.synchronize()
        out_trace = args.out.replace('.json', '_trace.json')
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(min(args.iters, 20)):
                with torch.autograd.profiler.record_function('sample_view'):
                    view = sampler.get(dataset=train_ds)['view']
                with torch.autograd.profiler.record_function('image_load_view_rgb'):
                    rgb_gt = view.rgb
                with torch.autograd.profiler.record_function('forward_render'):
                    image, ad2 = trainer.renderer.render_image_training(
                        view=view, update_densification_info=False,
                        bg_color=view.camera.background_color,
                        adam_step_count=300000 + i + 1, autograd_dummy=ad)
                with torch.autograd.profiler.record_function('loss'):
                    loss = trainer.loss(image, rgb_gt) + 0.0 * ad
                with torch.autograd.profiler.record_function('backward'):
                    loss.backward()
            torch.cuda.synchronize()
        prof.export_chrome_trace(out_trace)
        ka = prof.key_averages()
        rows = []
        for e in ka:
            ct = getattr(e, 'self_cuda_time_total', 0); cput = getattr(e, 'self_cpu_time_total', 0)
            if ct or cput:
                rows.append({'name': e.key, 'count': int(e.count),
                             'self_cuda_time_us': int(ct), 'self_cpu_time_us': int(cput)})
        rows.sort(key=lambda r: -(r['self_cuda_time_us'] + r['self_cpu_time_us']))
        json.dump({'schema': 'SAT_AUDIT0_TIMELINE', 'scene': args.scene, 'mode': 'C_timeline',
                   'iters_profiled': min(args.iters, 20), 'trace': out_trace, 'rows': rows[:60]},
                  open(args.out, 'w'), indent=2)
        print('WROTE', args.out, 'trace', out_trace, 'rows', len(rows))

    elif args.mode == 'sync':
        trainer, dataset, train_ds, sampler, ad, g = build(args.config, args.checkpoint, 0)
        _iter = make_iter(trainer.renderer, trainer.loss)
        sync_calls = {'synchronize': 0, 'item': 0}
        orig_sync = torch.cuda.synchronize
        def sync_spy(*a, **k):
            sync_calls['synchronize'] += 1; return orig_sync(*a, **k)
        torch.cuda.synchronize = sync_spy
        orig_item = torch.Tensor.item
        def item_spy(self, *a, **k):
            sync_calls['item'] += 1; return orig_item(self, *a, **k)
        torch.Tensor.item = item_spy
        seg, cpu_seg, nG, cam = run_timed(_iter, train_ds, sampler, ad, g, args.warmup, args.iters, 200000)
        torch.cuda.synchronize = orig_sync; torch.Tensor.item = orig_item
        med_total = summarize(seg['total'])['median_ms']
        out = {'schema': 'SAT_AUDIT0_SYNC_AUDIT', 'scene': args.scene, 'mode': 'D_sync_audit',
               'checkpoint': args.checkpoint, 'iters': args.iters,
               'median_total_ms': round(med_total, 4),
               'sync_calls_per_iter': {'synchronize': round(sync_calls['synchronize'] / max(1, args.iters), 3),
                                        'item': round(sync_calls['item'] / max(1, args.iters), 3)},
               'note': 'Counts synchronize/.item during timed window. High => implicit sync stalls.',
               'segment_medians_ms': {k: round(summarize(v)['median_ms'], 4) for k, v in seg.items() if k != 'total'}}
        json.dump(out, open(args.out, 'w'), indent=2)
        print('WROTE', args.out, 'sync/iter', out['sync_calls_per_iter'])

    elif args.mode == 'cpu':
        trainer, dataset, train_ds, sampler, ad, g = build(args.config, args.checkpoint, 0)
        _iter = make_iter(trainer.renderer, trainer.loss)
        import cProfile, pstats, io
        for i in range(args.warmup):
            _iter(train_ds, sampler, ad, g, 400000 + i)
        torch.cuda.synchronize()
        pr = cProfile.Profile(); pr.enable()
        for i in range(min(args.iters, 30)):
            _iter(train_ds, sampler, ad, g, 400000 + i)
        torch.cuda.synchronize(); pr.disable()
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(30)
        text = s.getvalue()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(min(args.iters, 30)):
            view = sampler.get(dataset=train_ds)['view']; _ = view.rgb
        torch.cuda.synchronize()
        rgb_alone_ms = (time.perf_counter() - t) * 1000.0 / min(args.iters, 30)
        json.dump({'schema': 'SAT_AUDIT0_CPU_PROFILE', 'scene': args.scene, 'mode': 'E_cpu_profile',
                   'iters_profiled': min(args.iters, 30),
                   'rgb_load_alone_ms_per_iter': round(rgb_alone_ms, 4), 'cprofile_text': text},
                  open(args.out, 'w'), indent=2)
        print('WROTE', args.out, 'rgb_alone_ms=%.3f' % rgb_alone_ms)


if __name__ == '__main__':
    main()