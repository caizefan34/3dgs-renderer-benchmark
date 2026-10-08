#!/usr/bin/env python3
"""FUSED-HOTSPOT0 harness. Corrected SAT-AUDIT0 timing policy (preloaded).
Modes: clean (H1), census (H2 per-kernel), stats (H4 workload)."""
import sys, os, json, argparse, statistics, time, hashlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import utils
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Optim.Samplers.DatasetSamplers import DatasetSampler
_C = None; extract_settings = None

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def reconstruct_moments(g, value=1e-4):
    refs = {'moments_means': g.means, 'moments_sh_coefficients_0': g._sh_coefficients_0,
            'moments_sh_coefficients_rest': g._sh_coefficients_rest, 'moments_opacities': g._opacities,
            'moments_scales': g._scales, 'moments_rotations': g._rotations}
    for name, ref in refs.items():
        setattr(g, name, torch.full(tuple(ref.shape) + (2,), value, dtype=torch.float32, device='cuda'))

def build(config_path, checkpoint, preload_level=2):
    global _C, extract_settings
    Framework.setup(config_path=config_path, require_custom_config=True)
    trainer = MI.get_training_instance(method='FasterGSFused', checkpoint=None)
    from Methods.FasterGSFused.Renderer import extract_settings as _es
    extract_settings = _es
    import sys as _sys
    _rz = None
    for name, mod in _sys.modules.items():
        if name.endswith('FasterGSFusedCudaBackend.torch_bindings.rasterization'): _rz = mod; break
    if _rz is None or not hasattr(_rz, '_C'): raise RuntimeError('no _C')
    _C = _rz._C
    dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE, path=Framework.config.DATASET.PATH)
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
    cc = torch.stack([v.position for v in dataset])
    radius = (1.1 * torch.max(torch.linalg.norm(cc - torch.mean(cc, dim=0), dim=1))).item()
    g.training_setup(trainer, radius)
    trainer.DATA.PRELOADING_LEVEL = preload_level
    trainer._prepare_dataset(None, dataset)
    train_ds = dataset.train()
    sampler = DatasetSampler(dataset=train_ds, random=True)
    ad = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device='cuda'))
    model.train()
    return trainer, dataset, train_ds, sampler, ad, g

def make_iter(renderer, loss_fn):
    def _iter(train_ds, sampler, ad, g, iteration, ev=None, cpu=None):
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
        image, ad2 = renderer.render_image_training(
            view=view, update_densification_info=False, bg_color=bg_color,
            adam_step_count=iteration + 1, autograd_dummy=ad)
        end('forward')
        rec('loss'); loss = loss_fn(image, rgb_gt) + 0.0 * ad; end('loss')
        rec('backward'); loss.backward(); end('backward')
        return float(loss.detach().item()), int(g.means.shape[0]), view.camera_index
    return _iter

def summarize(xs):
    xs = sorted(xs); n = len(xs)
    return {'n': n, 'median_ms': statistics.median(xs), 'mean_ms': statistics.fmean(xs), 'min_ms': xs[0], 'max_ms': xs[-1],
            'p05_ms': xs[max(0,int(0.05*n))], 'p95_ms': xs[min(n-1,int(0.95*n))], 'std_ms': statistics.pstdev(xs), 'raw_ms': xs}
def ev_pair(): return [torch.cuda.Event(True), torch.cuda.Event(True)]
def gpu_pci():
    try: return subprocess.check_output(['nvidia-smi','--query-gpu=pci.bus_id','--format=csv,noheader'], text=True).strip().splitlines()[0].strip()
    except Exception: return 'unknown'

def run_clean(_iter, train_ds, sampler, ad, g, warmup, iters, base):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for i in range(warmup): _iter(train_ds, sampler, ad, g, base+i)
    torch.cuda.synchronize()
    seg = {k: [] for k in ['sample','bg','rgb','forward','loss','backward','total']}
    cpu_seg = {k: [] for k in ['sample','bg','rgb','forward','loss','backward']}
    nG = cam = None
    for i in range(iters):
        it = base + 100000 + i
        ev = {k: ev_pair() for k in seg}
        cpu = {k: [0.0,0.0] for k in cpu_seg}
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True); t0.record()
        _, nG, cam = _iter(train_ds, sampler, ad, g, it, ev=ev, cpu=cpu); t1.record()
        torch.cuda.synchronize()
        seg['total'].append(t0.elapsed_time(t1))
        for k in seg:
            if k=='total': continue
            seg[k].append(ev[k][0].elapsed_time(ev[k][1]))
        for k in cpu_seg:
            cpu_seg[k].append((cpu[k][1]-cpu[k][0])*1000.0)
    return seg, cpu_seg, nG, cam, int(torch.cuda.max_memory_allocated())

def stats_safe(config_path, checkpoint, scene, stage, out_path, preload=2):
    global _C, extract_settings
    trainer, dataset, train_ds, sampler, ad, g = build(config_path, checkpoint, preload)
    _C_local = _C
    train_list = list(train_ds)
    view = train_list[0]
    W = int(view.camera.width); H = int(view.camera.height)
    n_tiles_x = (W+15)//16; n_tiles_y = (H+15)//16; n_tiles = n_tiles_x*n_tiles_y
    n_prim = int(g.means.shape[0])
    res = {'schema':'FUSED_HOTSPOT0_WORKLOAD_STATS','scene':scene,'stage':stage,'mode':'C_stats',
           'probe_camera_index':int(view.camera_index),'image_wh':[W,H],'tile_size':16,
           'n_tiles_image':n_tiles,'n_tiles_xy':[n_tiles_x,n_tiles_y],'n_gaussians_total':n_prim,
           'n_instances_tile_intersections':None,'n_buckets':None,'surviving_instance_fraction':None,
           'mean_instances_per_tile':None,'mean_instances_per_bucket':None,'bucket_occupancy':None,
           'probe_status':'ok','probe_note':''}
    try:
        settings = extract_settings(view, g.active_sh_bases, view.camera.background_color, g.lr_means, 1)
        out = _C_local.forward(g.means, g.scales, g.rotations, g.opacities, g.sh_coefficients_0, g.sh_coefficients_rest, *settings.as_tuple())
        n_instances = int(out[5]); n_buckets = int(out[6])
        res['n_instances_tile_intersections']=n_instances; res['n_buckets']=n_buckets
        res['surviving_instance_fraction']=round(n_instances/max(1,n_prim),4)
        res['mean_instances_per_tile']=round(n_instances/max(1,n_tiles),3)
        if n_buckets>0:
            res['mean_instances_per_bucket']=round(n_instances/n_buckets,3)
            res['bucket_occupancy']=round(n_instances/n_buckets,3)
        res['probe_note']='n_instances = surviving per-tile intersections after intrinsic exact reject (will_primitive_contribute), single probe camera. Full per-tile list-length percentiles require retaining instance_buffers (OOM-prone); not measured.'
    except Exception as e:
        res['probe_status']='unavailable'; res['probe_note']='forward probe failed: '+type(e).__name__+': '+str(e)[:200]
        res['n_instances_note']='production forward materializes full instance buffers; overflows int32 / OOM at this scene size'
    json.dump(res, open(out_path,'w'), indent=2)
    print('WSTATS', scene, stage, res['probe_status'], 'nG=%d ninst=%s'%(n_prim, res['n_instances_tile_intersections']), flush=True)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--scene', required=True); ap.add_argument('--stage', required=True, choices=['s1_5k','s2_10k','s3_mature'])
    ap.add_argument('--mode', required=True, choices=['clean','census','stats'])
    ap.add_argument('--warmup', type=int, default=50); ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    gpu = gpu_pci()
    if args.mode == 'clean':
        trainer, dataset, train_ds, sampler, ad, g = build(args.config, args.checkpoint, 2)
        _iter = make_iter(trainer.renderer, trainer.loss)
        seg, cpu_seg, nG, cam, vram = run_clean(_iter, train_ds, sampler, ad, g, args.warmup, args.iters, 200000)
        out = {'schema':'FUSED_HOTSPOT0_CLEAN_TIMING','scene':args.scene,'stage':args.stage,'mode':'A_clean_preloaded_PL2',
               'preloading_level':2,'checkpoint':args.checkpoint,'n_gaussians':nG,'camera_index_example':cam,
               'warmup':args.warmup,'iters':args.iters,'gpu_pci_bus_id':gpu,'peak_vram_bytes':vram,
               'segment_medians_ms':{k:round(summarize(v)['median_ms'],4) for k,v in seg.items()},
               'cpu_segment_medians_ms':{k:round(summarize(v)['median_ms'],4) for k,v in cpu_seg.items()},
               'median_total_ms':round(summarize(seg['total'])['median_ms'],4),
               'cuda_event_segments_ms':{k:summarize(v) for k,v in seg.items()}}
        json.dump(out, open(args.out,'w'), indent=2)
        print('WROTE',args.out,'total=%.3f fwd=%.3f bwd=%.3f loss=%.3f rgb=%.3f nG=%d vram=%.2fGB'%(
            out['median_total_ms'], summarize(seg['forward'])['median_ms'], summarize(seg['backward'])['median_ms'],
            summarize(seg['loss'])['median_ms'], summarize(seg['rgb'])['median_ms'], nG, vram/1e9))
    elif args.mode == 'census':
        trainer, dataset, train_ds, sampler, ad, g = build(args.config, args.checkpoint, 2)
        _iter = make_iter(trainer.renderer, trainer.loss)
        for i in range(args.warmup): _iter(train_ds, sampler, ad, g, 300000+i)
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        out_trace = args.out.replace('.json','_trace.json')
        N = min(args.iters, 30)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(N): _iter(train_ds, sampler, ad, g, 400000+i)
            torch.cuda.synchronize()
        prof.export_chrome_trace(out_trace)
        ka = prof.key_averages()
        rows = []
        for e in ka:
            ct = getattr(e,'self_cuda_time_total',0)
            if ct and ct>0:
                rows.append({'name':e.key,'count':int(e.count),'self_cuda_time_us':int(ct)})
        rows.sort(key=lambda r:-r['self_cuda_time_us'])
        json.dump({'schema':'FUSED_HOTSPOT0_KERNEL_CENSUS','scene':args.scene,'stage':args.stage,'mode':'B_census',
                   'iters_profiled':N,'trace':out_trace,'kernels':rows}, open(args.out,'w'), indent=2)
        print('WROTE',args.out,'kernels',len(rows))
    elif args.mode == 'stats':
        stats_safe(args.config, args.checkpoint, args.scene, args.stage, args.out, 2)

if __name__ == '__main__':
    main()