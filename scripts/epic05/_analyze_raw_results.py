"""Analyze raw validation results."""
import json, os

results_dir = os.path.expanduser('~/3dgs-renderer-benchmark/results/epic05/official/raw')
files = sorted(os.listdir(results_dir))
latest = os.path.join(results_dir, files[-1])
with open(latest) as f:
    data = json.load(f)

for scene_name, scene_data in data['scenes'].items():
    print('=== ' + scene_name + ' ===')
    if 'error' in scene_data:
        print('  ERROR: ' + scene_data['error'])
        continue
    
    for tile_key in sorted(scene_data['tile_results'].keys()):
        tr = scene_data['tile_results'][tile_key]
        t = tr['tile_size']
        stats = tr['timing_stats']
        print('  tile_size=%d: mean=%.2fms FPS=%.1f P99=%.2fms VRAM=%.0fMB CV=%.4f' % (
            t, stats['mean_ms'], stats['fps'], stats['p99_ms'], stats['vram_mb'], stats['cv']))
        for i, r in enumerate(tr['repeats']):
            print('    Repeat %d: %.2fms (frames=%d P99=%.2fms)' % (
                i+1, r['mean_ms'], r['num_frames'], r['p99_ms']))
    
    # Speedup comparison
    tile_results = scene_data['tile_results']
    if 'tile16' in tile_results and 'tile32' in tile_results:
        m16 = tile_results['tile16']['timing_stats']['mean_ms']
        m32 = tile_results['tile32']['timing_stats']['mean_ms']
        print('  tile32 vs tile16: %.4fx' % (m16 / m32))
    if 'tile8' in tile_results and 'tile16' in tile_results:
        m8 = tile_results['tile8']['timing_stats']['mean_ms']
        m16 = tile_results['tile16']['timing_stats']['mean_ms']
        print('  tile16 vs tile8: %.4fx' % (m8 / m16))
    print()
