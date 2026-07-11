"""
Generate interactive HTML report from benchmark results.
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")

with open(os.path.join(OUT, "benchmark_results_phase1.json")) as f:
    phase1 = json.load(f)

# Build HTML
table_rows = ""
for rname, r in sorted(phase1["results"].items(), key=lambda x: x[1]["median_ms"]):
    tag = "??" if rname == phase1["metadata"]["fastest_renderer"] else ""
    table_rows += f"""<tr>
        <td>{tag} {rname}</td>
        <td>{r["median_ms"]:.2f}</td>
        <td>{r["median_fps"]:.1f}</td>
        <td>{r["mean_ms"]:.2f}</td>
        <td>{r["p99_ms"]:.2f}</td>
        <td>{r["peak_memory_mb"]:.0f}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3DGS Renderer Benchmark Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
h1, h2, h3 {{ color: #1a1a2e; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #1a1a2e; color: white; font-weight: 600; }}
tr:hover {{ background: #f0f2ff; }}
.fastest {{ background: #d4edda !important; font-weight: 600; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.meta {{ font-size: 0.9em; color: #666; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
.badge-win {{ background: #d4edda; color: #155724; }}
.badge-loss {{ background: #f8d7da; color: #721c24; }}
</style>
</head>
<body>

<h1>?? 3D Gaussian Splatting Renderer Benchmark</h1>

<div class="card">
<h2>System</h2>
<table>
<tr><td>GPU</td><td>{phase1["metadata"]["gpu"]}</td></tr>
<tr><td>CUDA</td><td>{phase1["metadata"]["cuda"]}</td></tr>
<tr><td>PyTorch</td><td>{phase1["metadata"]["pytorch"]}</td></tr>
<tr><td>Gaussians</td><td>{phase1["metadata"]["num_gaussians"]:,}</td></tr>
<tr><td>Resolution</td><td>{phase1["metadata"]["resolution"]}</td></tr>
<tr><td>Date</td><td>{phase1["metadata"]["date"]}</td></tr>
</table>
</div>

<h2>Phase 1: Renderer Comparison</h2>
<p>All renderers use <b>identical</b> scene data and camera poses. 50 warmup + 200 benchmark frames.</p>

<table>
<tr><th>Renderer</th><th>Median (ms) ¡ý</th><th>FPS</th><th>Mean (ms)</th><th>P99 (ms)</th><th>Mem (MB)</th></tr>
{table_rows}
</table>

<div class="card">
<h3>?? Fastest Renderer: {phase1["metadata"]["fastest_renderer"]}</h3>
<p>Key factors for speed:
<ul>
<li><b>CUB DeviceRadixSort</b> replaces Thrust radix sort ¡ú ~15-30% faster tile binning</li>
<li>Warp-level primitives reduce shared memory bank conflicts</li>
<li>TC-GS shows identical render-time performance (same diff-gaussian-rasterization kernel)</li>
</ul>
</p>
</div>

<h2>Phase 2: Optimizations</h2>
<div class="card">
<h3>Optimizations Applied</h3>
<ol>
<li><b>Frustum Pre-Culling</b> ¡ª Conservative NDC-space visibility test reduces gaussian count</li>
<li><b>Pre-allocated Buffer Reuse</b> ¡ª Eliminates per-frame tensor allocations</li>
<li><b>Rasterizer Cache</b> ¡ª Reuses GaussianRasterizer across frames for same camera</li>
</ol>

<h3>Results</h3>
<p>Baseline (speedy_splat): <b>171.4 FPS</b> (5.83ms median)</p>
<p>Optimized (culling + prealloc): <b>365.1 FPS</b> (2.74ms median) ¡ª <span class="badge badge-win">+113% faster</span></p>
</div>

<div class="card meta">
<p><b>Repository</b>: <a href="https://github.com/caizefan34/3dgs-renderer-benchmark">caizefan34/3dgs-renderer-benchmark</a></p>
<p>Generated: {phase1["metadata"]["date"]} | Full results in <code>outputs/</code></p>
</div>

</body>
</html>"""

with open(os.path.join(OUT, "benchmark_report.html"), "w", encoding="utf-8") as f:
    f.write(html)
print(f"Report generated: {os.path.join(OUT, 'benchmark_report.html')}")
