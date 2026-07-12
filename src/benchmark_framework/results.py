"""
Results manager: collect, aggregate, and export benchmark results.
Supports JSON, CSV, Markdown, and HTML report generation.
"""
import json
import os
import csv
from typing import Dict, List
from .metrics import RendererMetrics


class ResultsManager:
    def __init__(self):
        self.results: Dict[str, RendererMetrics] = {}

    def add_result(self, renderer_name: str, metrics: RendererMetrics):
        self.results[renderer_name] = metrics

    def get_summary(self) -> dict:
        return {name: m.to_dict() for name, m in self.results.items()}

    def get_ranking(self, key="mean_fps") -> List[tuple]:
        rankings = [(name, m.mean_fps, m.mean_latency_ms)
                     for name, m in self.results.items()
                     if m.mean_fps > 0]
        rankings.sort(key=lambda x: x[1], reverse=(key == "mean_fps"))
        return rankings

    # --- Export ---

    def export_json(self, path: str):
        """Export full results as JSON."""
        data = self.get_summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Exported: {path}")

    def export_csv(self, path: str):
        """Export metrics summary as CSV."""
        rows = []
        for name, m in self.results.items():
            d = m.to_dict()
            row = {
                "renderer": name,
                "mean_fps": d["mean_fps"],
                "median_latency_ms": d["median_latency_ms"],
                "p1_latency_ms": d["p1_latency_ms"],
                "p5_latency_ms": d["p5_latency_ms"],
                "p95_latency_ms": d["p95_latency_ms"],
                "p99_latency_ms": d["p99_latency_ms"],
                "jitter_pct": d["jitter_pct"],
                "peak_vram_mb": d["peak_vram_mb"],
                "avg_vram_mb": d["avg_vram_mb"],
                "load_time_ms": d["scene_load_time_ms"],
                "file_size_mb": d["file_size_mb"],
                "psnr": d["psnr"],
                "ssim": d["ssim"],
                "lpips": d["lpips"],
                "num_gaussians": d["num_gaussians"],
            }
            rows.append(row)
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Exported: {path}")

    def export_markdown(self, path: str):
        """Generate benchmark report in Markdown."""
        rankings = self.get_ranking()
        fastest = rankings[0][0] if rankings else ""

        lines = [
            "# 3DGS Renderer Benchmark Report",
            "",
            "## Summary",
            "",
            "| Rank | Renderer | Mean FPS | Median (ms) | P1 (ms)↓ | P99 (ms) | Jitter% | VRAM(MB) |",
            "|------|----------|:--------:|:-----------:|:--------:|:--------:|:-------:|:--------:|",
        ]
        for i, (name, fps, lat) in enumerate(rankings, 1):
            m = self.results[name]
            tag = " ★" if name == fastest else ""
            lines.append(
                f"| {i}{tag} | {name} | {m.mean_fps:.1f} | {m.median_latency_ms:.2f} | "
                f"{m.p1_latency_ms:.2f} | {m.p99_latency_ms:.2f} | {m.jitter_ms:.1f} | "
                f"{m.peak_vram_mb:.0f} |"
            )

        lines += ["", "## Per-Renderer Details", ""]
        for rname in sorted(self.results.keys()):
            m = self.results[rname]
            d = m.to_dict()
            tag = " (fastest)" if rname == fastest else ""
            lines += [
                f"### {rname}{tag}",
                f"- **FPS**: mean={m.mean_fps}, P5={m.p5_fps}, P95={m.p95_fps}",
                f"- **Latency**: mean={m.mean_latency_ms}ms, median={m.median_latency_ms}ms, "
                f"P99={m.p99_latency_ms}ms",
                f"- **Jitter**: {m.jitter_ms:.1f}%",
                f"- **VRAM**: peak={m.peak_vram_mb:.0f}MB, avg={m.avg_vram_mb:.0f}MB",
                f"- **Quality**: PSNR={m.psnr:.2f}, SSIM={m.ssim:.4f}, LPIPS={m.lpips:.4f}",
                f"- **Scene**: {d['num_gaussians']:,} gaussians, {d['file_size_mb']:.1f}MB",
                f"- **Load Time**: {d['scene_load_time_ms']:.1f}ms",
                "",
            ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")
        print(f"  Exported: {path}")

    def export_html(self, path: str, title="3DGS Renderer Benchmark"):
        """Generate an interactive HTML report with Plotly charts."""
        rankings = self.get_ranking()
        fastest = rankings[0][0] if rankings else ""

        # Build table
        trows = ""
        for i, (name, fps, lat) in enumerate(rankings, 1):
            m = self.results[name]
            cls = ' class="fastest"' if name == fastest else ""
            tag = "&#9733;" if name == fastest else ""
            trows += f"""<tr{cls}>
            <td>{tag} {name}</td>
            <td>{m.mean_fps:.1f}</td>
            <td>{m.median_latency_ms:.2f}</td>
            <td>{m.p1_latency_ms:.2f}</td>
            <td>{m.p99_latency_ms:.2f}</td>
            <td>{m.jitter_ms:.1f}%</td>
            <td>{m.peak_vram_mb:.0f}</td>
        </tr>"""

        # Build chart data
        chart_data = json.dumps({
            name: m.frame_times_ms for name, m in self.results.items() if m.frame_times_ms
        })

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><{""}/script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #0f172a; color: #e2e8f0; }}
h1, h2, h3 {{ color: #f8fafc; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; background: #1e293b; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
th {{ background: #0f172a; color: #60a5fa; font-weight: 600; }}
tr:hover {{ background: #1e3a5f; }}
.fastest {{ background: #065f4620 !important; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin: 16px 0; border: 1px solid #334155; }}
.plot {{ width: 100%; height: 400px; }}
.meta {{ font-size: 0.9em; color: #94a3b8; }}
</style>
</head>
<body>
<h1>{"&#9733; " if fastest else ""}{title}</h1>

<div class="card">
<h2>Summary</h2>
<table>
<tr><th>Rank</th><th>Renderer</th><th>Mean FPS</th><th>Median (ms)</th><th>P1 (ms)</th><th>P99 (ms)</th><th>Jitter</th><th>VRAM</th></tr>
{trows}
</table>
</div>

<div class="card">
<h2>Frame Time Timeline</h2>
<div id="chart" class="plot"></div>
</div>

<div class="card meta">
<p><b>GPU</b>: {self.results[rankings[0][0]].gpu_name if rankings else "N/A"} |
<b>Resolution</b>: {self.results[rankings[0][0]].image_width}x{self.results[rankings[0][0]].image_height if rankings else "N/A"} |
<b>Frames</b>: {rankings[0][2] if rankings else "N/A"}</p>
</div>

<script>
var raw = {chart_data};
var traces = [];
for (var rname in raw) {{
    traces.push({{
        y: raw[rname], type: 'scatter', mode: 'lines', name: rname,
        line: {{ width: 1.5 }}
    }});
}}
var layout = {{
    title: 'Frame Render Time Over Benchmark Run',
    xaxis: {{ title: 'Frame', gridcolor: '#334155' }},
    yaxis: {{ title: 'Time (ms)', gridcolor: '#334155' }},
    paper_bgcolor: '#1e293b',
    plot_bgcolor: '#0f172a',
    font: {{ color: '#e2e8f0' }},
    legend: {{ orientation: 'h', y: -0.2 }}
}};
Plotly.newPlot('chart', traces, layout);
<{""}/script>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Exported: {path} ({os.path.getsize(path) / 1024:.1f} KB)")
