"""Dependency-free HTML visualization for a computed Pareto frontier."""
import html
import json


def export_pareto_html(records, pareto_result, path, title="3DGS Pareto Frontier"):
    frontier = set(pareto_result["frontier"])
    plotted = [
        record for record in records
        if record.get("fps") is not None and record.get("psnr") is not None
    ]
    payload = [{
        "renderer": record["renderer"],
        "fps": record["fps"],
        "psnr": record["psnr"],
        "ssim": record.get("ssim"),
        "lpips": record.get("lpips"),
        "pareto": record["renderer"] in frontier,
    } for record in plotted]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><{''}/script>
<style>body{{font-family:system-ui;max-width:1000px;margin:auto;padding:24px;background:#0f172a;color:#e2e8f0}}#plot{{height:600px}}.note{{color:#94a3b8}}</style>
</head><body><h1>{html.escape(title)}</h1>
<p class="note">Experimental speed-quality analysis. Only records with FPS and all GT metrics are eligible. Synthetic stress results are excluded from quality claims.</p>
<div id="plot"></div><script>
const rows = {json.dumps(payload, ensure_ascii=False)};
const groups = [false, true].map(isPareto => {{
  const points = rows.filter(row => row.pareto === isPareto);
  return {{x: points.map(row => row.fps), y: points.map(row => row.psnr),
    text: points.map(row => `${{row.renderer}}<br>SSIM=${{row.ssim}}<br>LPIPS=${{row.lpips}}`),
    hovertemplate: '%{{text}}<br>FPS=%{{x}}<br>PSNR=%{{y}}<extra></extra>',
    mode: 'markers+text', textposition: 'top center',
    name: isPareto ? 'Pareto-optimal' : 'Dominated',
    marker: {{size: 13, color: isPareto ? '#5cfc9c' : '#7c5cfc'}}}};
}});
Plotly.newPlot('plot', groups, {{xaxis:{{title:'FPS (higher is better)'}}, yaxis:{{title:'PSNR vs GT (dB)'}}, paper_bgcolor:'#0f172a',plot_bgcolor:'#1e293b',font:{{color:'#e2e8f0'}}}});
<{''}/script></body></html>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
