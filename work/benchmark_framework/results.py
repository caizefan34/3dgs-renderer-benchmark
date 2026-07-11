"""
Results manager: collect, aggregate, and export benchmark results.
"""
import json
import os
from typing import Dict, List
from .metrics import RendererMetrics


class ResultsManager:
    def __init__(self):
        self.results: Dict[str, RendererMetrics] = {}
    
    def add_result(self, renderer_name: str, metrics: RendererMetrics):
        self.results[renderer_name] = metrics
    
    def get_summary(self) -> dict:
        summary = {}
        for name, m in self.results.items():
            summary[name] = m.to_dict()
        return summary
    
    def get_ranking(self) -> List[tuple]:
        rankings = [(name, m.mean_fps, m.mean_latency_ms) 
                     for name, m in self.results.items()
                     if m.mean_fps > 0]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def export_json(self, path: str):
        data = self.get_summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Exported: {path}")
    
    def export_markdown(self, path: str):
        rankings = self.get_ranking()
        lines = [
            "# 3DGS Renderer Benchmark Results",
            "",
            f"## Summary",
            "",
            f"| Rank | Renderer | Mean FPS | Mean Latency (ms) | Median Latency (ms) |",
            f"|------|----------|----------|-------------------|---------------------|",
        ]
        for i, (name, fps, lat) in enumerate(rankings, 1):
            m = self.results[name]
            lines.append(f"| {i} | {name} | {fps:.1f} | {lat:.2f} | {m.median_latency_ms:.2f} |")
        
        lines += ["", "## Per-Renderer Details", ""]
        for rname, m in self.results.items():
            d = m.to_dict()
            lines += [
                f"### {rname}",
                f"- **FPS (mean)**: {d['mean_fps']}",
                f"- **FPS (median)**: {1000/d['median_latency_ms']:.1f}" if d['median_latency_ms'] > 0 else "",
                f"- **Latency**: mean={d['mean_latency_ms']}ms, median={d['median_latency_ms']}ms",
                f"- **Min/Max**: {d['min_latency_ms']}ms / {d['max_latency_ms']}ms",
                f"- **Std**: {d['std_latency_ms']}ms",
                f"- **Frames**: {d['num_frames']}",
                "",
            ]
        
        content = "\n".join(lines).strip() + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Exported: {path}")
