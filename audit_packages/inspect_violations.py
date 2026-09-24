#!/usr/bin/env python3
"""Extract per-window violation summaries from R3 certificate files."""
import json
from pathlib import Path

SRC = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/r3")
WINDOWS = ["5000", "15000", "29970"]

def summarize_entries(data, label):
    """Recursively find dicts with violation-ish keys and sum them."""
    counts = {}
    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("violation_count", "violations") and isinstance(v, (int, float)):
                    counts.setdefault(k, 0)
                    counts[k] += int(v)
                else:
                    walk(v, path + "/" + str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, path + f"[{i}]")
    walk(data)
    print(f"  [{label}] keys: {sorted(counts.items()) if counts else 'no violation fields'}")

for w in WINDOWS:
    wdir = SRC / w
    print(f"=== {w} ===")
    for f in sorted(wdir.glob("certificate_*.json")):
        print(f"  -- {f.name}")
        try:
            data = json.loads(f.read_text())
        except Exception as exc:
            print(f"     parse error: {exc}")
            continue
        if isinstance(data, dict):
            for week, v in list(data.items())[:3]:
                if isinstance(v, dict):
                    inner = next(iter(v.items())) if v else ("?", "?")
                    print(f"     top key {week}: <{type(v).__name__} keys={list(v.keys())[:8]}>")
            # recurse deeper to find violation_count fields
            def find_violations(obj, depth=0, path=""):
                if depth > 8:
                    return
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ("violation_count", "violations") and isinstance(v, (int, float)):
                            print(f"     {path}/{k} = {v}")
                        else:
                            find_violations(v, depth + 1, path + "/" + str(k))
                elif isinstance(obj, list):
                    for i, v in enumerate(obj[:3]):
                        find_violations(v, depth + 1, path + f"[{i}]")
            find_violations(data)
        else:
            print(f"     <{type(data).__name__}>")
    print()
