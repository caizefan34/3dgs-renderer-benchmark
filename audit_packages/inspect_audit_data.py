#!/usr/bin/env python3
"""Inspect R3 results to understand violation counts and file layout."""
import json
import os
from pathlib import Path

SRC = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/r3")

print("=== Top-level files in r3 ===")
for p in sorted(SRC.iterdir()):
    if p.is_file():
        print(f"  {p.name}  {p.stat().st_size}")

print("\n=== Window directories ===")
for w in sorted([p for p in SRC.iterdir() if p.is_dir()]):
    print(f"  {w.name}/")
    for f in sorted(w.iterdir()):
        print(f"    {f.name}  {f.stat().st_size}")

print("\n=== Aggregated summary keys ===")
agg_path = SRC / "aggregated_summary.json"  # note: actual filename may differ
for cand in list(SRC.glob("*ggregat*")):
    print(f"-- {cand.name}:")
    try:
        data = json.loads(cand.read_text())
    except Exception as exc:
        print(f"   parse error: {exc}")
        continue
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                print(f"   {k}: <{type(v).__name__}, keys={list(v.keys())[:8] if isinstance(v, dict) else f'{len(v)} items'}>")
            else:
                print(f"   {k}: {v}")
    else:
        print(f"   <{type(data).__name__}>")

print("\n=== Per-window certificate JSON structure ===")
for w in ["5000", "15000", "29970"]:
    wdir = SRC / w
    if not wdir.is_dir():
        continue
    for f in sorted(wdir.glob("certificate_*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception as exc:
            print(f"  {w}/{f.name}: parse error {exc}")
            continue
        print(f"  {w}/{f.name}:")
        if isinstance(data, dict):
            for k, v in list(data.items())[:5]:
                if isinstance(v, dict):
                    print(f"    {k}: <dict, keys={list(v.keys())[:6]}>")
                elif isinstance(v, list):
                    print(f"    {k}: <list of {len(v)}>")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"    <{type(data).__name__}>")
