"""
C19-2 Rasterizer Isolation & Resource-Pressure Gate - Report Generator.

Synthesizes results from:
- results/phase-c19/c19-2_reproducibility.json  (Goal 0)
- results/phase-c19/ncu_tile16_*.csv / ncu_tile20_*.csv  (Goal 1)
- results/phase-c19/c19-2_rasterizer_replay.json  (Goal 2)
- results/phase-c19/c19-2_block_geometry_control.json  (Goal 3)
- results/phase-c19/ncu_stalls_tile16_*.csv / _tile20_*.csv  (Goal 4)

Produces:
- reports/phase-c19/c19-2_rasterizer_isolation.md
- results/phase-c19/c19-2_rasterizer_isolation.json
"""
import json, os, sys, math, glob, csv
from datetime import datetime

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))

results_dir = os.path.join(repo, "results", "phase-c19")
reports_dir = os.path.join(repo, "reports", "phase-c19")
os.makedirs(reports_dir, exist_ok=True)

def load_json(name):
    path = os.path.join(results_dir, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def parse_ncu_csv(pattern):
    """Parse ncu CSV output file."""
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        return None
    # Take the first match
    path = files[0]
    data = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                data[row[0].strip()] = row[1].strip()
    return data

# === Load data ===
repro = load_json("c19-2_reproducibility.json")
replay = load_json("c19-2_rasterizer_replay.json")
block_geo = load_json("c19-2_block_geometry_control.json")
ncu_t16 = parse_ncu_csv("ncu_tile16_*.csv")
ncu_t20 = parse_ncu_csv("ncu_tile20_*.csv")
ncu_stall_t16 = parse_ncu_csv("ncu_stalls_tile16_*.csv")
ncu_stall_t20 = parse_ncu_csv("ncu_stalls_tile20_*.csv")

# === Build structured results JSON ===
structured = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "experiment": "C19-2 Rasterizer Isolation & Resource-Pressure Gate",
    "gpu": "NVIDIA A100-PCIE-40GB",
    "driver": "595.71.05",
    "scene": "room",
    "resolution": "1920x1080",
    "reproducibility": repro,
    "ncu_resource_data": {
        "tile_16": ncu_t16,
        "tile_20": ncu_t20,
    },
    "ncu_stall_data": {
        "tile_16": ncu_stall_t16,
        "tile_20": ncu_stall_t20,
    },
    "rasterizer_replay": replay,
    "block_geometry_control": block_geo,
    "mechanism_classification": {},
    "h1_decision": "",
}

# === Mechanism classification ===
# Placeholder - populated by manual/script analysis
classification = {
    "A_register_pressure_spill": {
        "label": "UNKNOWN",
        "evidence": ""
    },
    "B_occupancy_collapse": {
        "label": "UNKNOWN",
        "evidence": ""
    },
    "C_local_memory_spill_traffic": {
        "label": "UNKNOWN",
        "evidence": ""
    },
    "D_shared_memory_pressure": {
        "label": "UNKNOWN",
        "evidence": ""
    },
    "E_warp_stall": {
        "label": "UNKNOWN",
        "evidence": ""
    },
    "F_instruction_throughput": {
        "label": "UNKNOWN",
        "evidence": ""
    },
    "G_memory_cache_limit": {
        "label": "UNKNOWN",
        "evidence": ""
    },
    "H_batch_loop_overhead": {
        "label": "UNKNOWN",
        "evidence": ""
    },
}

structured["mechanism_classification"] = classification

# Don't save structured results until classifications are done
# We'll update them below after analysis

# === Build Markdown Report ===

# --- Helper: build reproducibility table ---
def make_repro_table(rd):
    if not rd or "tile_scaling" not in rd:
        return "_No data_"
    rows = rd["tile_scaling"]
    lines = []
    header = "| Tile | Grid | Tiles | Active | Visible Gs | Ints | Ints/G | Mean/Tile | P50 | P90 | P99 | Max | isect(ms) | rast(ms) | fwd(ms) | ns/int |"
    sep   = "|:---:|:----:|:----:|:-----:|:--------:|:---:|:------:|:----------:|:---:|:---:|:----:|:---:|:--------:|:--------:|:--------:|:------:|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        if isinstance(r.get("mean_ints_per_tile"), (int, float)):
            lines.append(
                f"| {r['tile_size']} | {r['grid']} | {r['total_tiles']} | {r.get('active_tiles','?')} "
                f"| {r.get('visible_gaussians','?'):,} | {r['total_intersections']:,} | "
                f"{r.get('intersection_duplication', '?'):.1f} | {r['mean_ints_per_tile']:.1f} "
                f"| {r['p50']} | {r['p90']} | {r['p99']} | {r['max_ints']} "
                f"| {r.get('isect_tiles_ms', '?'):.4f} | {r.get('rasterization_ms', '?'):.4f} "
                f"| {r.get('full_forward_ms', '?'):.4f} | {r.get('rasterizer_ns_per_intersection', '?'):.4f} |"
            )
    return "\n".join(lines)

def make_replay_table(rd):
    if not rd or "replay_data" not in rd:
        return "_No data_"
    rows = rd["replay_data"]
    lines = []
    header = "| Target ints/tile | Actual ints | Active tiles | Mean ints/tile | Max ints/tile | Block | Rasterize (ms) | CV | ns/int | Throughput (ints/s) |"
    sep   = "|:--------------:|:---------:|:----------:|:------------:|:------------:|:----:|:------------:|:-:|:-----:|:-----------------:|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        label = r.get("target_ints_per_tile", "?")
        lines.append(
            f"| {label} | {r.get('actual_total_ints','?'):,} | {r.get('active_tiles','?')} "
            f"| {r.get('mean_ints_per_tile','?'):.1f} | {r.get('max_ints_per_tile','?')} "
            f"| {r.get('block_dim') or r.get('tile_size','?')}x{r.get('tile_size','?')}x1 "
            f"| {r.get('rasterize_ms_mean','?'):.4f} "
            f"| {r.get('rasterize_ms_cv','?'):.4f} "
            f"| {r.get('ns_per_intersection','?'):.4f} "
            f"| {r.get('throughput_ints_per_sec','?'):,.0f} |"
        )
    return "\n".join(lines)

report = f"""# C19-2 — Rasterizer Isolation & Resource-Pressure Gate

**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}
**GPU:** NVIDIA A100-PCIE-40GB (GPU 0–7)
**Scene:** `room` (official Mip-NeRF 360 pretrained checkpoint)
**Resolution:** 1920×1080
**gsplat:** 1.5.3
**PyTorch:** 2.7.1+cu118
**CUDA:** 11.8 / SM80

---

## 0. Experiment Summary

This gate tests **H1 (register-pressure hypothesis)** by:

1. Reproducing the canonical tile-size scaling baseline (Goal 0)
2. Obtaining **direct compiler resource data** (registers, spills, shared memory) via ncu and ptxas (Goal 1)
3. Constructing a **controlled rasterizer replay benchmark** that decouples tile geometry from intersection count (Goal 2)
4. Running a **block-geometry control** to separate intersection-count effects from block-dimension effects (Goal 3)
5. Classifying the mechanism using all available evidence (Goal 4)

The overarching question: **Is the ~200-ints/tile efficiency cliff caused by register spilling?**

---

## 1. Reproducibility (Goal 0)

### 1.1 Canonical Tile-Size Scaling (Re-run)

{make_repro_table(repro)}

### 1.2 Comparison to C19-1

| Metric | C19-1 (Novel) | C19-2 (Repro) | Delta |
|--------|:------------:|:------------:|:-----:|

> Mark: Fill after C19-1 vs C19-2 comparison

---

## 2. Compiler / Resource Data (Goal 1)

### 2.1 ncu Resource Profile (tile=16)

| Metric | Value |
|--------|-------|
"""

# Add ncu data if available
if ncu_t16:
    for k, v in sorted(ncu_t16.items())[:30]:
        report += f"| `{k}` | {v} |\n"
else:
    report += "| _ncu data not available_ | _check logs_ |\n"

report += """
### 2.2 ncu Resource Profile (tile=20)

| Metric | Value |
|--------|-------|
"""

if ncu_t20:
    for k, v in sorted(ncu_t20.items())[:30]:
        report += f"| `{k}` | {v} |\n"
else:
    report += "| _ncu data not available_ | _check logs_ |\n"

report += """
### 2.3 Resource Comparison: tile=16 vs tile=20

| Metric | tile=16 | tile=20 | Delta |
|--------|:------:|:------:|:-----:|
"""

if ncu_t16 and ncu_t20:
    key_metrics = [
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_dynamic",
        "launch__shared_mem_per_block_static",
        "l1tex__local_load_hit_rate",
        "l1tex__local_store_hit_rate",
        "l1tex__local_load_transactions",
        "l1tex__local_store_transactions",
        "dram__local_load_transactions",
        "dram__local_store_transactions",
        "sm__warps_active",
        "sm__maximum_warps_per_active_cycle",
    ]
    for km in key_metrics:
        v16 = ncu_t16.get(km, "?")
        v20 = ncu_t20.get(km, "?")
        delta = "—"
        try:
            d = float(v20) - float(v16)
            delta = f"{d:+.2e}"  if abs(d) < 0.001 else f"{d:+.4f}"
            if d > 0: delta = f"**+{abs(d):.4f}**"
            elif d < 0: delta = f"{d:.4f}"
            else: delta = "0"
        except: pass
        report += f"| `{km}` | {v16} | {v20} | {delta} |\n"

report += f"""

---

## 3. Rasterizer Replay / Isolation (Goal 2)

### 3.1 Fixed-Geometry Intersection Sweep

Block geometry fixed at **{replay.get('base_tile_size','16')}×{replay.get('base_tile_size','16')}×1** (tile_size={replay.get('base_tile_size','16')}).
Intersection counts varied by truncating/replicating flatten_ids within each tile.

{make_replay_table(replay)}

### 3.2 Key Finding

"""

if replay and "replay_data" in replay:
    canonical = None
    replay_points = []
    for r in replay["replay_data"]:
        if r.get("target_ints_per_tile") == "canonical":
            canonical = r
        else:
            replay_points.append(r)

    if replay_points:
        # Find largest jump in ns/int
        replay_points.sort(key=lambda x: x.get("target_ints_per_tile", 0))
        report += "| Target | Actual Ints | ns/int | Δ from prev |\n"
        report += "|:-----:|:---------:|:-----:|:----------:|\n"
        prev_ns = None
        for r in replay_points:
            ns = r.get("ns_per_intersection", 0)
            delta = "—" if prev_ns is None else f"{ns/prev_ns:.2f}×" if prev_ns > 0 else "—"
            report += f"| {r['target_ints_per_tile']} | {r.get('actual_total_ints','?'):,} | {ns:.4f} | {delta} |\n"
            prev_ns = ns
        if canonical:
            cn = canonical.get("ns_per_intersection", 0)
            report += f"| canonical (~{canonical.get('mean_ints_per_tile','?')}) | {canonical.get('actual_total_ints','?'):,} | {cn:.4f} | — |\n"

report += f"""

---

## 4. Block-Geometry Control (Goal 3)

### 4.1 Canonical Parameters

| Tile | Block | Threads/block | Shmem (B) | Mean ints/tile | Rasterize (ms) | ns/int |
|:---:|:----:|:------------:|:---------:|:------------:|:------------:|:-----:|
"""

if block_geo and "canonical_geometry_data" in block_geo:
    for r in block_geo["canonical_geometry_data"]:
        report += (
            f"| {r.get('tile_size','?')} | {r.get('block_shape','?')} "
            f"| {r.get('threads_per_block','?')} | {r.get('shmem_bytes','?')} "
            f"| {r.get('mean_ints_per_tile','?'):.1f} | {r.get('rasterize_ms','?'):.4f} "
            f"| {r.get('ns_per_intersection','?'):.4f} |\n"
        )

report += """
### 4.2 Intersection Count vs Block Geometry

**Hypothesis: Does the degradation track intersections/tile or block geometry?**

"""

# Present the control data as a table sorted by ints/tile
if block_geo and "canonical_geometry_data" in block_geo:
    sorted_data = sorted(block_geo["canonical_geometry_data"],
                        key=lambda x: x.get("mean_ints_per_tile", 0))
    report += "| Tile | Ints/tile | Threads | Shmem | ns/int |\n"
    report += "|:---:|:---------:|:-------:|:-----:|:-----:|\n"
    for r in sorted_data:
        report += f"| {r['tile_size']} | {r['mean_ints_per_tile']:.1f} | {r['threads_per_block']} | {r['shmem_bytes']} | {r['ns_per_intersection']:.4f} |\n"

report += """
---

## 5. Mechanism Classification (Goal 4)

### Classification Key

| Label | Meaning |
|-------|---------|
| **CONFIRMED** | Direct measurement proves this mechanism |
| **SUPPORTED** | Multiple evidence lines point here, no contradiction |
| **PLAUSIBLE** | Consistent with evidence, but not uniquely determined |
| **INCONCLUSIVE** | Cannot determine with available data |
| **REJECTED** | Evidence contradicts this mechanism |

### A. Register Pressure / Register Spill

"""
# We'll fill in after ncu results
report += """**Evidence:**
- ncu register count: _See Section 2_
- Local memory transactions: _See Section 2_
- Register spill load/store: _See Section 2_

**Classification: **

### B. Occupancy Collapse

**Evidence:**
- Warps active per SM: _See Section 2_
- Theoretical occupancy: _See Section 2_

**Classification: **

### C. Local-Memory Spill Traffic

**Evidence:**
- L1 local load/store transactions: _See Section 2_
- DRAM local transactions: _See Section 2_

**Classification: **

### D. Shared-Memory Pressure

**Evidence:**
- Dynamic shared memory per block: _See Section 2_
- Shared memory capacity (A100): 48 KB configurable, up to 164 KB

**Classification: **

### E. Warp Stall / Dependency

**Evidence:**
- ncu warp stall metrics: _See Section 2_

**Classification: **

### F. Instruction Throughput

**Evidence:**

**Classification: **

### G. Memory / Cache Limitation

**Evidence:**

**Classification: **

### H. Batch-Loop Overhead

**Evidence:**
- The replay experiment with fixed block geometry varies only the per-tile batch count

**Classification: **

---

## 6. H1 Decision

### GO / NO-GO for Register-Aware Tile Subdivision

| Criterion | Met? | Evidence |
|-----------|:---:|----------|

"""

report += """---

## 7. Next Recommended Experiment

Based on the mechanism classification, the recommended next step is:

**

---

## 8. Output Files

- **Report:** `reports/phase-c19/c19-2_rasterizer_isolation.md` (this file)
- **Data:** `results/phase-c19/c19-2_rasterizer_isolation.json`
- **Reproducibility raw:** `results/phase-c19/c19-2_reproducibility.json`
- **Replay raw:** `results/phase-c19/c19-2_rasterizer_replay.json`
- **Block geometry raw:** `results/phase-c19/c19-2_block_geometry_control.json`
- **ncu profiles:** `results/phase-c19/ncu_tile16_*.csv`, `ncu_tile20_*.csv`
- **ncu stall profiles:** `results/phase-c19/ncu_stalls_tile16_*.csv`, `ncu_stalls_tile20_*.csv`
"""

# Write report
md_path = os.path.join(reports_dir, "c19-2_rasterizer_isolation.md")
with open(md_path, "w") as f:
    f.write(report)

# Write structured results
structured["report_md"] = report
res_path = os.path.join(results_dir, "c19-2_rasterizer_isolation.json")
with open(res_path, "w") as f:
    json.dump(structured, f, indent=2, default=str)

print(f"Report: {md_path}")
print(f"Data: {res_path}")
print("DONE")
