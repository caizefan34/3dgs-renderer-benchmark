#!/usr/bin/env python3
"""Extract ptxas resource metrics from build logs.

Parses nvcc/ptxas verbose output for:
- registers/thread
- shared memory/block
- local memory/thread
- spill stores
- spill loads
- stack frame
"""
import sys
import re
import json
from pathlib import Path

def extract_ptxas_for_kernel(log_text: str, kernel_name: str = "rasterize_to_pixels_3dgs_bwd_kernel"):
    """Extract ptxas info for a specific kernel from build log.
    
    ptxas output looks like:
    ptxas info : Compiling entry function '...' for 'sm_80'
    ptxas info : Function properties for '...'
    ptxas info : used X registers, Y bytes cmem, Z bytes smem, ...
    ptxas info : ... spill stores, ... spill loads
    """
    lines = log_text.split('\n')
    
    # Find all ptxas info blocks
    results = []
    current_func = None
    current_info = {}
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Match function name in compilation entry
        if "Compiling entry function" in line_stripped:
            # Save previous function info
            if current_func and current_info:
                results.append({"function": current_func, **current_info})
            current_func = None
            current_info = {}
            # Extract function name
            m = re.search(r"'([^']+)'", line_stripped)
            if m:
                current_func = m.group(1)
            # Extract SM version
            m2 = re.search(r"'(sm_\d+)'", line_stripped)
            if m2:
                current_info["sm_version"] = m2.group(1)
        
        # Match "Function properties for"
        elif "Function properties for" in line_stripped:
            pass  # Already have func name from Compiling entry
        
        # Match register/smem/cmem info
        elif "ptxas info" in line_stripped and "registers" in line_stripped:
            m = re.search(r'(\d+)\s+registers', line_stripped)
            if m:
                current_info["registers"] = int(m.group(1))
            m = re.search(r'(\d+)\s+bytes\s+cmem', line_stripped)
            if m:
                current_info["cmem_bytes"] = int(m.group(1))
            m = re.search(r'(\d+)\s+bytes\s+smem', line_stripped)
            if m:
                current_info["smem_bytes"] = int(m.group(1))
            m = re.search(r'(\d+)\s+bytes\s+lmem', line_stripped)
            if m:
                current_info["lmem_bytes"] = int(m.group(1))
            m = re.search(r'(\d+)\s+bytes\s+stack', line_stripped)  
            if m:
                current_info["stack_frame_bytes"] = int(m.group(1))
        
        # Match spill info
        elif "ptxas info" in line_stripped and "spill" in line_stripped:
            m = re.search(r'(\d+)\s+bytes\s+spill\s+stores', line_stripped)
            if m:
                current_info["spill_stores_bytes"] = int(m.group(1))
            m = re.search(r'(\d+)\s+bytes\s+spill\s+loads', line_stripped)
            if m:
                current_info["spill_loads_bytes"] = int(m.group(1))
            # Sometimes it's just counts
            m = re.search(r'(\d+)\s+spill\s+stores', line_stripped)
            if m and "spill_stores_bytes" not in current_info:
                current_info["spill_stores"] = int(m.group(1))
            m = re.search(r'(\d+)\s+spill\s+loads', line_stripped)
            if m and "spill_loads_bytes" not in current_info:
                current_info["spill_loads"] = int(m.group(1))
    
    # Save last function
    if current_func and current_info:
        results.append({"function": current_func, **current_info})
    
    # Filter for our kernel
    bwd_results = [r for r in results if kernel_name in r.get("function", "")]
    
    # Also get all results for reference
    return bwd_results, results


def compute_theoretical_occupancy(registers: int, smem_bytes: int, sm_version: str = "sm_80"):
    """Compute theoretical occupancy for A100 (SM 8.0).
    
    A100 SM 8.0 specs:
    - Max 2048 threads per SM
    - Max 32 blocks per SM
    - 65536 registers per SM
    - 164KB shared memory per SM (configurable)
    - Max 1024 threads per block
    """
    # Extract SM version number
    sm_num = int(re.search(r'(\d+)', sm_version).group(1)) if sm_version else 80
    
    if sm_num >= 80:  # A100
        max_threads_per_sm = 2048
        max_blocks_per_sm = 32
        max_regs_per_sm = 65536
        # Shared memory: 164KB = 167936 bytes (default config)
        max_smem_per_sm = 167936  # can be up to 99KB without opt-in on A100
        # Actually A100 has 164KB configurable, 100KB default limit
        max_smem_per_block = 99 * 1024  # default without cudaFuncSetAttribute opt-in
    else:
        max_threads_per_sm = 2048
        max_blocks_per_sm = 32
        max_regs_per_sm = 65536
        max_smem_per_sm = 96 * 1024
        max_smem_per_block = 48 * 1024
    
    # Register limit: 255 regs/thread max, allocated in granularity of 256 regs/warp
    # For 256 regs/warp granularity: regs_per_thread rounded up to nearest 256/32 = 8? 
    # Actually, registers are allocated per warp in groups. On SM 8.0:
    # Register allocation granularity is 256 registers per warp.
    # So regs_per_warp = ceil(registers * 32 / 256) * 256 ... no.
    # Actually: each warp gets registers, allocated in chunks.
    # regs_per_warp = ceil(registers / (256/32)) * (256/32) = ceil(registers / 8) * 8
    # Wait, the allocation unit is per-warp. On SM 8.0, the register allocation
    # granularity is 256 registers per warp, i.e., each warp gets a multiple of 256 registers.
    # No — it's: each thread gets `registers` registers, and the allocation is per warp
    # with granularity. The register file is 65536 regs.
    # Allocation granularity on SM 8.0: 256 regs per warp.
    # So effective_regs_per_warp = ceil(registers * 32 / 256) * 256 ... no, that's wrong.
    # 
    # Actually: warp gets ceil(registers/ (alloc_granularity/32)) * (alloc_granularity/32) regs per thread
    # On SM 8.0, alloc_granularity = 256, so per-thread granularity = 256/32 = 8
    # Wait, that's the register allocation unit per thread.
    # Hmm, let me use the simpler approach:
    # regs_per_warp = ceil(registers * 32 / 256) * 256 (rounded up to 256 granularity)
    # Actually no. Let me just use:
    # warps_per_sm_from_regs = max_regs_per_sm // (ceil(registers * 32 / 256) * 256 / 32)
    # This is getting complicated. Let me use the standard formula:
    
    # Register allocation granularity (per warp) = 256 on SM 8.0
    reg_alloc_unit = 256
    regs_per_warp = ((registers * 32) + reg_alloc_unit - 1) // reg_alloc_unit * reg_alloc_unit
    if regs_per_warp == 0:
        regs_per_warp = reg_alloc_unit
    warps_per_sm_regs = max_regs_per_sm // regs_per_warp
    
    # Shared memory limit
    if smem_bytes > 0:
        blocks_per_sm_smem = max_smem_per_block // smem_bytes if smem_bytes <= max_smem_per_block else 0
    else:
        blocks_per_sm_smem = max_blocks_per_sm
    
    # Warp limit: max 2048 threads / 32 = 64 warps per SM
    warps_per_sm_max = max_threads_per_sm // 32
    
    # Occupancy from registers
    occ_regs = min(warps_per_sm_regs, warps_per_sm_max)
    
    # Occupancy from shared memory (need to know threads per block)
    # We don't know threads per block here, so we report warps
    
    return {
        "registers": registers,
        "regs_per_warp": regs_per_warp,
        "warps_per_sm_from_regs": warps_per_sm_regs,
        "max_warps_per_sm": warps_per_sm_max,
        "occupancy_pct_from_regs": round(occ_regs / warps_per_sm_max * 100, 1),
        "smem_bytes": smem_bytes,
        "blocks_per_sm_from_smem": blocks_per_sm_smem,
        "sm_version": sm_version,
    }


def main():
    baseline_log = Path(sys.argv[1]).read_text()
    cpcb_log = Path(sys.argv[2]).read_text()
    output_path = sys.argv[3]
    
    # Extract for backward kernel (the kernel we care about)
    # The kernel is instantiated for multiple CDIM values, focus on CDIM=3
    baseline_bwd, baseline_all = extract_ptxas_for_kernel(baseline_log)
    cpcb_bwd, cpcb_all = extract_ptxas_for_kernel(cpcb_log)
    
    # For CDIM=3, the kernel name contains "3, float" or just the template
    # Actually the kernel name in ptxas would be like:
    # _ZN6gsplat31rasterize_to_pixels_3dgs_bwd_kernelILj3EfEvv
    # We want the CDIM=3 instantiation
    def find_cdim3(results):
        for r in results:
            func = r.get("function", "")
            # CDIM=3 instantiation: look for Lj3 (template param 3)
            if "Lj3E" in func or "3Ef" in func or "_3dgs_bwd" in func:
                if "registers" in r:
                    return r
        # Fallback: return first with registers
        for r in results:
            if "registers" in r:
                return r
        return {}
    
    baseline_kern = find_cdim3(baseline_bwd) if baseline_bwd else {}
    cpcb_kern = find_cdim3(cpcb_bwd) if cpcb_bwd else {}
    
    # Also try to find all CDIM instantiations
    def find_all_cdim(results):
        found = {}
        for r in results:
            func = r.get("function", "")
            for cdim in [3, 8]:
                tag = f"Lj{cdim}E" if cdim < 10 else f"Lj{cdim}E"
                if tag in func and "registers" in r:
                    found[cdim] = r
        return found
    
    baseline_all_cdim = find_all_cdim(baseline_bwd)
    cpcb_all_cdim = find_all_cdim(cpcb_bwd)
    
    # Compute occupancy
    baseline_occ = compute_theoretical_occupancy(
        baseline_kern.get("registers", 0),
        baseline_kern.get("smem_bytes", 0),
        baseline_kern.get("sm_version", "sm_80")
    ) if baseline_kern else {}
    
    cpcb_occ = compute_theoretical_occupancy(
        cpcb_kern.get("registers", 0),
        cpcb_kern.get("smem_bytes", 0),
        cpcb_kern.get("sm_version", "sm_80")
    ) if cpcb_kern else {}
    
    result = {
        "baseline_cdim3": {
            "kernel_info": baseline_kern,
            "occupancy": baseline_occ,
        },
        "cpcb_cdim3": {
            "kernel_info": cpcb_kern,
            "occupancy": cpcb_occ,
        },
        "baseline_all_cdim": {str(k): v for k, v in baseline_all_cdim.items()},
        "cpcb_all_cdim": {str(k): v for k, v in cpcb_all_cdim.items()},
        "all_baseline_functions": [{"function": r.get("function","")[:100], "registers": r.get("registers"), "smem": r.get("smem_bytes")} for r in baseline_all if "registers" in r],
        "all_cpcb_functions": [{"function": r.get("function","")[:100], "registers": r.get("registers"), "smem": r.get("smem_bytes")} for r in cpcb_all if "registers" in r],
        "note": "CDIM=3 is the standard RGB case. A100 is SM 8.0 with 65536 regs, 2048 max threads/SM.",
    }
    
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"Resource metrics saved to {output_path}")
    
    # Print summary
    print("\n=== RESOURCE SUMMARY (CDIM=3) ===")
    for label, kern, occ in [("BASELINE", baseline_kern, baseline_occ), ("CPCB", cpcb_kern, cpcb_occ)]:
        print(f"\n{label}:")
        print(f"  registers:     {kern.get('registers', 'N/A')}")
        print(f"  smem_bytes:    {kern.get('smem_bytes', 'N/A')}")
        print(f"  cmem_bytes:    {kern.get('cmem_bytes', 'N/A')}")
        print(f"  lmem_bytes:    {kern.get('lmem_bytes', 'N/A')}")
        print(f"  spill_stores:  {kern.get('spill_stores', kern.get('spill_stores_bytes', 'N/A'))}")
        print(f"  spill_loads:   {kern.get('spill_loads', kern.get('spill_loads_bytes', 'N/A'))}")
        print(f"  stack_frame:   {kern.get('stack_frame_bytes', 'N/A')}")
        if occ:
            print(f"  occupancy_pct: {occ.get('occupancy_pct_from_regs', 'N/A')}%")
    
    if baseline_kern.get("registers") and cpcb_kern.get("registers"):
        delta = cpcb_kern["registers"] - baseline_kern["registers"]
        print(f"\nRegister delta: {delta} ({'+' if delta >= 0 else ''}{delta})")


if __name__ == "__main__":
    main()
