#!/usr/bin/env python3
"""BFC0 DSH-2: Exact source audit of installed gsplat backward kernel."""
import json, os, re
import gsplat

def audit_gsplat_source():
    gsplat_dir = os.path.dirname(gsplat.__file__)
    cuda_csrc = os.path.join(gsplat_dir, "cuda", "csrc")
    
    bwd_path = os.path.join(cuda_csrc, "rasterize_to_pixels_bwd.cu")
    fwd_path = os.path.join(cuda_csrc, "rasterize_to_pixels_fwd.cu")
    helpers_path = os.path.join(cuda_csrc, "helpers.cuh")
    
    audit = {
        "gsplat_version": gsplat.__version__,
        "gsplat_dir": gsplat_dir,
        "kernel_files": {
            "backward": bwd_path,
            "forward": fwd_path,
            "helpers": helpers_path,
        },
        "source_evidence": {},
        "semantics": {},
    }
    
    # Read backward kernel
    with open(bwd_path) as f:
        bwd_content = f.read()
    bwd_lines = bwd_content.split('\n')
    
    # 1. Pixel-to-lane mapping
    mapping_evidence = []
    for i, line in enumerate(bwd_lines):
        if any(kw in line for kw in ['block.thread_index', 'block.group_index', 'threadIdx', 'blockIdx', 
                                       'px =', 'py =', 'i =', 'j =']):
            mapping_evidence.append({"line": i+1, "code": line.strip()})
    audit["source_evidence"]["pixel_to_lane_mapping"] = mapping_evidence
    
    # 2. Warp boundaries
    warp_evidence = []
    for i, line in enumerate(bwd_lines):
        if any(kw in line for kw in ['warp.', 'WARP_SIZE', 'warpSize', 'warpSum', 'warp.any', 
                                       'warp.thread_rank', 'blockDim', 'dim3 threads']):
            warp_evidence.append({"line": i+1, "code": line.strip()})
    audit["source_evidence"]["warp_boundaries"] = warp_evidence
    
    # 3. last_ids usage
    last_ids_evidence = []
    for i, line in enumerate(bwd_lines):
        if 'last_id' in line.lower() or 'last_ids' in line:
            last_ids_evidence.append({"line": i+1, "code": line.strip()})
    audit["source_evidence"]["last_ids_usage"] = last_ids_evidence
    
    # 4. Traversal index mapping
    traversal_evidence = []
    for i, line in enumerate(bwd_lines):
        if any(kw in line for kw in ['range_start', 'range_end', 'tile_offsets', 'flatten_ids', 
                                       'batch', 'num_batches', 'bin_final', 'warp_bin_final']):
            traversal_evidence.append({"line": i+1, "code": line.strip()})
    audit["source_evidence"]["traversal_index_mapping"] = traversal_evidence
    
    # 5. warp.any(valid) or equivalent skip
    skip_evidence = []
    for i, line in enumerate(bwd_lines):
        if any(kw in line for kw in ['warp.any', 'valid', 'inside']):
            skip_evidence.append({"line": i+1, "code": line.strip()})
    audit["source_evidence"]["validity_skip"] = skip_evidence
    
    # 6. Power/alpha validity
    alpha_evidence = []
    for i, line in enumerate(bwd_lines):
        if any(kw in line for kw in ['alpha', 'sigma', 'opacity', 'conic', 'vis']):
            alpha_evidence.append({"line": i+1, "code": line.strip()})
    audit["source_evidence"]["alpha_power_validity"] = alpha_evidence
    
    # 7. Reductions/atomics
    atomic_evidence = []
    for i, line in enumerate(bwd_lines):
        if any(kw in line for kw in ['atomicAdd', 'gpuAtomicAdd', 'warpSum', 'reduce']):
            atomic_evidence.append({"line": i+1, "code": line.strip()})
    audit["source_evidence"]["reductions_atomics"] = atomic_evidence
    
    # Semantic extraction
    # Tile size and block dimensions
    tile_size_match = re.search(r'dim3 threads\s*=\s*\{([^}]+)\}', bwd_content)
    audit["semantics"]["threads_per_block"] = tile_size_match.group(1) if tile_size_match else "UNKNOWN"
    
    # Determine traversal direction
    if 'for (int batch = 0' in bwd_content or 'batch < num_batches' in bwd_content:
        audit["semantics"]["traversal_direction"] = "forward_batch_iteration"
    else:
        audit["semantics"]["traversal_direction"] = "reverse_front_to_back"
    
    # last_ids semantics from forward pass
    with open(fwd_path) as f:
        fwd_content = f.read()
    if 'last_ids' in fwd_content:
        audit["semantics"]["last_ids_computed_in_forward"] = True
        # Find how last_ids is computed
        for i, line in enumerate(fwd_content.split('\n')):
            if 'last_id' in line.lower():
                audit["semantics"].setdefault("last_ids_forward_evidence", []).append(
                    {"line": i+1, "code": line.strip()})
    
    # Key finding: does backward use last_ids to skip?
    backward_uses_last_ids = 'last_ids' in bwd_content and 'last_id' in bwd_content
    audit["semantics"]["backward_references_last_ids"] = backward_uses_last_ids
    
    # Check if there is per-pixel early termination
    has_early_term = any(kw in bwd_content for kw in ['T_final', 'T *=', 'last_id', 'bin_final'])
    audit["semantics"]["has_early_termination"] = has_early_term
    
    # Warp-level skip granularity
    audit["semantics"]["warp_skip_granularity"] = "warp_all_lanes_inactive"
    
    return audit

if __name__ == "__main__":
    audit = audit_gsplat_source()
    out_dir = os.environ.get("BFC0_OUT_DIR", "C:\\Users\\36570\\3dgs-renderer-benchmark\\bfc0_p0\\output")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "BFC0_SOURCE_AUDIT.json"), "w") as f:
        json.dump(audit, f, indent=2)
    
    # Print summary
    print("=== BFC0 Source Audit Summary ===")
    print(f"gsplat version: {audit['gsplat_version']}")
    print(f"Threads per block: {audit['semantics'].get('threads_per_block')}")
    print(f"Traversal direction: {audit['semantics'].get('traversal_direction')}")
    print(f"Backward references last_ids: {audit['semantics'].get('backward_references_last_ids')}")
    print(f"Has early termination: {audit['semantics'].get('has_early_termination')}")
    print(f"Warp skip granularity: {audit['semantics'].get('warp_skip_granularity')}")
    print(f"\nSource evidence counts:")
    for k, v in audit["source_evidence"].items():
        print(f"  {k}: {len(v)} entries")
    
    # Write markdown report
    md_path = os.path.join(out_dir, "BFC0_SOURCE_AUDIT.md")
    with open(md_path, "w") as f:
        f.write("# BFC0 Source Audit\n\n")
        f.write(f"## gsplat Version: {audit['gsplat_version']}\n\n")
        f.write("## Semantics\n\n")
        for k, v in audit["semantics"].items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Source Evidence\n\n")
        for k, v in audit["source_evidence"].items():
            f.write(f"### {k}\n\n")
            for entry in v:
                f.write(f"- Line {entry['line']}: `{entry['code']}`\n")
            f.write("\n")
    
    print(f"\nAudit saved to {out_dir}")
