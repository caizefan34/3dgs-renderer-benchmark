#!/usr/bin/env python3
"""
Verify gsplat installation, CUDA binaries, and sparse tile API.
"""
import os
import sys
import json
import hashlib
from pathlib import Path

def sha256_file(path):
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def inspect_gsplat_installation():
    """Inspect gsplat installation and verify sparse tile API."""
    result = {
        "status": "UNKNOWN",
        "sparse_tile_api": False,
        "binaries": []
    }
    
    try:
        import gsplat
        result["status"] = "IMPORTED"
        result["version"] = gsplat.__version__
        result["path"] = gsplat.__file__
        
        # Check for experimental module
        try:
            from gsplat import experimental
            result["has_experimental"] = True
            
            # Check for HiGS functions
            higs_funcs = []
            for name in dir(experimental):
                if 'higs' in name.lower():
                    higs_funcs.append(name)
            
            result["higs_functions"] = higs_funcs
            
            # Check for rasterize_gaussian_higs_dynamic
            if hasattr(experimental, 'rasterize_gaussian_higs_dynamic'):
                result["has_higs_dynamic"] = True
                result["sparse_tile_api"] = True
                
                # Get function signature
                import inspect
                sig = inspect.signature(experimental.rasterize_gaussian_higs_dynamic)
                result["higs_dynamic_signature"] = str(sig)
                
                # Check for tile_mask parameter
                params = list(sig.parameters.keys())
                result["higs_dynamic_params"] = params
                result["has_tile_mask_param"] = 'tile_mask' in params
            else:
                result["has_higs_dynamic"] = False
                
        except ImportError as e:
            result["has_experimental"] = False
            result["experimental_import_error"] = str(e)
        
        # Find compiled binaries
        gsplat_dir = Path(gsplat.__file__).parent
        cuda_dir = gsplat_dir / "cuda"
        
        if cuda_dir.exists():
            # Look for .so files
            for so_file in cuda_dir.rglob("*.so"):
                result["binaries"].append({
                    "path": str(so_file),
                    "size_mb": round(so_file.stat().st_size / (1024*1024), 2)
                })
        
        return result
        
    except Exception as e:
        result["status"] = "IMPORT_FAILED"
        result["error"] = str(e)
        return result

def inspect_cuda_backend():
    """Inspect CUDA backend in benchmark file."""
    benchmark_path = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/run_higs_train_benchmark.py"
    
    if not os.path.exists(benchmark_path):
        return {"status": "FILE_NOT_FOUND"}
    
    result = {
        "status": "FILE_FOUND",
        "path": benchmark_path
    }
    
    # Read and analyze
    with open(benchmark_path) as f:
        content = f.read()
    
    # Check for key API usage
    checks = {
        "make_forward_fn": "make_forward_fn" in content,
        "higs_dynamic_ts": "higs_dynamic_ts" in content,
        "tile_mask": "tile_mask" in content,
        "rasterize_gaussian_higs_dynamic": "rasterize_gaussian_higs_dynamic" in content,
        "error_guided_mask": "_error_guided_mask" in content,
        "importance_l1_loss": "_importance_l1_loss" in content,
        "masked_l1_loss": "_masked_l1_loss" in content,
    }
    
    result["api_checks"] = checks
    result["all_apis_present"] = all(checks.values())
    
    return result

def main():
    print("=== GSPLAT AND CUDA BINARY INSPECTION ===\n")
    
    # Inspect gsplat
    print("1. GSPLAT INSTALLATION")
    print("-" * 60)
    gsplat_info = inspect_gsplat_installation()
    
    print(f"Status: {gsplat_info['status']}")
    if gsplat_info['status'] == 'IMPORTED':
        print(f"Version: {gsplat_info.get('version')}")
        print(f"Path: {gsplat_info.get('path')}")
        print(f"Has experimental: {gsplat_info.get('has_experimental')}")
        print(f"HiGS functions: {gsplat_info.get('higs_functions', [])}")
        print(f"Has higs_dynamic: {gsplat_info.get('has_higs_dynamic')}")
        print(f"Has tile_mask param: {gsplat_info.get('has_tile_mask_param')}")
        print(f"Sparse tile API: {gsplat_info.get('sparse_tile_api')}")
        
        if gsplat_info.get('higs_dynamic_signature'):
            print(f"Signature: {gsplat_info['higs_dynamic_signature']}")
        
        print(f"\nBinaries found: {len(gsplat_info.get('binaries', []))}")
        for b in gsplat_info.get('binaries', [])[:5]:
            print(f"  {b['path']}: {b['size_mb']} MB")
    
    print("\n2. BENCHMARK FILE API CHECK")
    print("-" * 60)
    benchmark_info = inspect_cuda_backend()
    
    print(f"Status: {benchmark_info['status']}")
    if benchmark_info['status'] == 'FILE_FOUND':
        print("API checks:")
        for api, present in benchmark_info.get('api_checks', {}).items():
            mark = "[OK]" if present else "[MISSING]"
            print(f"  {mark} {api}")
        print(f"\nAll APIs present: {benchmark_info.get('all_apis_present')}")
    
    # Save results
    output = {
        "gsplat": gsplat_info,
        "benchmark": benchmark_info
    }
    
    output_path = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/GSPLAT_CUDA_AUDIT.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAudit saved to: {output_path}")
    print("\n=== INSPECTION COMPLETE ===")

if __name__ == "__main__":
    main()
