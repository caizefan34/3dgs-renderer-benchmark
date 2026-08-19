#!/usr/bin/env python3
"""
Phase 5: Extract kernel resource metadata from compiled gsplat binaries.
Uses CUDA Driver API directly via ctypes to query kernel launch attributes.
Works without NCU (WDDM-blocked).
"""
import os, sys, json, ctypes, struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

# --------------------------------------------------------------------------
# CUDA Driver API constants & helpers
# --------------------------------------------------------------------------
CUDA_SUCCESS = 0

class CUfunc_attribute:
    MAX_THREADS_PER_BLOCK       = 0
    SHARED_SIZE_BYTES           = 1
    CONST_SIZE_BYTES            = 2
    LOCAL_SIZE_BYTES            = 3
    NUM_REGS                    = 4
    PTX_VERSION                 = 5
    BINARY_VERSION              = 6
    CACHE_MODE_CA               = 7
    MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    PREFERRED_SHARED_MEMORY_CARVEOUT = 9

def load_cuda():
    if sys.platform == "win32":
        return ctypes.WinDLL("nvcuda.dll")
    else:
        return ctypes.CDLL("libcuda.so.1")

CUDA = load_cuda()

def check(result, msg=""):
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"CUDA error {result}: {msg}")

# --------------------------------------------------------------------------
# Find compiled gsplat extension binary
# --------------------------------------------------------------------------
def find_gsplat_module():
    """Find the compiled gsplat CUDA extension binary path."""
    import importlib
    # First, try to get the .so/.pyd file from the loaded gsplat module
    try:
        import gsplat
        gsplat_file = Path(gsplat.__file__)
        print(f"  gsplat package: {gsplat_file}")
        # The compiled extension is likely a sibling .pyd or in a subdirectory
        ext_dir = gsplat_file.parent
        for p in ext_dir.rglob("*.pyd"):
            if "rasterize" in p.stem.lower() or "gsplat" in p.stem.lower():
                return p
        for p in ext_dir.rglob("*.so"):
            if "rasterize" in p.stem.lower() or "gsplat" in p.stem.lower():
                return p
    except Exception as e:
        print(f"  gsplat import error: {e}")

    # Fallback: torch extensions cache
    for base in [
        Path.home() / "AppData/Local/torch_extensions",
        Path.home() / ".cache/torch_extensions",
    ]:
        if base.exists():
            for p in base.rglob("*.pyd"):
                if "gsplat" in p.stem.lower():
                    return p
            for p in base.rglob("*.so"):
                if "gsplat" in p.stem.lower():
                    return p
    return None

# --------------------------------------------------------------------------
# Query kernel attributes via CUDA Driver API
# --------------------------------------------------------------------------
def query_kernel_attributes_driver_api(module_path: Path):
    """Load a CUDA module and query all kernel functions' resource attributes.

    This works even without NCU hardware counters (WDDM mode).
    """
    # Initialize CUDA driver
    check(CUDA.cuInit(0), "cuInit")

    # Get device
    dev = ctypes.c_int()
    check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

    # Create context
    ctx = ctypes.c_void_p()
    check(CUDA.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    results = {}

    try:
        # Load the module from the cubin file
        # The .pyd is actually a PE DLL, not a cubin. We need the actual cubin.
        # But CUDA modules are embedded as .nvfatbin sections in the DLL.
        # We can try loading the DLL directly - CUDA runtime already has it loaded.
        
        # Alternative: find the cubin in the extension cache directory
        cubin_path = module_path.parent / module_path.stem.replace("_ext", "") / "cuda" / "rasterize.cu.obj"
        if not cubin_path.exists():
            # Search for cubin files
            cubins = list(module_path.parent.rglob("*.cubin")) + list(module_path.parent.rglob("*.fatbin"))
            if not cubins:
                cubins = list(module_path.parent.rglob("*.o")) + list(module_path.parent.rglob("*.obj"))
            if cubins:
                cubin_path = cubins[0]
        
        if cubin_path and cubin_path.exists():
            print(f"  Found cubin: {cubin_path}")
        else:
            print(f"  No cubin found near {module_path.parent}")
            # Try to enumerate modules already loaded in CUDA context
            # cuCtxGetModuleEnumerator might help (CUDA 12+)
            
            # Fallback: try to get function from the default context
            # Most gsplat compiled extensions embed cubin data and register it
            # We can try to find the function by mangled name
            
            # Try using CUPTI's module enumeration
            cuda_ver = torch.version.cuda
            print(f"  CUDA version: {cuda_ver}")
            
            # List known gsplat kernel mangled names
            mangled_names = [
                # rasterize_to_pixels (template with different tile sizes may vary)
                "_ZN6gsplat35rasterize_to_pixels_3dgs_fwd_kernelILj3EfEEvjjjbPKN3glm3vecILi2EfLNS1_9qualifierE0EEEPKNS2_ILi3EfLS3_0EEEPKT0_SC_SC_PKbjjjjjPKiSG_PSA_SH_Pi",
                # spherical harmonics fwd
                "_ZN6gsplat30spherical_harmonics_fwd_kernelIfEEvjjjPKN3glm3vecILi3EfLNS1_9qualifierE0EEEPKT_PKbPS7_",
                # projection ewa
                "_ZN6gsplat37projection_ewa_3dgs_packed_fwd_kernelIfEEvjjjPKT_S3_S3_S3_S3_S3_S3_jjffffPKiNS_15CameraModelTypeEPiS7_PxS8_S8_S7_PS1_S9_S9_S9_",
                # intersect tile
                "_ZN6gsplat21intersect_tile_kernelIfEEvbjjjPKxS2_PKT_PKiS5_S2_jjjjjPiPxS8_",
                # intersect offset
                "_ZN6gsplat23intersect_offset_kernelEjPKxjjjPi",
            ]
            
            # The mangled name may differ between compilations
            # Try to find ANY gsplat function in the default context
            
            # For CUDA, we can use cuModuleGetFunction with NULL module
            # to search all loaded modules (CUDA 11+)
            
            for name in mangled_names:
                func = ctypes.c_void_p()
                name_bytes = name.encode("utf-8")
                name_c = ctypes.c_char_p(name_bytes)
                
                result = CUDA.cuModuleGetFunction(
                    ctypes.byref(func),
                    None,  # NULL = search all loaded modules (CUDA 11.2+)
                    name_c,
                )
                
                if result == CUDA_SUCCESS:
                    attrs = {}
                    for attr_name, attr_id in [
                        ("max_threads_per_block", CUfunc_attribute.MAX_THREADS_PER_BLOCK),
                        ("shared_size_bytes", CUfunc_attribute.SHARED_SIZE_BYTES),
                        ("const_size_bytes", CUfunc_attribute.CONST_SIZE_BYTES),
                        ("local_size_bytes", CUfunc_attribute.LOCAL_SIZE_BYTES),
                        ("num_registers", CUfunc_attribute.NUM_REGS),
                        ("max_dynamic_shared_size", CUfunc_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES),
                        ("preferred_shared_carveout", CUfunc_attribute.PREFERRED_SHARED_MEMORY_CARVEOUT),
                    ]:
                        val = ctypes.c_int()
                        r = CUDA.cuFuncGetAttribute(ctypes.byref(val), attr_id, func)
                        if r == CUDA_SUCCESS:
                            attrs[attr_name] = val.value
                        else:
                            attrs[attr_name] = f"error={r}"
                    
                    kshort = name.split("gsplat")[1][:60] if "gsplat" in name else name[:60]
                    results[kshort] = attrs
                    print(f"  Found kernel: {kshort}")
                    for k, v in attrs.items():
                        print(f"    {k}: {v}")
                else:
                    kshort = name.split("gsplat")[1][:60] if "gsplat" in name else name[:60]
                    print(f"  Not found: {kshort} (error={result})")

    finally:
        CUDA.cuCtxDestroy_v2(ctx)

    return results

# --------------------------------------------------------------------------
# Simplified approach: compile a tiny CUDA module to query attributes  
# --------------------------------------------------------------------------
def query_via_torch_jit_ext():
    """Use torch.utils.cpp_extension to compile a tiny CUDA helper that queries kernel attrs."""
    try:
        import torch.utils.cpp_extension as cpp_ext
        
        cuda_source = """
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <torch/extension.h>
        #include <string>
        #include <unordered_map>
        #include <vector>
        
        std::unordered_map<std::string, std::vector<int>> get_kernel_attrs(
                std::vector<std::string> kernel_names) {
            std::unordered_map<std::string, std::vector<int>> results;
            
            CUresult res;
            
            for (const auto& name : kernel_names) {
                CUfunction func;
                
                // Search all loaded modules (NULL module handle = all modules in CUDA 11.2+)
                res = cuModuleGetFunction(&func, NULL, name.c_str());
                if (res == CUDA_SUCCESS) {
                    int regs, shmem, block_size, dyn_shmem;
                    
                    cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
                    cuFuncGetAttribute(&shmem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
                    cuFuncGetAttribute(&block_size, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func);
                    cuFuncGetAttribute(&dyn_shmem, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func);
                    
                    results[name] = {regs, shmem, block_size, dyn_shmem};
                } else {
                    results[name] = {res, -1, -1, -1};  // Error code
                }
            }
            
            return results;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("get_kernel_attrs", &get_kernel_attrs, "Get CUDA kernel attributes");
        }
        """
        
        module = cpp_ext.load_inline(
            name="_phase5_kernel_attrs",
            cpp_sources=cuda_source,
            functions=["get_kernel_attrs"],
            verbose=False,
        )
        
        mangled_names = [
            "_ZN6gsplat35rasterize_to_pixels_3dgs_fwd_kernelILj3EfEEvjjjbPKN3glm3vecILi2EfLNS1_9qualifierE0EEEPKNS2_ILi3EfLS3_0EEEPKT0_SC_SC_PKbjjjjjPKiSG_PSA_SH_Pi",
            "_ZN6gsplat30spherical_harmonics_fwd_kernelIfEEvjjjPKN3glm3vecILi3EfLNS1_9qualifierE0EEEPKT_PKbPS7_",
            "_ZN6gsplat37projection_ewa_3dgs_packed_fwd_kernelIfEEvjjjPKT_S3_S3_S3_S3_S3_S3_jjffffPKiNS_15CameraModelTypeEPiS7_PxS8_S8_S7_PS1_S9_S9_S9_",
            "_ZN6gsplat21intersect_tile_kernelIfEEvbjjjPKxS2_PKT_PKiS5_S2_jjjjjPiPxS8_",
            "_ZN6gsplat23intersect_offset_kernelEjPKxjjjPi",
        ]
        
        print(f"\n  Querying {len(mangled_names)} kernel names via JIT extension...")
        result = module.get_kernel_attrs(mangled_names)
        
        parsed = {}
        for name, attrs in result.items():
            if len(attrs) >= 1 and attrs[0] >= 0:
                kshort = name.split("gsplat")[1][:60] if "gsplat" in name else name[:60]
                parsed[kshort] = {
                    "registers_per_thread": attrs[0],
                    "shared_mem_bytes": attrs[1],
                    "max_threads_per_block": attrs[2],
                    "max_dynamic_shared_mem_bytes": attrs[3],
                }
                print(f"  {kshort}:")
                print(f"    registers: {attrs[0]}, shared_mem: {attrs[1]} B, "
                      f"block_size: {attrs[2]}, dyn_shared: {attrs[3]} B")
            else:
                kshort = name.split("gsplat")[1][:60] if "gsplat" in name else name[:60]
                print(f"  {kshort}: NOT FOUND (CUDA error {attrs[0]})")
                parsed[kshort] = {"error": f"CUDA error {attrs[0]}"}
        
        return parsed
        
    except Exception as e:
        print(f"  JIT extension query failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*70)
    print("  Phase 5: Kernel Resource Metadata")
    print("  (WDDM mode — using CUDA Driver API)")
    print("="*70)
    
    # GPU properties
    print("\n--- GPU Properties ---")
    props = torch.cuda.get_device_properties(0)
    info = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "max_threads_per_block": props.max_threads_per_block,
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "shared_mem_per_block_kb": props.shared_memory_per_block / 1024,
        "shared_mem_per_sm_kb": props.shared_memory_per_multiprocessor / 1024,
        "regs_per_sm": props.regs_per_multiprocessor,
        "warp_size": props.warp_size,
        "total_vram_gb": props.total_memory / (1024**3),
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # Find gsplat module
    print("\n--- Finding gsplat CUDA Module ---")
    module_path = find_gsplat_module()
    if module_path:
        print(f"  Module: {module_path}")
        print(f"  Size: {module_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"  Not found in common paths")
        # Try to get the location from the loaded module
        try:
            import gsplat.rasterize as gsplat_ext
            print(f"  gsplat.rasterize: {gsplat_ext.__file__}")
            module_path = Path(gsplat_ext.__file__)
        except (ImportError, AttributeError):
            try:
                from gsplat import csrc as gsplat_csrc
                print(f"  gsplat.csrc: {gsplat_csrc.__file__}")
                module_path = Path(gsplat_csrc.__file__)
            except (ImportError, AttributeError):
                pass
    
    # Query kernel attributes via JIT extension
    print("\n--- Kernel Attribute Query ---")
    results = query_via_torch_jit_ext()
    
    # Save results
    if results:
        output = {
            "gpu_properties": info,
            "kernel_attributes": results,
            "note": "Kernel attributes from CUDA Driver API (cuFuncGetAttribute)",
            "ncus_status": "BLOCKED (WDDM, ERR_NVGPUCTRPERM)",
        }
        out_dir = Path("results/epic05/phase5")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "kernel_attributes.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {out_path}")

if __name__ == "__main__":
    main()
