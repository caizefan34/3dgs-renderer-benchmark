#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx")

try:
    import gsplat
    print(f"✓ gsplat imported successfully")
    print(f"  Version: {gsplat.__version__}")
    print(f"  Location: {gsplat.__file__}")
    
    # Check for rasterization function
    from gsplat import rasterization
    print(f"✓ rasterization function available")
    
    # Check for CUDA wrappers
    try:
        from gsplat.cuda import _wrapper
        print(f"✓ CUDA wrapper available")
    except ImportError as e:
        print(f"✗ CUDA wrapper not available: {e}")
    
    # Check for experimental HiGS kernels
    try:
        from gsplat.experimental.render.kernels import _backend as higs_backend
        print(f"✓ HiGS experimental kernels available")
    except ImportError as e:
        print(f"✗ HiGS experimental kernels not available: {e}")
    
    print("\n=== Binary Reuse Assessment ===")
    print("SUCCESS: Existing binaries can be reused")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Failed to import gsplat: {e}")
    import traceback
    traceback.print_exc()
    print("\n=== Binary Reuse Assessment ===")
    print("FAILED: Need to recompile or fix installation")
    sys.exit(1)
