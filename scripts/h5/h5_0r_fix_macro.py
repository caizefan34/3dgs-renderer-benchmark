#!/usr/bin/env python3
"""Fix the H5-0R launch macro: use compile-time FRONTIER_VALUE, not runtime h5_frontier."""
import sys
from pathlib import Path

SRC = Path("/tmp/higs_h3_fwd_1a_source/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu")
code = SRC.read_text()

# Fix 1: Change macro to accept 4th arg FRONTIER_VALUE (compile-time constant)
# Replace h5_frontier> with (FRONTIER_VALUE)> in the macro body
code = code.replace(
    "(SCALAR_VALUE), h5_frontier>",
    "(SCALAR_VALUE), (FRONTIER_VALUE)>"
)
code = code.replace(
    "(SCALAR_VALUE), h5_frontier><<<",
    "(SCALAR_VALUE), (FRONTIER_VALUE)><<<"
)

# Update macro signature to accept 4th arg
code = code.replace(
    "#define HIGS_LAUNCH_BLEND_BWD_PX(CDIM, PX_VALUE, SCALAR_VALUE) \\",
    "#define HIGS_LAUNCH_BLEND_BWD_PX(CDIM, PX_VALUE, SCALAR_VALUE, FRONTIER_VALUE) \\"
)

# Fix 2: Replace call sites with switch on h5_frontier
# For PX=1:
code = code.replace(
    """            if(scalar_adjoint) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, true);                            \\
            else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, false);                                         \\""",
    """            if(scalar_adjoint) { switch(h5_frontier) { case 0: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, true, 0); break; case 1: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, true, 1); break; case 2: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, true, 2); break; default: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, true, 3); break; } } else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, false, 0); \\"""
)

# For PX=2:
code = code.replace(
    """            if(scalar_adjoint) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, true);                            \\
            else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, false);                                         \\""",
    """            if(scalar_adjoint) { switch(h5_frontier) { case 0: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, true, 0); break; case 1: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, true, 1); break; case 2: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, true, 2); break; default: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, true, 3); break; } } else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, false, 0); \\"""
)

# For PX=4:
code = code.replace(
    """            if(scalar_adjoint) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, true);                            \\
            else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, false);                                         \\""",
    """            if(scalar_adjoint) { switch(h5_frontier) { case 0: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, true, 0); break; case 1: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, true, 1); break; case 2: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, true, 2); break; default: HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, true, 3); break; } } else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, false, 0); \\"""
)

SRC.write_text(code)

# Verify
checks = [
    ("FRONTIER_VALUE)", "macro 4th arg"),
    ("switch(h5_frontier)", "runtime switch"),
    ("true, 0)", "FRONTIER=0 literal"),
    ("true, 1)", "FRONTIER=1 literal"),
    ("true, 2)", "FRONTIER=2 literal"),
    ("true, 3)", "FRONTIER=3 literal"),
    ("false, 0)", "baseline literal"),
]
for marker, desc in checks:
    if marker in code:
        print(f"  ✓ {desc}: found '{marker}'")
    else:
        print(f"  ✗ {desc}: MISSING '{marker}'")
        sys.exit(1)
print("Fix applied successfully")
