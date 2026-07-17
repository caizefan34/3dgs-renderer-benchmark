"""Select the pinned renderer backend for each isolated benchmark process."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "artifacts" / "renderer-sources"
ARGS = set(sys.argv[1:])

if ARGS & {"original_3dgs", "diff_gaussian"}:
    sys.path.insert(0, str(SOURCES / "original-diff-gaussian-rasterization"))
elif "tcgs" in ARGS:
    sys.path.insert(
        0,
        str(SOURCES / "3DGSTensorCore" / "submodules" / "tcgs_speedy_rasterizer"),
    )
elif ARGS & {"gsplat", "gsplat_dense", "gsplat_higs"}:
    import benchmark_renderer_preload
elif ARGS & {"speedy_splat", "speedy_splat_raw"}:
    sys.path.insert(0, str(SOURCES / "speedy-splat"))
