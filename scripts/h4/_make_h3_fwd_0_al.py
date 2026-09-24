"""Create /tmp/h3_fwd_0_al.py with a proper f2_b2 that bootstraps gsplat first."""
import pathlib

src = pathlib.Path("/tmp/h3_fwd_1b_0_break_even_oracle.py")
dst = pathlib.Path("/tmp/h3_fwd_0_al.py")

code = src.read_text(encoding="utf-8")

# Remove the old alias if present
code = code.replace("\n\n# --- H4-0 interface (frozen alias) ---\nf2_b2 = make_fixture\n", "")
code = code.replace("\n\nf2_b2 = make_fixture\n", "")

# Add a proper wrapper that calls bootstrap() before make_fixture
code += '''

# --- H4-0 interface (frozen, with gsplat bootstrap) ---
_BOOTSTRAPPED = False

def f2_b2(scene, max_long_side, device):
    """Frozen H3 fixture loader with gsplat bootstrap."""
    global _BOOTSTRAPPED
    if not _BOOTSTRAPPED:
        import os
        os.environ.setdefault("CUDA_HOME", "/mnt/storage_pool/liaoyuanjun/higs-13scene-env")
        bootstrap(
            "/tmp/higs_h3_fwd_1a_source",
            "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so",
        )
        _BOOTSTRAPPED = True
    return make_fixture(scene, max_long_side, device)
'''

dst.write_text(code, encoding="utf-8")
print("wrote", dst, dst.stat().st_size, "bytes")
