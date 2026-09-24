import gsplat, os
d = os.path.dirname(gsplat.__file__)
p = os.path.join(d, "cuda", "csrc", "IntersectTile.cu")
print("gsplat_dir:", d)
print("IntersectTile_exists:", os.path.exists(p))
if os.path.exists(p):
    with open(p) as f:
        c = f.read()
    print("C1_applied:", "depth_upper" in c or ">> 16" in c)
    print("end_bit_baseline:", "32 + tile_n_bits" in c)
    print("end_bit_c1:", "16 + tile_n_bits" in c)
else:
    # Check compiled extension
    so = os.path.join(d, "_C.cpython-310-x86_64-linux-gnu.so")
    print("compiled_so_exists:", os.path.exists(so))
