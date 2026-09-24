import os, glob, importlib.util, torch

candidates = [
    "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so",
    "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_csrc.so",
    "/tmp/h1_b2_authoritative/gsplat_cuda/_backend.so",
]
# extend with any gsplat csrc/so under prknown dirs
for pat in ["/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/**/*.so",
            "/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat/**/*.so",
            "/mnt/storage_pool/3dgs-renderer-benchmark/**/gsplat*/**/*.so"]:
    candidates += glob.glob(pat, recursive=True)
candidates = sorted(set(candidates))
seen=set()
for p in candidates:
    if p in seen: continue
    seen.add(p)
    try:
        name = os.path.basename(p).replace(".so","")
        spec = importlib.util.spec_from_file_location("m"+str(len(seen)), p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        attrs = [a for a in dir(m) if not a.startswith('_')]
        has_enf = 'isect_offset_encode' in attrs
        has_it = hasattr(torch.ops.gsplat,'intersect_tile')
        has_rt = hasattr(torch.ops.gsplat,'rasterize_to_pixels_3dgs')
        core_attrs = [a for a in attrs if 'isect' in a or 'raster' in a or 'Renderer' in a or 'build' in a]
        print(p)
        print("   isect_offset_encode(m)=%s intersect_tile(ops)=%s raster3dgs(ops)=%s"%(has_enf,has_it,has_rt))
        print("   relevant mod attrs:", core_attrs[:20])
    except Exception as e:
        print(p, "ERR", str(e)[:120])