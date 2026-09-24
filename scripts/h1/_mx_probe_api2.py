import inspect
import gsplat.experimental as E

names = ["rasterize_gaussian_higs_frozen", "rasterize_gaussian_higs_dynamic",
         "create_himsg_renderer", "RasterizeGaussianHigsSpec",
         "RasterizeGaussianHigsDynamicSpec"]
for n in names:
    obj = getattr(E, n, None)
    if obj is not None:
        try:
            print("=== %s sig ===" % n)
            print(str(inspect.signature(obj))[:2200])
        except Exception:
            print("%s: no signature; type=%r" % (n, type(obj)))
    else:
        print("%s: NOT FOUND" % n)
