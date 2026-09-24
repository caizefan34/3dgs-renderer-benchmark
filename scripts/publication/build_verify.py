import json, glob, os

print("== build log summary ==")
for line in open("/mnt/storage_pool/liaoyuanjun/pubphase/build_runner.log", errors="replace"):
    if any(k in line for k in ("B1_RC", "B0_RASTER_RC", "B0_KNN_RC", "built:", "WROTE", "error", "Error")):
        print(line.rstrip()[:150])

print("\n== B1 .so ==")
for p in glob.glob("/mnt/storage_pool/liaoyuanjun/gsplat_b1clean_pub_cache/gsplat_cuda_b1clean/*.so"):
    print(p, os.path.getsize(p))

print("\n== B1 identity ==")
d = json.load(open("/mnt/storage_pool/liaoyuanjun/b1_clean_build_identity.json"))
print("sha256:", d["extension_sha256"])
print("base:", d["base_commit"], d["base_tag"], "| torch:", d["torch"], "| sources:", d["n_sources"])

print("\n== B0 rasterizer .so ==")
for p in glob.glob("/mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub/submodules/diff-gaussian-rasterization/*.so"):
    print(p, os.path.getsize(p))
print("\n== B0 simple-knn .so ==")
for p in glob.glob("/mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub/submodules/simple-knn/*.so"):
    print(p, os.path.getsize(p))
