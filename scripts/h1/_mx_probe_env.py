"""H1 discovery probe: confirm B1/B2 renderers, torch, gsplat, nsys, checkpoints on mx.
Run inside conda env anysplat. Prints a structured report to stdout. Read-only."""
import os, sys, glob, json, importlib, subprocess, shutil

def section(name):
    print(f"\n=== {name} ===")

section("PY")
print(sys.version)

section("TORCH")
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("ndev", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"dev{i}", p.name, "uuid", torch.cuda.get_device_properties(i))
        # CUDA runtime version via cuda module
        print("driver", torch.cuda.get_device_properties(0))
except Exception as e:
    print("TORCH_ERR", repr(e))

section("GSPLAT")
try:
    import gsplat
    print("gsplat_file", os.path.dirname(gsplat.__file__))
    print("gsplat_ver", getattr(gsplat, "__version__", "NONE"))
    # check the wrapper / rendering entry
    from gsplat import rasterization
    print("rasterization_ok", rasterization is not None)
except Exception as e:
    print("GSPLAT_ERR", repr(e))

section("GSPLAT_CUDA_SRC")
# locate the installed gsplat cuda source (per memory: /home/.../.local/... or site-packages)
try:
    import gsplat
    base = os.path.dirname(gsplat.__file__)
    for sub in ["cuda/csrc", "cuda", "csrc"]:
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            print("dir", d)
            files = sorted(os.listdir(d))
            print("files", files[:40])
except Exception as e:
    print("GSPLAT_SRC_ERR", repr(e))

section("HIGS_RENDERER_AVAIL")
# Try to import the HiGS / trainable renderer. Names from memory: benchmark.run_higs_train_benchmark
# Check for a 'higs' or 'gslam' or 'anysplat' module, and the patches dir.
repo = "/mnt/storage_pool/3dgs-renderer-benchmark/repo"
print("repo_exists", os.path.isdir(repo))
for p in [
    os.path.join(repo, "patches/higs-differentiable.patch"),
    os.path.join(repo, "patches/IntersectTile.c1.cu"),
    os.path.join(repo, "patches/gsplat_orig"),
    os.path.join(repo, "benchmark/run_higs_train_benchmark.py"),
    os.path.join(repo, "benchmark/higs_paper_protocol.py"),
    os.path.join(repo, "variants"),
    os.path.join(repo, "examples/simple_trainer.py"),
]:
    print("exists" if os.path.exists(p) else "MISSING", p)

section("NSYS")
print("nsys_which", shutil.which("nsys"))
for cand in ["/opt/nvidia/nsight-systems/bin/nsys",
             "/usr/local/cuda/bin/nsys",
             "/usr/bin/nsys"]:
    print("cand", cand, os.path.exists(cand))
# search common roots
for root in ["/opt/nvidia", "/opt", "/usr/local"]:
    if os.path.isdir(root):
        for f in glob.glob(root + "/**/nsys", recursive=True)[:5]:
            print("found", f)

section("CUDA_TOOLKIT")
print("nvcc", shutil.which("nvcc"))
print("CUDA_HOME", os.environ.get("CUDA_HOME", "UNSET"))

section("DATASETS")
for d in [
    os.path.join(repo, "datasets"),
    os.path.join(repo, "data"),
    "/mnt/storage_pool/3dgs-renderer-benchmark/datasets",
]:
    if os.path.isdir(d):
        print("dir", d, sorted(os.listdir(d))[:30])

section("CHECKPOINTS_PLY")
# 3DGS trained checkpoints are usually point_cloud/iteration_30000/point_cloud.ply
hits = []
for root in [repo, "/mnt/storage_pool"]:
    for f in glob.glob(root + "/**/point_cloud.ply", recursive=True):
        hits.append(f)
        if len(hits) >= 60:
            break
    if len(hits) >= 60:
        break
print("n_ply", len(hits))
for h in hits[:60]:
    print("ply", h)

section("CHECKPOINTS_PTH")
pth_hits = []
for pat in ["**/chkpnt*.pth", "**/checkpoint*.pth", "**/*.pt"]:
    for root in [repo, "/mnt/storage_pool"]:
        for f in glob.glob(root + "/" + pat, recursive=True):
            pth_hits.append(f)
            if len(pth_hits) >= 60:
                break
        if len(pth_hits) >= 60:
            break
    if len(pth_hits) >= 60:
        break
print("n_pth", len(pth_hits))
for h in pth_hits[:60]:
    print("pth", h)

section("TRAIN_RESULTS")
tr = os.path.join(repo, "results/training")
if os.path.isdir(tr):
    print("dir", tr, sorted(os.listdir(tr))[:40])
else:
    print("no results/training")

print("\n=== PROBE_DONE ===")
