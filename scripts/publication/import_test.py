import subprocess, sys, glob

r0 = "/mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub/submodules"
for so in glob.glob(f"{r0}/*/**/*.so", recursive=True):
    print("so:", so)

code_b0 = (
    "import sys; sys.path.insert(0, '" + r0 + "/diff-gaussian-rasterization'); "
    "from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer; "
    "print('B0 rasterizer importable OK')"
)
code_knn = (
    "import sys; sys.path.insert(0, '" + r0 + "/simple-knn'); "
    "from simple_knn._C import distCUDA2; print('simple_knn distCUDA2 OK')"
)
r = subprocess.run(["/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python", "-c", code_b0],
                   capture_output=True, text=True, timeout=120)
print("B0:", r.stdout.strip() or r.stderr.strip()[-300:])
r = subprocess.run(["/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python", "-c", code_knn],
                   capture_output=True, text=True, timeout=120)
print("KNN:", r.stdout.strip() or r.stderr.strip()[-300:])

# B1 import test through the trainer bootstrap
code_b1 = """
import sys
sys.path.insert(0, '/home/liaoyuanjun/3dgs-renderer-benchmark')
import publication_trainer as PT
ident = PT.bootstrap_b1()
from gsplat import rasterization
import torch
print('B1 bootstrap OK:', ident['so_sha256'][:16])
print('torch cuda available:', torch.cuda.is_available())
"""
r = subprocess.run(["/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python", "-c", code_b1],
                   capture_output=True, text=True, timeout=300)
print("B1:", (r.stdout.strip() or r.stderr.strip()[-400:]))
