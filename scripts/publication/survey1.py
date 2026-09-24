import subprocess, os

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (r.stdout + r.stderr).strip()

print("== ~/3dgs-renderer-benchmark ==")
print(run("ls ~/3dgs-renderer-benchmark/"))
print("\n== ~ top-level ==")
print(run("ls ~/ | head -40"))
print("\n== higs env python: gsplat ==")
print(run("/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python -c \"import gsplat; print(gsplat.__version__, gsplat.__file__)\""))
print("\n== higs env python: torch ==")
print(run("/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python -c \"import torch; print(torch.__version__)\""))
print("\n== higs env: higs-renderer import ==")
print(run("/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python -c \"import higs_render; print(higs_render.__file__)\" 2>&1 | tail -2"))
print("\n== pip list (higs env, filtered) ==")
print(run("/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/pip list 2>/dev/null | grep -i -E 'gsplat|diff|gaussian|higs|plyfile|simple'"))
print("\n== storage pool dirs ==")
print(run("ls /mnt/storage_pool/liaoyuanjun/ | head -40"))
print("\n== nvidia-smi ==")
print(run("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"))
print(run("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader"))
