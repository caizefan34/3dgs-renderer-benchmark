import subprocess

def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90, cwd=cwd)
    return (r.stdout + r.stderr).strip()

ACC = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"

print("== accutile rendering.py diff (is absgrad upstream or patched?) ==")
print(run(f"cd {ACC} && git diff --ignore-submodules -- gsplat/rendering.py"))
print("\n== accutile _wrapper.py diff (absgrad plumb) ==")
print(run(f"cd {ACC} && git diff --ignore-submodules -- gsplat/cuda/_wrapper.py"))
print("\n== absgrad occurrences in accutile tree python ==")
print(run(f"grep -rn 'absgrad' {ACC}/gsplat/*.py {ACC}/gsplat/cuda/_wrapper.py 2>/dev/null | head -10"))
print("\n== upstream gsplat 1.5.3 rasterization signature (accutile tree file) ==")
print(run(f"grep -n 'def rasterization' {ACC}/gsplat/rendering.py"))
print(run(f"sed -n \"$(grep -n 'def rasterization' {ACC}/gsplat/rendering.py | head -1 | cut -d: -f1),+40p\" {ACC}/gsplat/rendering.py | grep -E 'def |absgrad|eps2d|packed|tile_size|radius_clip'"))
print("\n== accutile EXPLORATION.md: eps2d 0.1 provenance ==")
print(run(f"grep -n -i 'eps2d' {ACC}/EXPLORATION.md | head -8"))
print("\n== b1a_build.py (local repo, mx copy?) ==")
print(run(f"ls {REPO}/scripts/final30k/ 2>/dev/null | head; ls {REPO}/scripts/ 2>/dev/null | head -20"))
print("\n== strong_baseline prior results: faster-gs summary ==")
print(run("cat /mnt/storage_pool/liaoyuanjun/strong_baseline_results/faster-gs/training_summary.json 2>/dev/null | head -40"))
print(run("cat /mnt/storage_pool/liaoyuanjun/strong_baseline_results/all_metrics.json 2>/dev/null | head -30"))
