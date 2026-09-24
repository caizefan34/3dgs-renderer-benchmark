import subprocess

def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90, cwd=cwd)
    return (r.stdout + r.stderr).strip()

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
ACC = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
C0WT = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
HIGS = "/home/liaoyuanjun/higs-13scene"

print("== reference/graphdeco identity ==")
print(run(f"cd {REPO}/reference/graphdeco && git log --oneline -3 && git status --short | head -5 && ls"))
print(run(f"cd {REPO}/reference/graphdeco && ls submodules 2>/dev/null; find . -name '*.so' -o -name 'diff_gaussian_rasterization*' 2>/dev/null | grep -v .git | head -8"))

print("\n== accutile tree: working-tree state (accutile = uncommitted patch?) ==")
print(run(f"cd {ACC} && git status --short | head -20"))
print(run(f"cd {ACC} && git diff --stat | tail -8"))

print("\n== C0 worktree git state ==")
print(run(f"cd {C0WT} && git log --oneline -3 && git status --short | head -10"))

print("\n== higs tree: PX_RUNTIME / SCALAR_ADJOINT / H8_MR handling (py + cpp) ==")
print(run(f"grep -rn 'HIGS_PX_RUNTIME' {C0WT} --include='*.py' --include='*.cu' --include='*.h' --include='*.cpp' -l 2>/dev/null | head -6"))
print(run(f"grep -rn 'HIGS_PX_RUNTIME' {C0WT} 2>/dev/null | grep -v '.git' | head -8"))
print(run(f"grep -rn 'HIGS_BWD_SCALAR_ADJOINT' {C0WT} 2>/dev/null | grep -v '.git' | head -6"))
print(run(f"grep -rn 'HIGS_BWD_H8_MR' {C0WT} 2>/dev/null | grep -v '.git' | head -6"))

print("\n== rasterize_gaussian_higs_dynamic kwargs (line 2990+) ==")
print(run(f"sed -n '2990,3040p' {C0WT}/gsplat/experimental/render/functional/gaussian_inference.py"))

print("\n== F9 disable context (line 1740-1790) ==")
print(run(f"sed -n '1745,1790p' {C0WT}/gsplat/experimental/render/functional/gaussian_inference.py"))

print("\n== strong_baseline_results inventory ==")
print(run("ls /mnt/storage_pool/liaoyuanjun/strong_baseline_results/ 2>/dev/null"))
print(run("ls /mnt/storage_pool/liaoyuanjun/strong_baseline_results/faster-gs/ 2>/dev/null | head -15"))
print(run("ls /mnt/storage_pool/liaoyuanjun/strong_baselines/ 2>/dev/null"))
