import subprocess, os

def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90, cwd=cwd)
    out = (r.stdout + r.stderr).strip()
    return out

C0WT = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
ACC = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"

print("== C0 worktree: env-switch handling ==")
print(run(f"grep -rn 'HIGS_PX_RUNTIME|HIGS_DISABLE_F9' {C0WT} --include='*.py' -l | head"))
print(run(f"grep -rn 'HIGS_PX_RUNTIME' {C0WT}/gsplat --include='*.py' | head -12"))
print(run(f"grep -rn 'HIGS_DISABLE_F9' {C0WT}/gsplat --include='*.py' | head -8"))
print(run(f"grep -rn 'HIGS_BWD_SCALAR_ADJOINT|HIGS_BWD_H8_MR' {C0WT}/gsplat --include='*.py' | head -12"))

print("\n== rasterize_gaussian_higs_dynamic signature ==")
print(run(f"grep -rn 'def rasterize_gaussian_higs_dynamic' {C0WT}/gsplat --include='*.py'"))
print(run(f"sed -n '1,40p' $(grep -rln 'def rasterize_gaussian_higs_dynamic' {C0WT}/gsplat --include='*.py' | head -1)"))

print("\n== repo baseline/ reference/ src/ dirs ==")
for d in ("baseline", "reference", "src", "configs"):
    print(f"-- {d}:", run(f"ls {REPO}/{d} 2>/dev/null | head -20"))

print("\n== strong_baseline_train.py header ==")
print(run(f"head -40 {REPO}/strong_baseline_train.py"))

print("\n== fastergs / speedy prior state ==")
print(run(f"head -30 {REPO}/fastergs_train.py"))
print(run(f"ls ~/Compute 2>/dev/null | head -15"))
print(run("ls ~/ 2>/dev/null | grep -i -E 'faster|speedy|fastgs|graphdeco|gaussian-splatting' | head"))

print("\n== accutile tree identity ==")
print(run(f"ls {ACC} | head -20"))
print(run(f"cd {ACC} && git log --oneline -3 2>/dev/null; git remote -v 2>/dev/null | head -2"))
