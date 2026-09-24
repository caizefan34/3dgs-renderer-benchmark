import subprocess

def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90, cwd=cwd)
    return (r.stdout + r.stderr).strip()

C0WT = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
ACC = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"

print("== HigsNativeBackward.cu: scalar_adjoint / h8_mr / px_runtime block (1355-1515) ==")
print(run(f"sed -n '1355,1515p' {C0WT}/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu"))

print("\n== b1a accutile build identity manifest ==")
print(run("cat /mnt/storage_pool/liaoyuanjun/b1a_accutile_build_identity.json 2>/dev/null | head -60"))

print("\n== accutile tree diff vs upstream (ignore submodules) ==")
print(run(f"cd {ACC} && git status --ignore-submodules --short 2>&1 | head -15"))
print(run(f"cd {ACC} && git diff --ignore-submodules --stat 2>&1 | tail -10"))

print("\n== full reference/graphdeco listing ==")
print(run(f"ls -la {REPO}/reference/graphdeco/"))
print(run(f"head -60 {REPO}/reference/graphdeco/train.py"))
