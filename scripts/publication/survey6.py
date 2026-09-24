import subprocess

def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90, cwd=cwd)
    return (r.stdout + r.stderr).strip()

ACC = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
C0WT = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"

print("== all HIGS_ env vars read by the C0 build (cpp + py) ==")
print(run(f"grep -rhn 'getenv(\"HIGS_' {C0WT}/gsplat --include='*.cu' --include='*.h' --include='*.cpp' | sed 's/^ *//' | sort -u | head -20"))
print(run(f"grep -rhn 'getenv(\"HIGS_' {C0WT}/gsplat --include='*.py' | sed 's/^ *//' | sort -u | head -20"))

print("\n== accutile base commit identity ==")
print(run(f"cd {ACC} && git rev-parse HEAD && git describe --tags 2>/dev/null; git log -1 --format='%H %ad %s' --date=short"))

print("\n== reference_v1 config: optimizer (B0 gate input) ==")
print(run(f"cat {REPO}/baseline/reference_v1/config.py"))
