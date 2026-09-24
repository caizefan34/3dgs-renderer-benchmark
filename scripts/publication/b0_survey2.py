import subprocess

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (r.stdout + r.stderr).strip()

B0 = "/mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub"
REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"

print("== B0 rasterizer settings fields + call signature ==")
print(run(f"grep -n 'class GaussianRasterizationSettings' -A 25 {B0}/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py"))
print(run(f"grep -n 'def forward' -A 22 {B0}/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py"))

print("\n== graphdeco getProjectionMatrix ==")
print(run(f"grep -rn 'def getProjectionMatrix' -A 18 {B0}/utils/graphics_utils.py"))

print("\n== graphdeco train.py render call site ==")
print(run(f"grep -n 'GaussianRasterizationSettings' -B 2 -A 14 {B0}/train.py"))
print(run(f"grep -n 'rasterizer(' -A 10 {B0}/train.py | head -14"))

print("\n== GTDataset camera fields ==")
print(run(f"grep -n 'def get_camera\\|def get_item\\|self.viewmatrix\\|self.K\\|camera_center\\|fov\\|image_width\\|image_height\\|class ' {REPO}/scripts/epic05/phase7/dataset.py | head -30"))
