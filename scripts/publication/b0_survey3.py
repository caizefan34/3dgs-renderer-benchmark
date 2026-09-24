import subprocess

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (r.stdout + r.stderr).strip()

C = "/home/liaoyuanjun/3dgs-renderer-benchmark/src/benchmark_framework/cameras.py"
print("== Camera class attrs ==")
print(run(f"grep -n 'self\\.' {C} | head -30"))
print("\n== load_cameras_from_json fov/proj construction ==")
print(run(f"grep -n 'def load_cameras_from_json' -A 45 {C} | head -60"))
print("\n== tanfovx anywhere ==")
print(run(f"grep -n 'tanfovx\\|tanfov\\|fov_x\\|fovX' {C} | head -10"))
print("\n== projmatrix / world_view_transform construction ==")
print(run(f"grep -n 'projmatrix\\|world_view_transform\\|full_proj\\|FoVx\\|getProjectionMatrix' {C} | head -20"))
