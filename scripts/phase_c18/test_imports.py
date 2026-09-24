import sys
sys.path.insert(0, ".")
sys.path.insert(0, "src")
from benchmark_framework import load_ply
import gsplat
import torch
print(f"gsplat={gsplat.__version__}, torch={torch.__version__}, gpu={torch.cuda.get_device_name(0)}")
data = load_ply("data/official/mipnerf360/room/point_cloud.ply", device="cuda")
print(f"PLY ok: {data['xyz'].shape}")
print("All imports OK")
