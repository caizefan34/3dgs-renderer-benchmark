import gsplat
print("gsplat version:", gsplat.__version__)
print("gsplat path:", gsplat.__file__)
import os
gsplat_dir = os.path.dirname(gsplat.__file__)
# Check for compiled CUDA extension
for root, dirs, files in os.walk(gsplat_dir):
    for f in files:
        if f.endswith('.so'):
            print("SO:", os.path.join(root, f))
        if f.endswith('_backend.py'):
            print("Backend:", os.path.join(root, f))
