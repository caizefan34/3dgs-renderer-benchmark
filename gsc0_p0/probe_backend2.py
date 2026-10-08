export PYTHONIOENCODING=utf-8
echo "===== _backend.py 180-210 (the _C = load_extension call) ====="
sed -n "180,210p" /root/epic05_frozen/gsplat-true-accutile-v153/gsplat/cuda/_backend.py
echo "===== _wrapper.py how it imports _C ====="
grep -nE "_C|import|_backend" /root/epic05_frozen/gsplat-true-accutile-v153/gsplat/cuda/_wrapper.py | head -15
echo "===== cache build dir contents ====="
ls -la /root/.cache/torch_extensions/py310_cu128/gsplat_cuda/ 2>&1
echo "===== does cached .so export the expected symbols? ====="
/root/miniforge3/envs/gsplat/bin/python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("gsplat_cuda_pre", "/root/.cache/torch_extensions/py310_cu128/gsplat_cuda/gsplat_cuda.so")
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
    print("LOADED OK; attrs:", [a for a in dir(m) if not a.startswith("__")][:20])
except Exception as e:
    print("LOAD FAIL:", repr(e)[:200])
PY
echo "===== versioner source (the crash site) ====="
sed -n "1,60p" /root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/utils/_cpp_extension_versioner.py