export PYTHONIOENCODING=utf-8
echo "===== build_params.json (provenance) ====="
cat /root/.cache/torch_extensions/py310_cu128/gsplat_cuda/build_params.json 2>&1 | head -40
echo "===== .hash file? ====="
ls -la /root/.cache/torch_extensions/py310_cu128/gsplat_cuda/.hash 2>&1
echo "===== accutile symbols in cached .so? ====="
strings /root/.cache/torch_extensions/py310_cu128/gsplat_cuda/gsplat_cuda.so 2>/dev/null | grep -i accutile | head -10
echo "===== count accutile symbol hits ====="
strings /root/.cache/torch_extensions/py310_cu128/gsplat_cuda/gsplat_cuda.so 2>/dev/null | grep -ci accutile