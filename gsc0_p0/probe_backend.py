export PYTHONIOENCODING=utf-8
echo "===== _backend.py load_extension (1-120) ====="
sed -n "1,120p" /root/epic05_frozen/gsplat-true-accutile-v153/gsplat/cuda/_backend.py
echo "===== existing built _C.so in torch extensions cache ====="
find /root/.cache/torch_extensions -name "*.so" 2>/dev/null | head
find /root/epic05_frozen/gsplat-true-accutile-v153 -name "*.so" 2>/dev/null | head