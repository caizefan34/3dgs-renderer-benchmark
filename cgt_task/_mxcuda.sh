echo "=== search nvcc 11.8 / cuda-11.8 ==="
ls -d /usr/local/cuda-11.8 2>/dev/null; find /opt /mnt/storage_pool -maxdepth 4 -name nvcc -type f 2>/dev/null | head
echo "=== pip nvidia cuda packages in runtime ==="
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/pip list 2>/dev/null | grep -iE "nvidia|cuda|nvcc" | head
echo "=== conda envs ==="
ls /mnt/storage_pool/*/envs 2>/dev/null | head; ls ~/miniconda3/envs ~/anaconda3/envs 2>/dev/null | head
echo "=== torch's bundled cuda libs ==="
/mnt_storage python -c "import torch,os;print(os.path.dirname(torch.__file__))" 2>/dev/null
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python -c "import torch,os;print(os.path.dirname(torch.__file__))"
ls /mnt/storage_pool/liaoyuanjun_mx0qual/lib/python3.10/site-packages/nvidia/ 2>/dev/null
