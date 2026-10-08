echo "=== cuda118 nvcc ==="
/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118/bin/nvcc --version 2>&1 | tail -3
echo "=== g1_compressibility.py: predict + tile_teacher + fit (numpy) ==="
grep -n "def predict\|def tile_teacher\|def fit_cgt\|def expert_values\|def predict_and_vjp\|softmax\|sigmoid\|def q2R\|POWER_CLIP" /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g1_compressibility.py | head -40
