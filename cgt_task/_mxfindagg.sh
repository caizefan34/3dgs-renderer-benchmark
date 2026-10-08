echo "=== search build/agg scripts on MX ==="
ls /home/liaoyuanjun/cgt_realscene/final/*.py /home/liaoyuanjun/cgt_realscene/final/*.sh 2>/dev/null
echo "--- grep for G3_GPU_MICROBENCH writers ---"
grep -rl "G3_GPU_MICROBENCH\|G3_gpu_local_microbench\|per_tile_gamma" /home/liaoyuanjun/cgt_realscene/final/ 2>/dev/null | head
echo "--- grep for G2_CONTINUOUS_DOMAIN aggregate ---"
grep -rl "G2_CONTINUOUS_DOMAIN_CERTIFICATE\b\|dense_vs_heldout\|pass_rate" /home/liaoyuanjun/cgt_realscene/final/*.py /home/liaoyuanjun/cgt_realscene/final/*.sh 2>/dev/null | head
echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
