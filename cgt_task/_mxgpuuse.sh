echo "=== G2 rigorous launcher (which GPUs) ==="
grep -rn "CUDA_VISIBLE_DEVICES\|g2_rigorous\|rigorous" /home/liaoyuanjun/cgt_realscene/final/launch_*.sh /home/liaoyuanjun/cgt_realscene/final/*.sh 2>/dev/null | grep -i rigorous | head
echo "=== search any launch rig ==="
ls /home/liaoyuanjun/cgt_realscene/final/launch_*.sh 2>/dev/null
echo "=== dense logs GPU? (check log headers) ==="
head -2 /home/liaoyuanjun/cgt_realscene/final/logs/dense_garden_r30k.log
echo "=== ENVIRONMENT.txt ==="
cat /home/liaoyuanjun/cgt_realscene/final/PROVENANCE/ENVIRONMENT.txt 2>/dev/null
