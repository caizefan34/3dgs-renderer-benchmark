cd /home/liaoyuanjun/cgt_realscene
echo "=== json produced ==="
ls -la logs/g1/*.json 2>&1
echo "=== gpu5/6/7 tails ==="
for g in 5 6 7; do echo "GPU$g:"; tail -2 logs/g1/gpu$g.out 2>&1; done
echo "=== procs alive ==="
pgrep -af "g1_compressibility" | wc -l