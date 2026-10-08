cd /home/liaoyuanjun/cgt_realscene/final
echo "=== NOW ==="; date
echo "=== G1 JSONs (mtime) ==="
ls -la --time-style=+%H:%M:%S G1_RAW_RECORDS/g1_*.json 2>/dev/null
echo "=== ALL DONE marker in G1 logs ==="
grep -l "ALL DONE" logs/g1*.log 2>/dev/null
echo "=== dense logs tail ==="
for f in logs/dense_*.log; do echo "--- $f ---"; tail -n 3 "$f" 2>/dev/null; done
echo "=== g3 logs tail ==="
for f in logs/g3r_*.log; do echo "--- $f ---"; tail -n 3 "$f" 2>/dev/null; done
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
