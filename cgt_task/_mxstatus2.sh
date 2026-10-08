cd /home/liaoyuanjun/cgt_realscene/final
echo "=== log mtimes ==="
ls -la --time-style=+%H:%M:%S logs/dense_*.log logs/g3r_*.log logs/g1*.log 2>/dev/null
echo "=== g3 completion markers ==="
for f in logs/g3r_*.log; do echo "--- $f ---"; grep -iE "MICROBENCH|median|DONE|Traceback|Error|NO_TILES|saved|written" "$f" 2>/dev/null | tail -n 4; done
echo "=== dense JSON presence ==="
ls -la --time-style=+%H:%M:%S G2_CERT/ 2>/dev/null | tail -n 20
echo "=== running python procs ==="
ps -eo pid,etime,cmd | grep -iE "g1_compress|g2_cert|g3_gpu|dense" | grep -v grep
