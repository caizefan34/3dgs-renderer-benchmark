echo "=== G1 alive ==="; ps -eo pid,cmd|grep g1_compress|grep -v grep|wc -l
echo "=== G3_GPU_MICROBENCH.json (current/old) ==="
python3 -c "import json;d=json.load(open('/home/liaoyuanjun/cgt_realscene/final/G3_GPU_MICROBENCH.json'));print(json.dumps(d,indent=1)[:1500])"
