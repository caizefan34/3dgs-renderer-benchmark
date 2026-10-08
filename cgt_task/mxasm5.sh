cd /home/liaoyuanjun/cgt_realscene
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python src/assemble_deliverables.py --logs logs/g1 --src src --out final 2>&1
echo "=== G2_G5_STATUS G1 block ==="
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python -c "import json;d=json.load(open('final/G2_G5_STATUS.json'));print(json.dumps(d['G1_compressibility'],indent=1))"