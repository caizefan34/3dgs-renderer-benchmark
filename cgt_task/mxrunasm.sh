cd /home/liaoyuanjun/cgt_realscene
rm -rf final; mkdir -p final
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python src/assemble_deliverables.py --logs logs/g1 --src src --out final 2>&1