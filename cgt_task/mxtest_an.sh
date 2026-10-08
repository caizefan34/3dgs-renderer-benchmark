/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python - <<'PYEOF'
import sys
sys.argv=['x','--inputs','/home/liaoyuanjun/cgt_realscene/logs/smoke.json','--out','/tmp/g1_test.json']
exec(open('/home/liaoyuanjun/cgt_realscene/src/analyze_g1.py').read().replace("if __name__ == \"__main__\":\n    main()",""))
PYEOF