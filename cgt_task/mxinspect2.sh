echo "=== room30k checkpoints ==="
ls -la /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_room_30k/checkpoints/ 2>&1
echo "=== c53 keys (room30k) ==="
/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python - <<'PY'
import json
for sc in ["room_30k","garden_30k","bicycle_30k"]:
    p=f"/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_{sc}/c53_workload_statistics.json"
    try:
        d=json.load(open(p))
        print(sc,"top-keys:",list(d.keys())[:10])
        cks=d.get("checkpoints",{})
        if cks:
            k=sorted(cks.keys())[-1]
            print("  ckpt",k,"subkeys",list(cks[k].keys()))
    except Exception as e:
        print(sc,"ERR",e)
PY
echo "=== data image counts ==="
for sc in room garden bicycle; do
  echo -n "$sc imgs="; ls /tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360/$sc/sparse/0/images/ 2>/dev/null | wc -l
  echo -n "$sc cams bin="; ls -la /tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360/$sc/sparse/0/cameras.bin 2>&1 | awk '{print $5}'
done
echo "=== timing all scenes ==="
for sc in garden_30k bicycle_30k; do echo "-- $sc --"; cat /mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs/B1_$sc/timing.json 2>&1 | /mnt/storage_pool/liaoyuanjun_mx0qual/bin/python -c "import json,sys; d=json.load(sys.stdin); print({k:(round(v.get('total_ms_mean',0),1),round(v.get('fwd_ms_mean',0),1),round(v.get('bwd_ms_mean',0),1)) for k,v in d.items()})" 2>&1; done