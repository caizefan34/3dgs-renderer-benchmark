set -e
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
ROOT=/mnt/storage_pool/liaoyuanjun/acg0_mx/ACG0_MX_DSH
SYN=/mnt/storage_pool/liaoyuanjun/acg0_mx/acg0_synth
rm -rf $SYN; mkdir -p $SYN
TSHA=$(sha256sum $ROOT/config/mx_target.json | cut -d" " -f1)
RUN=acg0_test
GPU=GPU-6d75016d-fc44-9443-9190-396375f20f6a
for st in e0 e1 e2 e3; do
  d=$SYN/$st; mkdir -p $d
  printf '{"schema":"acg0.host_evidence.v1","hostname":"bms-39468022-001","target_config_sha256":"%s","gpu_uuid":"%s","run_id":"%s","stage":"%s","gpu_inventory":[{"index":4,"uuid":"%s","name":"NVIDIA A100-PCIE-40GB","memory_total_mib":40960,"memory_used_mib":0,"utilization_percent":0}],"checkpoint_sha256":"744cfd83a13c199fc8b2ef0dd69f142e34d59edffbf1b7ae7f86b9a5c64e10fc"}\n' "$TSHA" "$GPU" "$RUN" "$st" "$GPU" > $d/GUARD.json
  echo '{"argv":["x"],"timeout_seconds":60}' > $d/COMMAND.json
  echo '{"status":"COMPLETED","exit_code":0,"elapsed_wall_seconds":1.0}' > $d/RUN_STATUS.json
  echo '{}' > $d/console.log
done
cd $ROOT
PROT=$($PY -c "import sys;sys.path.insert(0,'scripts');from provenance import protocol_digest;print(protocol_digest())")
IMPL=$($PY -c "import sys;sys.path.insert(0,'scripts');from provenance import implementation_digest;print(implementation_digest())")
printf '{"schema":"acg0.stage.v1","stage":"e0","run_id":"%s","protocol_sha256":"%s","implementation_sha256":"%s","selected_config_sha256":null,"status":"PASS","reason":"ok","evidence":["E0_raw.json"],"checks_passed":true,"test_count":10}\n' "$RUN" "$PROT" "$IMPL" > $SYN/e0/E0.json
echo '{"checks":[]}' > $SYN/e0/E0_raw.json
printf '{"schema":"acg0.stage.v1","stage":"e1","run_id":"%s","protocol_sha256":"%s","implementation_sha256":"%s","selected_config_sha256":null,"status":"BLOCKED","reason":"room only; bicycle/garden absent","evidence":["E1_raw.json"],"adapter_equivalent":true,"native_semantics_frozen":true,"no_online_oracle":true,"all_test_views_present":false,"fallback_full_composite":true,"production_routing_frozen":true,"scene_metrics":[]}\n' "$RUN" "$PROT" "$IMPL" > $SYN/e1/E1.json
echo '{}' > $SYN/e1/E1_raw.json
printf '{"schema":"acg0.stage.v1","stage":"e2","run_id":"%s","protocol_sha256":"%s","implementation_sha256":"%s","selected_config_sha256":null,"status":"NOT_RUN","reason":"E1 not PASS","evidence":["E2_raw.json"]}\n' "$RUN" "$PROT" "$IMPL" > $SYN/e2/E2.json
echo '{}' > $SYN/e2/E2_raw.json
printf '{"schema":"acg0.stage.v1","stage":"e3","run_id":"%s","protocol_sha256":"%s","implementation_sha256":"%s","selected_config_sha256":null,"status":"NOT_RUN","reason":"E2 not PASS","evidence":["E3_raw.json"]}\n' "$RUN" "$PROT" "$IMPL" > $SYN/e3/E3.json
echo '{}' > $SYN/e3/E3_raw.json
cd $ROOT
echo "[reduce]"; $PY scripts/reduce_verdict.py $SYN | tail -n 20
echo "[report]"; $PY make_report.py $SYN
echo "[package]"; $PY scripts/package_results.py $SYN /mnt/storage_pool/liaoyuanjun/acg0_mx/ACG0_RESULTS_SYNTH.zip
