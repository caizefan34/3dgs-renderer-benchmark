cd /home/liaoyuanjun/cgt_realscene/final
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
cp SCRIPTS/g2_benefit_budget.py ./g2_benefit_budget.py
echo "=== G2 benefit budget ==="; $PY g2_benefit_budget.py 2>&1
echo "=== fail_localize ==="; $PY SCRIPTS/fail_localize.py --inputs "G1_RAW_RECORDS/g1_*.json" --out G2_FAILURE_LOCALIZATION.json 2>&1
