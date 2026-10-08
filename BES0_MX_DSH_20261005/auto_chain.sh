#!/bin/bash
# BES0 full auto-chain: E0 -> E1 -> E2 -> E3 (guarded launcher), then
# collect_review -> reduce_verdict -> auto_report -> package_results.
# Re-checks allowed-GPU idleness before EACH stage; retries transient guard
# errors (rc=2); Halts only on genuine stage failure/budget. Detached.
W=/mnt/storage_pool/liaoyuanjun/bes0_mx
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
DSH="$W/BES0_MX_DSH"
LOG="$W/auto_chain.log"
BUDGET_CAP=$((12*3600))
cd "$DSH" || { echo "CHAIN_ABORT no DSH dir" >> "$LOG"; exit 1; }
echo "=== CHAIN_START $(date -u +%H:%M:%S) ===" >> "$LOG"

find_idle_uuid() {
  for idx in 1 2 3 4; do
    line=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i $idx 2>/dev/null)
    mem=$(echo "$line" | awk -F", " "{print \$2}")
    util=$(echo "$line" | awk -F", " "{print \$3}")
    procs=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i $idx 2>/dev/null | grep -c .)
    if [ -n "$mem" ] && [ "$mem" -le 512 ] 2>/dev/null && [ "$util" -le 2 ] 2>/dev/null && [ "$procs" -eq 0 ] 2>/dev/null; then
      nvidia-smi --query-gpu=index,uuid --format=csv,noheader -i $idx | awk -F", " "{print \$2}"
      return 0
    fi
  done
  return 1
}

wait_for_idle() {
  while true; do
    u=$(find_idle_uuid)
    if [ -n "$u" ]; then echo "$u"; return 0; fi
    sleep 30
  done
}

charged_sec() {
  "$PY" - <<'PYEOF'
import json,glob,os
base="/mnt/storage_pool/liaoyuanjun/bes0_mx"
c=0.0
for cp in glob.glob(base+"/*/*/COMMAND.json"):
    sp=cp.replace("COMMAND.json","RUN_STATUS.json")
    c+= float(json.load(open(sp))["elapsed_wall_seconds"]) if os.path.exists(sp) else float(json.load(open(cp))["timeout_seconds"])
print(int(c))
PYEOF
}

stage_json_name() {
  case "$1" in
    e0) echo E0.json ;;
    e1) echo E1.json ;;
    e2) echo E2.json ;;
    e3) echo E3.json ;;
  esac
}

run_stage() {
  local runid="$1" stage="$2" entry="$3" mins="$4"
  local attempt=0 maxtry=4
  while [ $attempt -lt $maxtry ]; do
    attempt=$((attempt+1))
    echo "--- WAIT_IDLE $stage attempt=$attempt $(date -u +%H:%M:%S) ---" >> "$LOG"
    local uuid; uuid=$(wait_for_idle)
    echo "--- LAUNCH $stage runid=$runid uuid=$uuid attempt=$attempt $(date -u +%H:%M:%S) ---" >> "$LOG"
    "$PY" scripts/run_on_mx.py --run-id "$runid" --gpu-uuid "$uuid" \
      --stage "$stage" --entry "$entry" --max-minutes "$mins" >> "$LOG" 2>&1
    local rc=$?
    local sd="$W/$runid/$stage"
    local st; st=$(grep -o '"status": *"[^"]*"' "$sd/RUN_STATUS.json" 2>/dev/null | head -1 | sed 's/.*"status": *"//;s/"//')
    echo "--- STAGE_DONE $stage rc=$rc status=$st attempt=$attempt $(date -u +%H:%M:%S) ---" >> "$LOG"
    if [ "$st" = "COMPLETED" ]; then return 0; fi
    if [ "$rc" = "2" ] && [ $attempt -lt $maxtry ]; then
      echo "--- RETRY $stage (guard rc=2) ---" >> "$LOG"
      sleep 60
      continue
    fi
    echo "CHAIN_HALT $stage rc=$rc status=$st" >> "$LOG"
    return 1
  done
  echo "CHAIN_HALT $stage exhausted retries" >> "$LOG"
  return 1
}

run_stage bes0_e0_chain e0 scripts/run_e0.py 20 || exit 10
run_stage bes0_e1_chain e1 scripts/run_e1.py 60 || exit 11
run_stage bes0_e2_chain e2 scripts/run_e2.py 180 || exit 12

charged=$(charged_sec)
echo "--- BUDGET charged=${charged}s cap=${BUDGET_CAP}s ; E3 needs $((420*60))s $(date -u +%H:%M:%S) ---" >> "$LOG"
if [ $((charged + 420*60)) -gt "$BUDGET_CAP" ]; then
  echo "CHAIN_SKIP_E3_BUDGET charged=${charged}s remaining=$(( (BUDGET_CAP-charged)/60 ))min < 420min $(date -u +%H:%M:%S)" >> "$LOG"
else
  run_stage bes0_e3_chain e3 scripts/run_e3.py 420 || echo "CHAIN_E3_INCOMPLETE (deliverable still produced)" >> "$LOG"
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REVIEW="$W/review_$STAMP"
echo "--- COLLECT start review=$REVIEW $(date -u +%H:%M:%S) ---" >> "$LOG"
CARG=(scripts/collect_review.py --output "$REVIEW" --selected-config "$DSH/config/selected_config.json")
for s in e0 e1 e2 e3; do
  d="$W/bes0_${s}_chain_$s"
  jn=$(stage_json_name "$s")
  if [ -f "$d/$jn" ]; then
    CARG+=(--$s "$d")
    echo "collect: including $s ($d/$jn)" >> "$LOG"
  else
    echo "collect: $s NOT_RUN (missing $d/$jn)" >> "$LOG"
  fi
done
"$PY" "${CARG[@]}" >> "$LOG" 2>&1 && echo "--- COLLECT ok ---" >> "$LOG" || { echo "CHAIN_HALT collect failed" >> "$LOG"; exit 13; }

echo "--- REDUCE start $(date -u +%H:%M:%S) ---" >> "$LOG"
( cd scripts && "$PY" reduce_verdict.py "$REVIEW" ) >> "$LOG" 2>&1
echo "--- REDUCE rc=$? verdict=$(grep -o '\"verdict\": *\"[^\"]*\"' "$REVIEW/VERDICT.json" 2>/dev/null | head -1) $(date -u +%H:%M:%S) ---" >> "$LOG"

echo "--- REPORT start $(date -u +%H:%M:%S) ---" >> "$LOG"
( cd scripts && "$PY" auto_report.py "$REVIEW" ) >> "$LOG" 2>&1 && echo "--- REPORT ok ---" >> "$LOG" || echo "CHAIN_HALT report failed" >> "$LOG"

ZIP="$W/BES0_MX_DSH_results_$STAMP.zip"
echo "--- PACKAGE start zip=$ZIP $(date -u +%H:%M:%S) ---" >> "$LOG"
"$PY" scripts/package_results.py "$REVIEW" "$ZIP" >> "$LOG" 2>&1 && echo "--- PACKAGE ok zip=$ZIP ---" >> "$LOG" || echo "CHAIN_HALT package failed" >> "$LOG"

echo "=== CHAIN_ALL_DONE $(date -u +%H:%M:%S) ===" >> "$LOG"
