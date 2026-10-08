#!/usr/bin/env bash
# TI0-A guarded auto-fire launcher.
# Waits for a CLEAN GPU in the allowed set {0,2,3,4,5,6,7} (GPU1 = comax0 lane, excluded),
# then runs TI0-A on it to completion. Records GPU provenance. Never touches GPU1.
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
HARNESS=/mnt/storage_pool/liaoyuanjun/cgt_ti0/src/ti0a_update_cost.py
SRCDIR=/mnt/storage_pool/liaoyuanjun/cgt_ti0/src
RES=/mnt/storage_pool/liaoyuanjun/cgt_ti0/results
LOG=/mnt/storage_pool/liaoyuanjun/cgt_ti0/ti0a_full.log
PROV=/mnt/storage_pool/liaoyuanjun/cgt_ti0/_gpu_provenance.json
ALLOWED=(0 2 3 4 5 6 7)
CLEAN_MIB=500
POLL_S=60
MAX_ITERS=${1:-120}   # up to ~2h at 60s
export TMPDIR=/mnt/storage_pool/liaoyuanjun/cgt_ti0/tmp
mkdir -p "$TMPDIR"
mkdir -p "$RES"

clean_gpu() {
  # echo first allowed gpu with mem.used < CLEAN_MIB, else empty
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
  while IFS=',' read -r idx mem; do
    idx=$(echo "$idx" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
    for a in "${ALLOWED[@]}"; do
      if [ "$idx" = "$a" ] && [ "$mem" -lt "$CLEAN_MIB" ] 2>/dev/null; then echo "$idx"; return; fi
    done
  done | head -n1
}

echo "[launcher] start $(date) allowed={0,2,3,4,5,6,7} clean<${CLEAN_MIB}MiB poll=${POLL_S}s max=${MAX_ITERS}" | tee -a "$LOG"
for i in $(seq 1 "$MAX_ITERS"); do
  G=$(clean_gpu)
  if [ -n "${G:-}" ]; then
    # stability re-check after 5s
    sleep 5
    G2=$(clean_gpu)
    if [ -n "${G2:-}" ] && [ "$G2" = "$G" ]; then
      MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G" | tr -d ' ')
      UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$G" | tr -d ' ')
      echo "[launcher] CLEAN GPU=$G mem=${MEM}MiB util=${UTIL}% at $(date) -> firing" | tee -a "$LOG"
      printf '{"ts":"%s","physical_gpu":%s,"mem_used_mib":%s,"util_pct":%s,"cuda_visible":"%s"}\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$G" "$MEM" "$UTIL" "$G" > "$PROV"
      CUDA_VISIBLE_DEVICES=$G $PY "$HARNESS" \
        --views 6 --max-tiles 10 --fit-iters 200 --fwd-repeats 100 --fwd-warmup 20 \
        --device 0 --out "$RES/TI0A_UPDATE_COST.json" >> "$LOG" 2>&1
      RC=$?
      echo "[launcher] harness exit=$RC at $(date)" | tee -a "$LOG"
      # merge provenance into the TI0A json
      $PY - "$RES/TI0A_UPDATE_COST.json" "$PROV" <<'PYEOF'
import json,sys
j=json.load(open(sys.argv[1])); prov=json.load(open(sys.argv[2]))
j["gpu_provenance"]=prov
json.dump(j, open(sys.argv[1],"w"), indent=2)
print("[launcher] merged gpu_provenance:", prov)
PYEOF
      # produce downstream deliverables (fail-closed) from TI0A result
      $PY "$SRCDIR/ti0_finalize.py" --ti0a "$RES/TI0A_UPDATE_COST.json" --outdir "$RES" >> "$LOG" 2>&1
      echo "[launcher] finalize exit=$? at $(date)" | tee -a "$LOG"
      echo "[launcher] DONE $(date)" | tee -a "$LOG"
      exit $RC
    else
      echo "[launcher] poll $i: unstable ($G -> ${G2:-none}), keep waiting" | tee -a "$LOG"
    fi
  else
    echo "[launcher] poll $i: no clean allowed GPU at $(date)" | tee -a "$LOG"
  fi
  sleep "$POLL_S"
done
echo "[launcher] TIMEOUT no clean GPU after ${MAX_ITERS} polls $(date)" | tee -a "$LOG"
exit 3