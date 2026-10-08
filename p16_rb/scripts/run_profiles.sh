#!/bin/bash
# P16-RB Stage: bottleneck profiling replays (prereg step 11).
# - copies the frozen source trees, applies timing-only instrumentation, stores diffs
# - A/B/A neutrality legs (seeded, 320 iters, room) per method on the patched trees
# - full profiled replay per (method, scene) with record, on GPU 2, sequential
# - analysis -> profiles/<M>/<SC>_{mid,final}.json
# Not a timing record: no wrapper, no sampler, no contamination captures (logging only).
set -u
D=/root/epic05_p16rb
GPU=2
PY=/root/miniforge3/envs/p16_fastergs/bin/python
PROF=$D/profiles
RAWSRC=$D/src
COPY=$D/profile_src
LOGS=$D/logs
SM=$D/scripts
source $SM/p16_recipes.sh
: > $LOGS/run_profiles.log
log() { echo "[$(date -Is)] $*" | tee -a $LOGS/run_profiles.log; }

mkdir -p $PROF/_raw $COPY
NPROF_W=20

# ---------------------------------------------------------------- copies + patches
log "sync profile source copies"
rsync -a --delete --exclude __pycache__ $RAWSRC/nerficg/ $COPY/nerficg/
rsync -a --delete --exclude __pycache__ $RAWSRC/FastGS/ $COPY/FastGS/
rsync -a --delete --exclude __pycache__ $RAWSRC/SADGS/ $COPY/SADGS/

$PY $SM/p16_prof_patch_nerficg.py --tree $COPY/nerficg --prof-module $SM/p16_prof.py \
  --report $PROF/_raw/patch_report_nerficg.json || { log "PATCH_FAIL nerficg"; exit 1; }
$PY $SM/p16_prof_patch_graphdeco.py --tree $COPY/FastGS --method FastGS --prof-module $SM/p16_prof.py \
  --report $PROF/_raw/patch_report_FastGS.json || { log "PATCH_FAIL FastGS"; exit 1; }
$PY $SM/p16_prof_patch_graphdeco.py --tree $COPY/SADGS --method SADGS --prof-module $SM/p16_prof.py \
  --report $PROF/_raw/patch_report_SADGS.json || { log "PATCH_FAIL SADGS"; exit 1; }

( cd $RAWSRC && diff -ruN --exclude=__pycache__ nerficg $COPY/nerficg > $PROF/_raw/instrumentation_nerficg.diff; true )
( cd $RAWSRC && diff -ruN --exclude=__pycache__ FastGS $COPY/FastGS > $PROF/_raw/instrumentation_FastGS.diff; true )
( cd $RAWSRC && diff -ruN --exclude=__pycache__ SADGS $COPY/SADGS > $PROF/_raw/instrumentation_SADGS.diff; true )

$PY -m py_compile $COPY/nerficg/src/p16_prof.py $COPY/nerficg/src/Methods/Base/Trainer.py \
  $COPY/nerficg/src/Methods/FasterGS/Trainer.py $COPY/nerficg/src/Methods/FasterGSFused/Trainer.py \
  || { log "PATCH_SYNTAX_FAIL nerficg"; exit 1; }
$PY -m py_compile $COPY/FastGS/train.py $COPY/FastGS/p16_prof.py || { log "PATCH_SYNTAX_FAIL FastGS"; exit 1; }
$PY -m py_compile $COPY/SADGS/train.py $COPY/SADGS/p16_prof.py || { log "PATCH_SYNTAX_FAIL SADGS"; exit 1; }
log "patches applied; syntax OK; diffs stored"

gpu_note() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tee -a $LOGS/run_profiles.log; }

# ---------------------------------------------------------------- A/B/A neutrality
ab_nerficg() {  # $1=method $2=lower
  local M=$1
  local LOW=$2
  local ABI=$PROF/_ab/$M
  mkdir -p $ABI
  local LEG PROFON
  for LEG in a1 a2 b; do
    sed -e "s|^  MODEL_NAME: .*|  MODEL_NAME: p16prof_ab_${M}_${LEG}|" \
        -e "s|^  NUM_ITERATIONS: .*|  NUM_ITERATIONS: 320|" \
        -e "s|^    RENDER_TESTSET: .*|    RENDER_TESTSET: false|" \
        $D/configs/p16_${LOW}_room_1920.yaml > $ABI/${LEG}.yaml
    PROFON=0; [ "$LEG" = b ] && PROFON=1
    log "AB $M $LEG start (P16_PROF=$PROFON)"
    ( cd $COPY/nerficg && \
      P16_PROF=$PROFON P16_PROF_OUT=$ABI/$LEG P16_PROF_N=30000 P16_PROF_BASE=0 \
      P16_PROF_W=$NPROF_W P16_PROF_METHOD=$M P16_PROF_SCENE=room P16_PROF_SEED=0 \
      PATH=/root/miniforge3/envs/p16_fastergs/bin:$PATH \
      $PY scripts/train.py -c $ABI/${LEG}.yaml ) > $LOGS/prof_ab_${M}_${LEG}.stdout.log 2>&1
    log "AB $M $LEG rc=$?"
  done
  $PY $SM/p16_prof_equiv.py --a1 $ABI/a1/iterations.csv --a2 $ABI/a2/iterations.csv \
    --b $ABI/b/iterations.csv --out $PROF/_raw/equiv_${M}.json --label "$M room" || true
  log "EQUIV $M -> $(grep -m1 '"status"' $PROF/_raw/equiv_${M}.json 2>/dev/null)"
}

ab_graphdeco() {  # $1=method(T=FastGS|SADGS) $2=env
  local T=$1
  local ENV=$2
  local ABI=$PROF/_ab/$T
  local MPY=/root/miniforge3/envs/${ENV}/bin/python
  mkdir -p $ABI
  local LEG PROFON
  for LEG in a1 a2 b; do
    PROFON=0; [ "$LEG" = b ] && PROFON=1
    log "AB $T $LEG start (P16_PROF=$PROFON)"
    ( cd $COPY/$T && \
      P16_PROF=$PROFON P16_PROF_OUT=$ABI/$LEG P16_PROF_N=30000 P16_PROF_BASE=1 \
      P16_PROF_W=$NPROF_W P16_PROF_METHOD=$T P16_PROF_SCENE=room P16_PROF_SEED=0 \
      CUDA_VISIBLE_DEVICES=$GPU OAR_JOB_ID=p16prof_ab_${T}_${LEG} \
      $MPY $(graphdeco_recipe $T room $(native_n $T)) --iterations 320 ) > $LOGS/prof_ab_${T}_${LEG}.stdout.log 2>&1
    log "AB $T $LEG rc=$?"
  done
  $PY $SM/p16_prof_equiv.py --a1 $ABI/a1/iterations.csv --a2 $ABI/a2/iterations.csv \
    --b $ABI/b/iterations.csv --out $PROF/_raw/equiv_${T}.json --label "$T room" || true
  log "EQUIV $T -> $(grep -m1 '"status"' $PROF/_raw/equiv_${T}.json 2>/dev/null)"
}

# ---------------------------------------------------------------- full replays
full_nerficg() {  # $1=method $2=lower $3=scene
  local M=$1 LOW=$2 SC=$3 N
  N=$(native_n $M)
  local CFG=$D/configs/prof_${M}_${SC}.yaml OUT=$PROF/_raw/${M}_${SC}
  sed -e "s|^  MODEL_NAME: .*|  MODEL_NAME: p16prof_${M}_${SC}|" \
      $D/configs/p16_${LOW}_${SC}_1920.yaml > $CFG
  rm -rf $OUT; mkdir -p $OUT
  log "FULL $M $SC start (N=$N)"; gpu_note
  ( cd $COPY/nerficg && \
    P16_PROF=1 P16_PROF_OUT=$OUT P16_PROF_N=$N P16_PROF_BASE=0 P16_PROF_W=$NPROF_W \
    P16_PROF_METHOD=$M P16_PROF_SCENE=$SC \
    PATH=/root/miniforge3/envs/p16_fastergs/bin:$PATH \
    $PY scripts/train.py -c $CFG ) > $LOGS/prof_${M}_${SC}.stdout.log 2>&1
  log "FULL $M $SC rc=$?"
  $PY $SM/p16_profile_analyze.py --rawdir $OUT --method $M --scene $SC --outdir $PROF/$M || log "ANALYZE_FAIL $M $SC"
  echo "PROFILE_${M}_${SC}_DONE $(date -Is)" | tee -a $LOGS/run_profiles.log
}

full_graphdeco() {  # $1=method $2=env $3=scene
  local T=$1
  local ENV=$2
  local SC=$3
  local N
  local OUT=$PROF/_raw/${T}_${SC}
  local MPY=/root/miniforge3/envs/${ENV}/bin/python
  N=$(native_n $T)
  rm -rf $OUT; mkdir -p $OUT
  log "FULL $T $SC start (N=$N)"; gpu_note
  ( cd $COPY/$T && \
    P16_PROF=1 P16_PROF_OUT=$OUT P16_PROF_N=$N P16_PROF_BASE=1 P16_PROF_W=$NPROF_W \
    P16_PROF_METHOD=$T P16_PROF_SCENE=$SC \
    CUDA_VISIBLE_DEVICES=$GPU OAR_JOB_ID=p16prof_${T}_${SC} \
    $MPY $(graphdeco_recipe $T $SC $(native_n $T)) ) > $LOGS/prof_${T}_${SC}.stdout.log 2>&1
  log "FULL $T $SC rc=$?"
  $PY $SM/p16_profile_analyze.py --rawdir $OUT --method $T --scene $SC --outdir $PROF/$T || log "ANALYZE_FAIL $T $SC"
  echo "PROFILE_${T}_${SC}_DONE $(date -Is)" | tee -a $LOGS/run_profiles.log
}

# ---------------------------------------------------------------- main sequence
ab_nerficg FasterGS fastergs
ab_nerficg FasterGSFused fastergsfused
ab_graphdeco FastGS p16_fastgs
ab_graphdeco SADGS p16_sadgs

equiv_pass() {  # $1=method; prereg step 11: instrumented run must match control cameras+loss
  $PY -c "import json,sys
try:
    d = json.load(open('$PROF/_raw/equiv_$1.json'))
except Exception:
    sys.exit(1)
sys.exit(0 if d.get('status') == 'PASS' else 1)"
}

for MPAIR in "FasterGS fastergs" "FasterGSFused fastergsfused" "FastGS p16_fastgs" "SADGS p16_sadgs"; do
  set -- $MPAIR; M=$1; ENV=$2
  if ! equiv_pass $M; then
    log "EQUIV_GATE_FAIL $M -> full replays skipped (profile not usable)"
    continue
  fi
  for SC in room bicycle garden; do
    [ -f $D/runs/$M/$SC/P16_RUN.json ] || { log "PROFILE_SKIP $M $SC (no run record)"; continue; }
    case $M in
      FasterGS|FasterGSFused) full_nerficg $M $ENV $SC ;;
      FastGS|SADGS)           full_graphdeco $M $ENV $SC ;;
    esac
  done
done

echo "PROFILES_ALL_DONE $(date -Is)" | tee -a $LOGS/run_profiles.log
