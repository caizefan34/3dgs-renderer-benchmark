#!/bin/bash
# P16-RB: 5-point TTQ curve replays, SEQUENTIAL on physical A100 GPU 2.
#
# Each (method, scene) pair is a FRESH run of the frozen native recipe with exactly one
# change: the checkpoint schedule. nerficg pairs run from the patched copy tree
# $D/curve_src/nerficg (see curve_schedule.patch) with P16_BACKUP_ITERS listing the exact
# save iterations; graphdeco pairs append --save_iterations to their native recipe.
# The primary timed runs are never touched. Renders for graphdeco points are separate,
# untimed render.py invocations. Metric evaluation runs on GPU 3 (eval card).
#
# usage: run_curves.sh FasterGS:room FasterGSFused:room FastGS:room SADGS:room ...
set -u
D=/root/epic05_p16rb
GPU=2
EVALGPU=3
WRAP=$D/bundle/p16rb_wrap_run.py
GS=/root/miniforge3/envs/gsplat/bin/python
S1=$D/logs/curves.log
: > $S1
log() { echo "[$(date -Is)] $*" | tee -a $S1; }

# nerficg curve points sit this many iterations before the nominal fraction so a snapshot is
# never taken in the same iteration as reset_opacities (priority 90 runs before the save+render
# callback priority 1 -> a save at a 3000-multiple captures the just-reset transparent state;
# observed as PSNR 9.8/10.5 at 6000/12000 vs 31.2 at 18000).
NERF_OFFSET=${NERF_OFFSET:--30}

capture() {
  local L=$D/logs/curve_$1
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_before.csv
}
capture_after() {
  local L=$D/logs/curve_$1
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_after.csv
}

. $D/scripts/p16_recipes.sh

for PAIR in "$@"; do
  M=${PAIR%%:*}; SC=${PAIR##*:}
  ENV=$(native_env $M); N=$(native_n $M)
  PY=/root/miniforge3/envs/$ENV/bin/python
  MAT=$D/curves/$M/${SC}_materials
  mkdir -p $MAT
  PTS=$MAT/points.json
  log "$M $SC curve replay start (N=$N, env=$ENV)"
  capture ${M}_${SC}

  if [ "$M" = "FasterGS" ] || [ "$M" = "FasterGSFused" ]; then
    LOWER=$(echo $M | tr 'A-Z' 'a-z')
    CFG=$D/configs/p16_curve_${LOWER}_${SC}.yaml
    NERF_SRC=$D/curve_src/nerficg
    $PY $D/scripts/p16_curve_prepare.py --method $M --scene $SC --native-iterations $N \
      --offset $NERF_OFFSET \
      --stage-config $D/configs/p16_${LOWER}_${SC}_1920.yaml --out-config $CFG \
      --out-points $PTS > $MAT/prepare.log
    BACKITERS=$($PY -c "import json;print(','.join(str(p['iteration']) for p in json.load(open('$PTS'))['points'] if p['fraction']<1.0))")
    log "$M $SC curve schedule: offset=$NERF_OFFSET P16_BACKUP_ITERS=$BACKITERS"
    $PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/curve_${M}_${SC}.vram.csv \
      --duration 14400 > $D/logs/curve_${M}_${SC}.sampler.log 2>&1 &
    SAMP=$!
    $PY $WRAP --method $M --scene $SC --cwd $NERF_SRC \
      --command "P16_BACKUP_ITERS=$BACKITERS PATH=/root/miniforge3/envs/$ENV/bin:\$PATH python scripts/train.py -c $CFG" \
      --out $MAT/wrapper_train.json --log $MAT/train.stdout.log
    log "$M $SC train rc=$?"
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
    OUTD=$(ls -d $NERF_SRC/output/$M/p16_curve_${LOWER}_${SC}_* 2>/dev/null | tail -1)
    [ -z "$OUTD" ] && { log "FATAL: $M $SC no curve outdir"; echo "CURVES_${M}_${SC}_FAIL"; continue; }
    $PY -c "import json;[print(p['fraction'],p['iteration']) for p in json.load(open('$PTS'))['points']]" | \
    while read F IT; do
      R=$OUTD/test_$IT
      $GS $D/scripts/p16_name_map.py --method $M --scene $SC --rgb-dir $R/rgb --gt-dir $R/rgb_gt \
        --out $MAT/namemap_$IT.json > $MAT/namemap_$IT.log 2>&1
      CUDA_VISIBLE_DEVICES=$EVALGPU $GS $D/scripts/p16_common_metrics.py --method $M --scene $SC \
        --rgb-dir $R/rgb --name-map $MAT/namemap_$IT.json --out $MAT/metrics_$IT.json \
        > $MAT/metrics_$IT.log 2>&1
      log "$M $SC point $F@$IT metrics rc=$? ($(ls $R/rgb 2>/dev/null | wc -l) renders)"
    done
    $PY $D/scripts/p16_build_curve.py --method $M --scene $SC --points-json $PTS \
      --train-log $MAT/train.stdout.log --materials-dir $MAT --nerficg-outdir $OUTD \
      --out $D/curves/$M/$SC.json >> $MAT/curve_build.log 2>&1
    log "$M $SC curve rc=$? -> $(tail -c 300 $MAT/curve_build.log | tr '\n' ' ')"
  else
    JOB=p16_curve_${M}_${SC}
    $GS $D/scripts/p16_curve_prepare.py --method $M --scene $SC --native-iterations $N \
      --out-points $PTS > $MAT/prepare.log
    SAVE=$($PY -c "import json;print(' '.join(str(p['iteration']) for p in json.load(open('$PTS'))['points']))")
    $PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/curve_${M}_${SC}.vram.csv \
      --duration 14400 > $D/logs/curve_${M}_${SC}.sampler.log 2>&1 &
    SAMP=$!
    $PY $WRAP --method $M --scene $SC --cwd $D/src/$M \
      --command "CUDA_VISIBLE_DEVICES=$GPU OAR_JOB_ID=$JOB PATH=/root/miniforge3/envs/$ENV/bin:\$PATH python $(graphdeco_recipe $M $SC "$SAVE")" \
      --out $MAT/wrapper_train.json --log $MAT/train.stdout.log
    log "$M $SC train rc=$?"
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
    for IT in $SAVE; do
      $PY $WRAP --method $M --scene $SC --cwd $D/src/$M \
        --command "CUDA_VISIBLE_DEVICES=$GPU PATH=/root/miniforge3/envs/$ENV/bin:\$PATH python render.py -m output/$JOB --skip_train --iteration $IT" \
        --out $MAT/wrapper_render_$IT.json --log $MAT/render_$IT.log
      R=$D/src/$M/output/$JOB/test/ours_$IT
      $GS $D/scripts/p16_name_map.py --method $M --scene $SC --rgb-dir $R/renders --gt-dir $R/gt \
        --out $MAT/namemap_$IT.json > $MAT/namemap_$IT.log 2>&1
      CUDA_VISIBLE_DEVICES=$EVALGPU $GS $D/scripts/p16_common_metrics.py --method $M --scene $SC \
        --rgb-dir $R/renders --name-map $MAT/namemap_$IT.json --out $MAT/metrics_$IT.json \
        > $MAT/metrics_$IT.log 2>&1
      log "$M $SC point @$IT render+metrics rc=$?"
    done
    $PY $D/scripts/p16_build_curve.py --method $M --scene $SC --points-json $PTS \
      --train-log $MAT/train.stdout.log --materials-dir $MAT \
      --out $D/curves/$M/$SC.json >> $MAT/curve_build.log 2>&1
    log "$M $SC curve rc=$? -> $(tail -c 300 $MAT/curve_build.log | tr '\n' ' ')"
  fi
  capture_after ${M}_${SC}
  echo "CURVES_${M}_${SC}_DONE $(date -Is)" | tee -a $S1
done
echo "CURVES_ALL_DONE $(date -Is)" | tee -a $S1
