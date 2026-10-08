#!/bin/bash
# P16-RB step 12: aggregate normalized run records -> artifacts/P16_BASELINE_TABLE.json
#
# The shipped bundle aggregator (bundle/p16rb_aggregate.py) scans --runs recursively and
# selects records by CONTENT (key-set match), so the garden_flatdata_corrupt records
# (retained verbatim as corruption evidence per artifacts/P16_DATA_HEAL_garden.json) would
# be double-counted and could shadow the healed rows. Per the heal record's verdict_note,
# flat-data records are corruption evidence, not results -> aggregate over a curated staging
# dir containing exactly the 15 canonical P16_RUN.json (5 methods x 3 scenes), with sha256
# recorded.
set -eu
D=/root/epic05_p16rb
AI=$D/aggregate_input_20260928
PY=/root/miniforge3/envs/p16_fastergs/bin/python

mkdir -p $AI
: > $AI/INPUTS.sha256
for M in P12_C0 FasterGS FasterGSFused FastGS SADGS; do
  for S in room bicycle garden; do
    SRC=$D/runs/$M/$S/P16_RUN.json
    [ -f "$SRC" ] || { echo "MISSING $SRC"; exit 1; }
    cp "$SRC" "$AI/${M}__${S}.json"
    sha256sum "$SRC" >> $AI/INPUTS.sha256
  done
done

$PY $D/bundle/p16rb_aggregate.py --runs $AI --out $D/artifacts/P16_BASELINE_TABLE.json
echo "TABLE_WRITTEN $D/artifacts/P16_BASELINE_TABLE.json"
cat $AI/INPUTS.sha256
