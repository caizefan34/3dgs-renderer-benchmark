#!/bin/bash
# D0 local deploy orchestrator (runs on the Windows Git-Bash side).
# 1) regenerate instrumented sources from the frozen originals
# 2) LF-normalize every deployed file (CRLF landmine)
# 3) scp to EPIC-05 /root/epic05_d0/scripts
# 4) remote syntax checks (bash -n / ast) + prepare_d0.sh
# Usage: bash d0_deploy.sh [--skip-regen]
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
D0="$(cd "$HERE/.." && pwd)"          # .../d0_waste_census
INS="$D0/instrumented"
SCR="$D0/scripts"
HOST=EPIC-05
ROOT=/root/epic05_d0

FILES_PY_INS="d0_gaussian_model.py d0_trace.py d0_trainer.py patch_renderer.py"
FILES_RUN="prepare_d0.sh write_identity.sh gpu_gate.sh run_stage0c.sh run_stage0c_trace.sh run_stage0c_reps.sh run_stage0c_reps2.sh run_stage0c_repair.sh run_stage0c_rep3.sh run_stage12.sh run_stage12_gated.sh run_stage12_bicycle_clean_ext.sh analyze_stage0c.py analyze_stage12.py trace_spread_diagnostic.py probe_f9.py d0_selftest.py stage3_accounting.py stage4_predictability.py stage5_audit.py d0_verdict.py finalize_verdict.py collect_contamination.py assemble_package.py"

echo "== [1/4] regenerate instrumented sources"
if [ "${1:-}" != "--skip-regen" ]; then
  ( cd "$INS" && python make_sources.py )
else
  echo "   (skipped)"
fi

echo "== [2/4] LF-normalize + hash"
norm () {
  sed -i 's/\r$//' "$1"
  if file "$1" | grep -q CRLF; then echo "CRLF STILL PRESENT: $1"; exit 1; fi
  sha256sum "$1" | awk '{print $1"  "$2}'
}
for f in $FILES_PY_INS; do norm "$INS/$f"; done
for f in $FILES_RUN; do norm "$SCR/$f"; done

echo "== [3/4] remote dirs + scp"
ssh "$HOST" 'mkdir -p /root/epic05_d0/scripts /root/epic05_d0/runs /root/epic05_d0/artifacts /root/epic05_d0/logs /root/epic05_d0/src'
for f in $FILES_PY_INS; do scp -q "$INS/$f" "$HOST:$ROOT/scripts/"; done
for f in $FILES_RUN;    do scp -q "$SCR/$f" "$HOST:$ROOT/scripts/"; done
scp -q "$INS/D0_SOURCE_DIFF.diff" "$HOST:$ROOT/artifacts/D0_SOURCE_DIFF.diff"

echo "== [4/4] remote syntax checks"
ssh "$HOST" 'cd /root/epic05_d0/scripts && for f in prepare_d0.sh gpu_gate.sh run_stage0c.sh; do bash -n "$f" && echo "bash -n OK  $f"; done'
ssh "$HOST" 'cd /root/epic05_d0/scripts && /root/miniforge3/envs/gsplat/bin/python - <<PYEOF
import ast, sys
for f in ("d0_gaussian_model.py","d0_trace.py","d0_trainer.py","patch_renderer.py","analyze_stage0c.py","analyze_stage12.py","trace_spread_diagnostic.py","probe_f9.py","d0_selftest.py","stage3_accounting.py","stage4_predictability.py","stage5_audit.py","d0_verdict.py","finalize_verdict.py","collect_contamination.py","assemble_package.py"):
    ast.parse(open(f).read(), filename=f)
    print("ast OK     ", f)
PYEOF'

echo "== deploy complete. next: ssh $HOST 'bash $ROOT/scripts/prepare_d0.sh'"
