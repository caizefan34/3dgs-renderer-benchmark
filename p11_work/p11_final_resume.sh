#!/bin/bash
# Run remaining P11 stages sequentially (T, O, closure, assemble).
cd /root/epic05_p11
for st in T O closure assemble; do
  echo ">>> final-resume stage $st"
  bash p11_run_all.sh "$st" || { echo "STAGE $st FAILED"; exit 1; }
done
echo ">>> final-resume complete"
