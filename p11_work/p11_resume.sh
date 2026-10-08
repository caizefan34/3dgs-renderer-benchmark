#!/bin/bash
# Resume the P11 pipeline from a given stage (states already generated).
# Stages: L1 L2 L3 L4 D1 T O assemble
cd /root/epic05_p11
FROM=${1:-L1}
ORDER="L1 L2 L3 L4 D1 T O assemble"
STARTED=0
for st in $ORDER; do
  if [ "$st" = "$FROM" ]; then STARTED=1; fi
  if [ $STARTED -eq 1 ]; then
    echo ">>> resume running stage $st"
    bash p11_run_all.sh "$st" || { echo "STAGE $st FAILED"; exit 1; }
  fi
done
echo ">>> resume complete"
