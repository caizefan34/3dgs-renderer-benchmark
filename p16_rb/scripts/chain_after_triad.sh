#!/bin/bash
# Wait for the P12 C0 triad marker, then run the common-split render/eval.
set -u
D=/root/epic05_p16rb
while ! grep -q P12_C0_TRIAD_DONE "$D/logs/P12_C0_triad.log" 2>/dev/null; do sleep 20; done
sleep 5
bash "$D/scripts/run_p12_common_eval.sh" >> "$D/logs/P12_C0_common_eval.log" 2>&1
echo "CHAIN_DONE $(date -Is)" >> "$D/logs/P12_C0_common_eval.log"
