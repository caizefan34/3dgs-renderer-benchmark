#!/bin/bash
set -e
pkill -f p14_o1.py 2>/dev/null || true
sleep 2
rm -f /root/epic05_p14/P14_O1_UNITS/*.json
rm -f /root/epic05_p14/P14_O2_UNITS/*.json
rm -f /root/epic05_p14/P14_O1_UNITS_SMOKE/*.json
rm -f /root/epic05_p14/P14_O2_UNITS_SMOKE/*.json
rm -f /root/epic05_p14/artifacts/P14_O1_AGG.json
rm -f /root/epic05_p14/artifacts/P14_O2_AGG.json
echo "cleaned all O1/O2 units (stream-labeling bug fix requires full re-run)"
ls /root/epic05_p14/P14_O1_UNITS/ /root/epic05_p14/P14_O2_UNITS/ 2>/dev/null
