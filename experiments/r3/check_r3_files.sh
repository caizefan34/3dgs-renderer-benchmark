#!/bin/bash
# Check for output JSON files
REPO=$(ls -d /home/*/3dgs-renderer-benchmark 2>/dev/null | head -1)
find "$REPO/results/reference_v1/r3" -type f 2>/dev/null | head -20
echo "=== json count ==="
find "$REPO/results/reference_v1/r3" -name "*.json" 2>/dev/null | wc -l
