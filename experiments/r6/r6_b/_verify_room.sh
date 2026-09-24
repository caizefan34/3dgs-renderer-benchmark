#!/bin/bash
# Verify room 30K correctness result
DIR=/mnt/storage_pool/liaoyuanjun/r6_profiling
echo "=== Files in $DIR matching r6_b_correctness ==="
ls -la "$DIR"/r6_b_correctness_*.json 2>&1
echo ""
echo "=== Room JSON content ==="
cat "$DIR/r6_b_correctness_room_30k.json" 2>&1
echo ""
echo "=== Room log tail ==="
tail -30 "$DIR/r6_b_correctness_room_30k.log" 2>&1
