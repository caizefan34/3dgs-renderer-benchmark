#!/bin/bash
WT=/root/epic05_frozen/higs_c0_final30k_worktree
echo "=== find csrc-like dirs ==="
find $WT -maxdepth 3 -type d -name "*csrc*" 2>/dev/null
find $WT -maxdepth 2 -type d | head -20
echo
echo "=== find intersect kernel sources anywhere ==="
grep -rln "intersect_tile" $WT --include=*.cu --include=*.cuh --include=*.h 2>/dev/null | head -5
