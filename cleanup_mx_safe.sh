#!/bin/bash
# Safe cleanup script for MX - removes only clearly temporary files

echo "Starting safe cleanup on MX..."
echo "================================"
echo ""

# Track space freed
TOTAL_FREED=0

# 1. Remove old F9 checkpoint files from /tmp (September 21)
echo "Removing old F9 checkpoint files from /tmp..."
CKPT_SIZE=$(ssh mx "du -sb /tmp/f9-1r-*.pt 2>/dev/null | awk '{s+=\$1} END {print s}'" 2>/dev/null)
if [ ! -z "$CKPT_SIZE" ]; then
    ssh mx "rm -f /tmp/f9-1r-*.pt"
    TOTAL_FREED=$((TOTAL_FREED + CKPT_SIZE))
    echo "  Freed: ~$((CKPT_SIZE / 1024 / 1024)) MB"
fi

# 2. Remove old deliverables tarballs from /tmp (September 15)
echo "Removing old deliverables tarballs from /tmp..."
TARBALL_SIZE=$(ssh mx "du -sb /tmp/n2_cpcb_deliverables*.tar.gz 2>/dev/null | awk '{s+=\$1} END {print s}'" 2>/dev/null)
if [ ! -z "$TARBALL_SIZE" ]; then
    ssh mx "rm -f /tmp/n2_cpcb_deliverables*.tar.gz"
    TOTAL_FREED=$((TOTAL_FREED + TARBALL_SIZE))
    echo "  Freed: ~$((TARBALL_SIZE / 1024 / 1024)) MB"
fi

# 3. Remove old temporary project directories from /tmp
echo "Removing old temporary project directories from /tmp..."
for dir in "/tmp/3dgs-renderer-benchmark" "/tmp/accutile_a100_results" "/tmp/gsplat_n2_cpcb" "/tmp/h1_b2_authoritative" "/tmp/higs_h2_bwd_cf"; do
    DIR_SIZE=$(ssh mx "du -sb $dir 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
    if [ ! -z "$DIR_SIZE" ]; then
        ssh mx "rm -rf $dir"
        TOTAL_FREED=$((TOTAL_FREED + DIR_SIZE))
        echo "  Removed: $dir (~$((DIR_SIZE / 1024 / 1024)) MB)"
    fi
done

# 4. Remove other old tarballs from /tmp
echo "Removing other old tarballs from /tmp..."
for tar in "/tmp/gsplat_c17_0_build" "/tmp/c0e01room.tar" "/tmp/figtables.tar"; do
    TAR_SIZE=$(ssh mx "du -sb $tar 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
    if [ ! -z "$TAR_SIZE" ]; then
        ssh mx "rm -rf $tar"
        TOTAL_FREED=$((TOTAL_FREED + TAR_SIZE))
        echo "  Removed: $tar (~$((TAR_SIZE / 1024 / 1024)) MB)"
    fi
done

echo ""
echo "================================"
echo "Total space freed: ~$((TOTAL_FREED / 1024 / 1024)) MB ($((TOTAL_FREED / 1024 / 1024 / 1024)) GB)"
echo ""
echo "Checking disk space after cleanup..."
ssh mx "df -h / /mnt/storage_pool"
echo ""
echo "Safe cleanup complete."
echo ""
echo "Additional large directories that may need attention:"
echo "  - /home/liaoyuanjun/projects/AnySplat/output: 46 GB"
echo "  - /home/liaoyuanjun/miniforge3: 81 GB"
echo "  - /home/liaoyuanjun/projects/AnySplat_mv: 31 GB"
