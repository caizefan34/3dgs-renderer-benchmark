FIN=/home/liaoyuanjun/cgt_realscene/final
cd $FIN
echo "=== MANIFEST format (head) ==="
head -5 MANIFEST.sha256
echo "=== MANIFEST line count ==="
wc -l MANIFEST.sha256
echo "=== is it relative paths? ==="
awk "{print \$2}" MANIFEST.sha256 | head -5
