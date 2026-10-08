echo "=== g2_certificate.py teacher ==="
grep -n "tile_teacher\|px\[\|py\[\|dx = \|dy = \|def .*teacher\|def .*predict" /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g2_certificate.py | head -30
echo ""
echo "=== g2_rigorous_certificate.py teacher ==="
grep -n "tile_teacher\|px\[\|py\[\|dx = \|dy = \|def .*teacher\|def .*predict\|concatenate" /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g2_rigorous_certificate.py | head -30
