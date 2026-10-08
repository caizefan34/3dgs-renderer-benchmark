sed -n '150,200p' /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g1_compressibility.py
echo "=== records store fitted params? ==="
grep -n "fit_cgt\|params\|reshape(rank\|\"p\"\|'p'" /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g1_compressibility.py | head
