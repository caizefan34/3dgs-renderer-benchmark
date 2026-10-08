sed -n '150,200p' /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g1_compressibility.py
echo "=== how records store fitted params? (grep p=, params, reshape) ==="
grep -n "fit_cgt\|\"p\"\|'p'\|params\|reshape(rank" /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/g1_compressibility.py | head
