echo "=== assemble_g1.sh / run_g1.sh (analyze invocation) ==="
grep -n "analyze_g1\|--inputs\|--out" /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/assemble_g1.sh /home/liaoyuanjun/cgt_realscene/final/SCRIPTS/run_g1.sh 2>/dev/null | head
echo "=== local assemble_g1.sh ==="
Get-Content assemble_g1.sh -ErrorAction SilentlyContinue | Select-Object -First 40
