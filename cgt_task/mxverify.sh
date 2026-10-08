cd /home/liaoyuanjun/cgt_realscene/final
echo "=== REPORT: grep key lines ==="
grep -c "q_w\|S(γ\|PASS\|HEADROOM\|Amdahl" CGT_MASTER_REPORT.md
echo "=== REPORT last 3 lines ==="
tail -3 CGT_MASTER_REPORT.md
echo "=== METHOD: grep sections ==="
grep -c "^##" METHOD_NOTES.md
echo "=== METHOD last 2 lines ==="
tail -2 METHOD_NOTES.md
echo "=== any stray heredoc delimiters? ==="
grep -n "RPTEOF\|MNEOf\|EOF" CGT_MASTER_REPORT.md METHOD_NOTES.md || echo "none"