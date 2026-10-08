echo "=== dualpath0 artifacts ==="
ls -la /root/epic05_dualpath0/artifacts/
echo "=== dualpath0 VERDICT ==="
cat /root/epic05_dualpath0/artifacts/*VERDICT*.json 2>/dev/null
echo "=== dualpath0 SUMMARY (head) ==="
head -c 2000 /root/epic05_dualpath0/artifacts/*SUMMARY*.json 2>/dev/null
echo
echo "=== dualpath0 prereg/handoff search ==="
find /root -maxdepth 3 -iname '*DUALPATH*' 2>/dev/null
find /root -maxdepth 3 -iname '*dual_path*' -o -maxdepth 3 -iname '*DUALPATH*PREREG*' 2>/dev/null | head
echo "=== local dualpath handoff in packet extracts ==="