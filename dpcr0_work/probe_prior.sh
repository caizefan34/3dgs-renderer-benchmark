echo "=== dirs of interest ==="
for d in madr0_p0 mr0_pkg mr0 dualpath0 epic05_dualpath0 _hm0_work _vos1_pub_harden cg0_run vb0_work; do
  if [ -d "/root/$d" ]; then echo "EXISTS /root/$d"; fi
done
echo "=== search for dpgr0 / direct_primitive / gather backward prototypes ==="
find /root -maxdepth 3 -iname '*dpgr0*' 2>/dev/null | head
find /root -maxdepth 3 -iname '*direct_primitive*' 2>/dev/null | head
find /root -maxdepth 2 -iname '*dualpath*' 2>/dev/null | head
echo "=== madr0_p0 contents ==="
ls -la /root/madr0_p0 2>/dev/null | head -30
echo "=== epic05_dualpath0 contents ==="
ls -la /root/epic05_dualpath0 2>/dev/null | head -30