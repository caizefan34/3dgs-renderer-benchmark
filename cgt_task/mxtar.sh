cd /home/liaoyuanjun/cgt_realscene
tar czf /tmp/cgt_final.tar.gz final
ls -la /tmp/cgt_final.tar.gz
echo "=== tar contents count ==="
tar tzf /tmp/cgt_final.tar.gz | wc -l