cd /home/liaoyuanjun/cgt_realscene
tar czf /tmp/cgt_final_sync.tar.gz -C /home/liaoyuanjun/cgt_realscene final
ls -la /tmp/cgt_final_sync.tar.gz
echo "=== file count in tar ==="
tar tzf /tmp/cgt_final_sync.tar.gz | wc -l
