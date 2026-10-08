cd /home/liaoyuanjun/cgt_realscene
rm -f /tmp/cgt_final.tar.gz
tar czf /tmp/cgt_final.tar.gz final
ls -la /tmp/cgt_final.tar.gz
tar tzf /tmp/cgt_final.tar.gz | wc -l