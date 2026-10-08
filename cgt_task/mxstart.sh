cd /home/liaoyuanjun/cgt_realscene
setsid bash launch_all.sh > logs/g1/launch_all.out 2>&1 < /dev/null &
sleep 1
echo "detached launcher started"