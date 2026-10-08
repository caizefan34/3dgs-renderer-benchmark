#!/bin/bash
cd /home/liaoyuanjun/cgt_realscene
mkdir -p logs/g1
export PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
# start 3 detached jobs, record PIDs
setsid bash run_g1.sh 5 "room:5000:r5k,room:10000:r10k" > logs/g1/gpu5.out 2>&1 < /dev/null &
echo $! > logs/g1/gpu5.pid
setsid bash run_g1.sh 6 "room:15000:r15k,room:30000:r30k" > logs/g1/gpu6.out 2>&1 < /dev/null &
echo $! > logs/g1/gpu6.pid
setsid bash run_g1.sh 7 "garden:30000:garden,bicycle:30000:bicycle" > logs/g1/gpu7.out 2>&1 < /dev/null &
echo $! > logs/g1/gpu7.pid
sleep 2
echo "PIDs: $(cat logs/g1/gpu5.pid) $(cat logs/g1/gpu6.pid) $(cat logs/g1/gpu7.pid)"
