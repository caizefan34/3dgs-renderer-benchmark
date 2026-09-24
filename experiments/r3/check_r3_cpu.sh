#!/bin/bash
# R3 process CPU check
ps -o pid,etime,time,pcpu,stat -p 328374 2>/dev/null
echo "sample1 done"
sleep 10
ps -o pid,etime,time,pcpu,stat -p 328374 2>/dev/null
echo "sample2 done"
