#!/bin/bash
cd /root/epic05_p13x
nohup bash launch_bicycle.sh > /dev/null 2>&1 < /dev/null &
echo "BICYCLE_LAUNCHED"
