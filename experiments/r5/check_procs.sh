#!/bin/bash
ps aux | grep r4_train | grep -v grep | awk '{print $NF}'
echo "SEP"
ps aux | grep r4_train | grep -v grep | wc -l
