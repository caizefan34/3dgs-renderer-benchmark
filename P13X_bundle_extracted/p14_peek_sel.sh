#!/bin/bash
ls -la /root/epic05_p14/raw/ 2>/dev/null
echo ---
grep -n "cell 0\|cell 1\|cell 2\|E(q)" /root/epic05_p14/logs/selector.log | tail -12
