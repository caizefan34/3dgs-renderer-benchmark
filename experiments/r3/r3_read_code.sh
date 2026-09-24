#!/bin/bash
echo "=== locate runner ==="
F=$(find / -name r3_certificate_runner.py 2>/dev/null | head -1)
echo "FILE=$F"
if [ -n "$F" ]; then
    echo "=== grep key logic ==="
    grep -n "start_iter\|checkpoint\|29970\|30000" "$F" | head -50
fi
