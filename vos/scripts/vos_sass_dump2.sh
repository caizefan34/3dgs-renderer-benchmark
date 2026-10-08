#!/bin/bash
SO=/root/.cache/torch_extensions/py310_cu128/vos_microbench_ext/vos_microbench_ext.so
which cuobjdump || ls /usr/local/cuda/bin/cuobjdump 2>/dev/null || echo NO_CUOBJDUMP
cat /tmp/mb_sass.txt
echo "=== try nvdisasm via cubin extract ==="
cuobjdump --list-elf "$SO" 2>&1 | head -5
cuobjdump -xelf all "$SO" -d 2>&1 | head -5
ls *.cubin 2>/dev/null | head -5
