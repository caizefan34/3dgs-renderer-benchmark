#!/bin/bash
# Dump SASS of the VOS microbench kernels and count loop-body instructions.
SO=$(ls -t /root/.cache/torch_extensions/*/vos_microbench_ext/vos_microbench_ext.so 2>/dev/null | head -1)
echo "SO=$SO"
if [ -z "$SO" ]; then
  find /root/.cache/torch_extensions -name "*.so" | head -10
  exit 1
fi
cuobjdump -sass "$SO" > /tmp/mb_sass.txt 2>&1
wc -l /tmp/mb_sass.txt
# extract the vos_adam_kernel function body
awk '/Function : .*vos_adam_kernel/,/^\s*$/' /tmp/mb_sass.txt > /tmp/vos_kernel_sass.txt
wc -l /tmp/vos_kernel_sass.txt
echo "=== BRA/LDG/MUFU counts in vos_adam_kernel ==="
grep -cE "BRA" /tmp/vos_kernel_sass.txt
grep -cE "LDG" /tmp/vos_kernel_sass.txt
grep -cE "MUFU" /tmp/vos_kernel_sass.txt
echo "=== dense kernel MUFU/LDG ==="
awk '/Function : .*dense_adam_kernel/,/^\s*$/' /tmp/mb_sass.txt > /tmp/dense_kernel_sass.txt
grep -cE "MUFU" /tmp/dense_kernel_sass.txt
echo "=== vos kernel full SASS (loop region) ==="
grep -nE "BRA|LDG|MUFU|FADD|FMUL|FFMA|MUFU.RCP|MUFU.SQRT" /tmp/vos_kernel_sass.txt | head -60
