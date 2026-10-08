#!/bin/bash
S=/tmp/mb_cubins/vos_microbench_ext.1.sm_80.cubin.sass
echo "=== vos_adam_kernel loop body ==="
awk '/vos_adam_kernel/,0' "$S" | sed -n '1,120p' | grep -E "LDG|MUFU|FMUL|FADD|FFMA|BRA|ISETP|FSEL|IMAD|F2F|MOV|ISETP" | head -50
echo ""
echo "=== instruction histogram vos_adam_kernel ==="
awk '/vos_adam_kernel/,0' "$S" | grep -oE "^\s+/\*[0-9a-f]+\*/\s+[A-Z0-9@!.]+" | awk '{print $NF}' | sort | uniq -c | sort -rn | head -20
echo ""
echo "=== dense_adam_kernel section ==="
awk '/dense_adam_kernel/,/vos_adam_kernel/' "$S" | grep -cE "MUFU"
