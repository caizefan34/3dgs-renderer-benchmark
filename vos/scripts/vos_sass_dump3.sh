#!/bin/bash
CU=/usr/local/cuda/bin/cuobjdump
SO=/root/.cache/torch_extensions/py310_cu128/vos_microbench_ext/vos_microbench_ext.so
cd /tmp && rm -f vos_mb_*.cubin
$CU --list-elf "$SO" | head -5
$CU -xelf vos_mb "$SO" 2>/dev/null
# extract all cubins from this so into a temp dir to avoid clutter
mkdir -p /tmp/mb_cubins && cd /tmp/mb_cubins && rm -f *.cubin
$CU -xelf all "$SO" 2>&1 | head -3
ls *.cubin 2>/dev/null
for c in *.cubin; do
  echo "=== $c ==="
  /usr/local/cuda/bin/nvdisasm -c "$c" 2>/dev/null > "$c.sass" || $CU -sass "$c" > "$c.sass" 2>&1
  wc -l "$c.sass"
done
