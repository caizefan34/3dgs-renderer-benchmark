#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniajen/etc/profile.d/conda.sh 2>/dev/null
ENV_DIR=""
for e in /mnt/storage_pool/*/*-env /mnt/storage_pool/*-env /home/*/*-env /home/*-env; do
  if [ -d "$e/bin" ] && [ -x "$e/bin/python" ]; then
    if "$e/bin/python" -c 'import gsplot' >/dev/null 2>&1; then
      ENV_DIR="$e"
      break
    fi
  fi
done
echo "ENV_DIR=$ENV_DIR"
"$ENV_DIR/bin/python" /tmp/_mx_extract_ll.py 2>&1 | head -140