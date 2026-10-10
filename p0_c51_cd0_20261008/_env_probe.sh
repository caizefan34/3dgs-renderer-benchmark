#!/bin/bash
set +e
echo "=== whoami ==="; whoami; id
echo "=== ls -l the gsplat env python ==="
ls -l /root/miniforge3/envs/gsplat/bin/python 2>&1
echo "=== can liaoyuanjun sudo? ==="
sudo -n true 2>&1 && echo "SUDO_OK" || echo "NO_PASSWORDLESS_SUDO"
echo "=== conda envs visible to this user ==="
conda env list 2>/dev/null || echo "no conda on PATH"
ls -d /home/liaoyuanjun/*envs* /home/liaoyuanjun/miniforge3/envs/* /home/liaoyuanjun/.conda/envs/* 2>/dev/null
echo "=== any python with torch+CUDA under home ==="
find /home/liaoyuanjun -maxdepth 5 -path "*bin/python" 2>/dev/null | while read p; do
  v=$("$p" -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>/dev/null)
  [ -n "$v" ] && echo "$p -> $v"
done | head -20
echo "=== venvs in repo ==="
ls -d /home/liaoyuanjun/3dgs-renderer-benchmark/.venv* /home/liaoyuanjun/3dgs-renderer-benchmark/venv* 2>/dev/null
echo "=== gsplat import under home pythons ==="
find /home/liaoyuanjun -maxdepth 6 -name "gsplat" -type d 2>/dev/null | head -5
echo "DONE"
