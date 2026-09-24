#!/bin/bash
echo "PATH=$PATH"
echo "---"
echo "which python3:"
which python3
echo "---"
echo "Which Python has torch?"
/usr/bin/python3 -c "
import os, subprocess
path = os.environ.get('PATH', '')
dirs = path.split(':')
seen = set()
for d in dirs:
    p = d + '/python3'
    if os.path.exists(p) and p not in seen:
        seen.add(p)
        try:
            r = subprocess.run([p, '-c', 'import torch; print(torch.__file__, gsplat.__version__)'],
                             capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                print(f'FOUND: {p} -> {r.stdout}')
        except:
            pass
print('---')
print('Also checking known paths:')
for p in ['/home/liaoyuanjun/.local/bin/python3', '/usr/bin/python3', '/usr/local/bin/python3']:
    if os.path.exists(p):
        r = subprocess.run([p, '-c', 'import torch, gsplat; print(torch.__file__, gsplat.__version__)'],
                         capture_output=True, text=True, timeout=3)
        print(f'{p}: exit={r.returncode} stdout={r.stdout[:100]} stderr={r.stderr[:100]}')
"
