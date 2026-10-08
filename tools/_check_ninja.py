import os
p = os.path.expandvars(r'%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu130\gsplat_cuda\build.ninja')
lines = open(p).read().split('\n')
for l in lines:
    if 'FLAGS' in l or 'flags' in l or 'Zc:preprocessor' in l:
        if l.strip():
            print(l.strip()[:200])
