#!/usr/bin/env python3
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153/setup.py"
with open(path) as f:
    content = f.read()
bad = 'nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"], "--extended-lambda"'
good = 'nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17", "--extended-lambda"]'
if bad in content:
    content = content.replace(bad, good, 1)
    with open(path, "w") as f:
        f.write(content)
    print("FIXED")
else:
    print("NOT FOUND - checking if already correct")
    if good in content:
        print("Already correct")
    else:
        print("UNEXPECTED STATE")
