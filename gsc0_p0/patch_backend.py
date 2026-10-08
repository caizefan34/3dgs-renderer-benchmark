import os, shutil, re
SRC = "/root/epic05_frozen/gsplat-true-accutile-v153/gsplat/cuda/_backend.py"
DST = "/root/gsc0_p0/_backend_patched.py"
os.makedirs("/root/gsc0_p0", exist_ok=True)
shutil.copy(SRC, DST)
txt = open(DST).read()
before = txt
# Fix the malformed flags line: `extra_cuda_cflags += ["-use_fast_math"], "--extended-lambda"`
txt = txt.replace(
    'extra_cuda_cflags += ["-use_fast_math"], "--extended-lambda"',
    'extra_cuda_cflags += ["-use_fast_math", "--extended-lambda"]')
assert txt != before, "flags line not found!"
open(DST, "w").write(txt)
# show the fixed line + surrounding
for i, ln in enumerate(txt.splitlines()):
    if "use_fast_math" in ln:
        print(f"L{i+1}: {ln}")
print("patched backend written to", DST)