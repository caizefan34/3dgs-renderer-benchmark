#!/usr/bin/env python3
import os
import subprocess

F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
r = subprocess.run(["bash", "-c",
                    f"find {F30K} -type f -newermt '2026-09-23 00:00' -printf '%T+ %p\\n' | sort"],
                   capture_output=True, text=True)
print("== files newer than 2026-09-23 ==")
print(r.stdout)
dirs = sorted(d for d in os.listdir(F30K)
              if os.path.isdir(f"{F30K}/{d}") and os.path.exists(f"{F30K}/{d}/results.json"))
print(f"== {len(dirs)} run dirs with results.json ==")
for d in dirs:
    print(" ", d)
