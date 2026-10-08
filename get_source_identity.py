#!/usr/bin/env python3
import sys
import subprocess
import json

# Get git commit
repo_path = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
result = subprocess.run(
    ["git", "-C", repo_path, "rev-parse", "HEAD"],
    capture_output=True, text=True
)
git_commit = result.stdout.strip()

# Get git status
result = subprocess.run(
    ["git", "-C", repo_path, "status", "--porcelain"],
    capture_output=True, text=True
)
git_status = result.stdout.strip() if result.stdout.strip() else "clean"

print(f"REPO_PATH={repo_path}")
print(f"GIT_COMMIT={git_commit}")
print(f"GIT_STATUS={git_status}")
