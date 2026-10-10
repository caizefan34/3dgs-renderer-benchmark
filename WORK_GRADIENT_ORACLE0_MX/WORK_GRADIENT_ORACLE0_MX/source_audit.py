#!/usr/bin/env python3
"""
Source identity audit for WORK-GRADIENT-ORACLE0-MX recovery.
Checks repository state and extracts benchmark file from git history.
"""
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr

def sha256_file(path):
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    repo_path = "/home/liaoyuanjun/3dgs-renderer-benchmark"
    os.chdir(repo_path)
    
    print("=== SOURCE IDENTITY AUDIT ===\n")
    
    # 1. Repository basic info
    print("1. REPOSITORY LOCATION")
    print(f"   Path: {repo_path}")
    
    # 2. Git identity
    print("\n2. GIT IDENTITY")
    rc, stdout, _ = run_cmd("git rev-parse HEAD")
    head_commit = stdout.strip()
    print(f"   HEAD: {head_commit}")
    
    rc, stdout, _ = run_cmd("git branch --show-current")
    current_branch = stdout.strip()
    print(f"   Branch: {current_branch}")
    
    rc, stdout, _ = run_cmd("git remote -v")
    remotes = stdout.strip()
    print(f"   Remotes: {remotes if remotes else 'NONE'}")
    
    # 3. Target commit
    target_commit = "32ab80e773f74f4d8e40ff4e338c86b29c7957f5"
    print(f"\n3. TARGET COMMIT AUDIT")
    print(f"   Target: {target_commit}")
    print(f"   Matches HEAD: {head_commit == target_commit}")
    
    # 4. Worktree status
    print(f"\n4. WORKTREE STATUS")
    rc, stdout, _ = run_cmd("git status --porcelain")
    modified_files = [line[3:] for line in stdout.strip().split('\n') if line]
    print(f"   Modified files: {len(modified_files)}")
    for f in modified_files[:10]:
        print(f"     - {f}")
    
    # 5. Benchmark file existence check
    print(f"\n5. BENCHMARK FILE EXISTENCE")
    benchmark_file = "benchmark/run_higs_train_benchmark.py"
    
    # Check worktree
    worktree_path = Path(repo_path) / benchmark_file
    in_worktree = worktree_path.exists()
    print(f"   In worktree: {in_worktree}")
    
    # Check HEAD
    rc, stdout, _ = run_cmd(f"git ls-tree HEAD {benchmark_file}")
    in_head = rc == 0 and stdout.strip()
    print(f"   In HEAD: {bool(in_head)}")
    
    # Check target commit
    rc, stdout, _ = run_cmd(f"git ls-tree {target_commit} {benchmark_file}")
    in_target = rc == 0 and stdout.strip()
    print(f"   In target commit {target_commit[:8]}: {bool(in_target)}")
    
    # Check history
    rc, stdout, _ = run_cmd(f"git log --all --oneline -- {benchmark_file}")
    history = stdout.strip().split('\n') if stdout.strip() else []
    print(f"   Commits touching file: {len(history)}")
    if history:
        print(f"     Latest: {history[0]}")
        print(f"     Oldest: {history[-1]}")
    
    # 6. Extract from history if not in worktree
    if not in_worktree and history:
        print(f"\n6. EXTRACTING FROM GIT HISTORY")
        latest_commit = history[0].split()[0]
        print(f"   Latest commit with file: {latest_commit}")
        
        # Get file size
        rc, stdout, _ = run_cmd(f"git show {latest_commit}:{benchmark_file} | wc -l")
        if rc == 0:
            line_count = stdout.strip()
            print(f"   File lines: {line_count}")
        
        # Extract to temp
        temp_output = f"/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/run_higs_train_benchmark.py"
        rc, stdout, stderr = run_cmd(
            f"git show {latest_commit}:{benchmark_file} > {temp_output}"
        )
        if rc == 0 and os.path.exists(temp_output):
            print(f"   Extracted to: {temp_output}")
            file_sha = sha256_file(temp_output)
            print(f"   SHA256: {file_sha}")
            
            # Verify it's the right file
            with open(temp_output) as f:
                first_lines = [f.readline().strip() for _ in range(5)]
                print(f"   First lines:")
                for line in first_lines:
                    print(f"     {line}")
        else:
            print(f"   Extraction failed: {stderr}")
    
    # 7. Current commit analysis
    print(f"\n7. CURRENT COMMIT ANALYSIS")
    rc, stdout, _ = run_cmd(f"git show --stat {target_commit} | head -20")
    print(f"   Commit {target_commit[:8]} changes:")
    for line in stdout.strip().split('\n')[:15]:
        print(f"     {line}")
    
    print("\n=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    main()
