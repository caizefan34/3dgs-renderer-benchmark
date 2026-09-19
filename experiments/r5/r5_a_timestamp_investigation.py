#!/usr/bin/env python3
"""Investigate discrepancy: check timestamps and file hashes."""
import json, os, hashlib
from pathlib import Path
from datetime import datetime

R4_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")

print("=" * 100)
print("File timestamps and hashes for train/truck candidate_c")
print("=" * 100)

for scene in ["train", "truck"]:
    for method in ["baseline", "candidate_c"]:
        run_dir = R4_BASE / scene / method
        
        print(f"\n--- {scene}/{method} ---")
        for fname in ["training_metrics.json", "provenance.json", "timing.json", "config.json"]:
            fpath = run_dir / fname
            if fpath.exists():
                stat = os.stat(fpath)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                with open(fpath, 'rb') as f:
                    sha = hashlib.sha256(f.read()).hexdigest()[:16]
                print(f"  {fname}: mtime={mtime}, sha256={sha}, size={stat.st_size}")
            else:
                print(f"  {fname}: NOT FOUND")
        
        # Check for checkpoint files
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.pt"))
            print(f"  checkpoints: {len(ckpts)} files")
            for c in sorted(ckpts):
                stat = os.stat(c)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"    {c.name}: mtime={mtime}, size={stat.st_size}")
        else:
            print(f"  checkpoints dir: NOT FOUND")

print("\n" + "=" * 100)
print("final_results.json timestamp")
print("=" * 100)
fr_path = R4_BASE / "final_results.json"
if fr_path.exists():
    stat = os.stat(fr_path)
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    with open(fr_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"  final_results.json: mtime={mtime}, sha256={sha}, size={stat.st_size}")

# Check log files timestamps
print("\n" + "=" * 100)
print("Log file timestamps")
print("=" * 100)
LOG_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2_logs")
for scene in ["train", "truck"]:
    for suffix in ["baseline", "candidate_c", "candidate"]:
        log_path = LOG_BASE / f"{scene}_{suffix}.log"
        if log_path.exists():
            stat = os.stat(log_path)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {log_path.name}: mtime={mtime}, size={stat.st_size}")
        else:
            print(f"  {log_path.name}: NOT FOUND")

# Check if there are other candidate log locations
print("\n--- Check home dir for candidate logs ---")
home = Path(os.path.expanduser("~"))
for scene in ["train", "truck"]:
    for pattern in [f"r4_{scene}_candidate*.log", f"r4_{scene}_candidate_c*.log"]:
        for p in home.glob(pattern):
            stat = os.stat(p)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {p}: mtime={mtime}, size={stat.st_size}")

# Now compare: what does the log say vs training_metrics.json?
print("\n" + "=" * 100)
print("Comparing log PSNR vs training_metrics.json PSNR for candidate_c")
print("=" * 100)

import re
def parse_log_for_eval(log_path):
    results = {}
    current_iter = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[ITER (\d+)\]', line)
            if m:
                current_iter = int(m.group(1))
            m = re.search(r'PSNR=([\d.]+)\s+SSIM=([\d.]+)\s+L1=([\d.]+)\s+N=(\d+)', line)
            if m and current_iter is not None:
                results[current_iter] = {
                    "psnr": float(m.group(1)),
                    "ssim": float(m.group(2)),
                    "N_gaussians": int(m.group(4)),
                }
    return results

for scene in ["train", "truck"]:
    print(f"\n--- {scene} candidate_c ---")
    
    # From training_metrics.json
    tm_path = R4_BASE / scene / "candidate_c" / "training_metrics.json"
    if tm_path.exists():
        tm = json.load(open(tm_path))
        c30k = tm.get("checkpoints", {}).get("30000", {})
        print(f"  training_metrics.json 30K: PSNR={c30k.get('psnr')}, SSIM={c30k.get('ssim')}, N={c30k.get('N_gaussians')}")
    
    # From log files
    for log_path in [
        LOG_BASE / f"{scene}_candidate_c.log",
        LOG_BASE / f"{scene}_candidate.log",
        home / f"r4_{scene}_candidate.log",
    ]:
        if log_path.exists():
            ckpts = parse_log_for_eval(log_path)
            if ckpts:
                max_iter = max(ckpts.keys())
                print(f"  {log_path.name} max_iter={max_iter}: PSNR={ckpts[max_iter]['psnr']:.6f}, SSIM={ckpts[max_iter]['ssim']:.6f}, N={ckpts[max_iter]['N_gaussians']}")
                if 30000 in ckpts:
                    print(f"  {log_path.name} 30K: PSNR={ckpts[30000]['psnr']:.6f}, SSIM={ckpts[30000]['ssim']:.6f}, N={ckpts[30000]['N_gaussians']}")
    
    # From final_results.json
    fr_path = R4_BASE / "final_results.json"
    if fr_path.exists():
        fr = json.load(open(fr_path))
        for item in fr:
            if item.get("scene") == scene:
                print(f"  final_results.json: C_PSNR={item.get('candidate_psnr')}, C_SSIM={item.get('candidate_ssim')}, C_N={item.get('candidate_n')}")
