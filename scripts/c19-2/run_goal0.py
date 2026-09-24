"""
Run Goal 0: reproducibility baseline on A100.
GPU allocation: GPU 0 (primary), GPU 7 (validation)
"""
import torch, json, os, sys, math, numpy as np, subprocess, threading, time

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.join(repo, "scripts", "c19-2")
log_dir = os.path.join(repo, "logs", "c19-2")
os.makedirs(log_dir, exist_ok=True)

print(f"Repo: {repo}")
print(f"Scripts: {script_dir}")
print(f"Log dir: {log_dir}")

# Run 00_reproducibility_baseline.py on GPU 0 and GPU 7 in parallel
scripts = [
    (0, "00_reproducibility_baseline.py", os.path.join(log_dir, "goal0_gpu0.log")),
    (7, "00_reproducibility_baseline.py", os.path.join(log_dir, "goal0_gpu7_validation.log")),
]

processes = []
for gpu, script, logf in scripts:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    p = subprocess.Popen(
        [sys.executable, os.path.join(script_dir, script)],
        env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT
    )
    processes.append(p)
    print(f"Started {script} on GPU {gpu} (PID {p.pid}), log: {logf}")

for p in processes:
    p.wait()

print("Goal 0 complete. Checking outputs...")
os.system(f"ls -la {os.path.join(repo, 'results', 'phase-c19', 'c19-2_reproducibility.json')} 2>/dev/null && echo 'Found!' || echo 'NOT FOUND'")
