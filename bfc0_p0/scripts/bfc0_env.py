#!/usr/bin/env python3
"""BFC0 DSH-0: Environment capture."""
import json, os, sys, platform, subprocess
import torch

def get_git_sha(repo_dir):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
    except:
        return "UNKNOWN"

def is_git_dirty(repo_dir):
    try:
        out = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        return len(out) > 0
    except:
        return False

def capture_env(repo_dir):
    import gsplat
    env = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
        "gsplat": gsplat.__version__,
        "numpy": __import__('numpy').__version__,
        "git_sha": get_git_sha(repo_dir),
        "git_dirty": is_git_dirty(repo_dir),
        "repo_dir": repo_dir,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
        env["gpu_multi_processor_count"] = props.multi_processor_count
    return env

if __name__ == "__main__":
    repo = os.environ.get("BFC0_REPO_DIR", "C:\\Users\\36570\\3dgs-renderer-benchmark")
    env = capture_env(repo)
    out_path = os.environ.get("BFC0_OUT_DIR", "C:\\Users\\36570\\3dgs-renderer-benchmark\\bfc0_p0\\output")
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "BFC0_ENVIRONMENT.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))
