"""C19-0 Step 1: Verify Environment on A100."""
import torch, gsplat, sys, os, json, subprocess
from datetime import datetime

env = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "host": os.uname().nodename if hasattr(os, 'uname') else "unknown",
    "python": sys.version.split()[0],
    "pytorch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}

# CUDA runtime
env["cuda_runtime"] = torch.version.cuda

# cuDNN
if torch.backends.cudnn.is_available():
    env["cudnn"] = torch.backends.cudnn.version()

# GPU
if torch.cuda.is_available():
    env["gpu_model"] = torch.cuda.get_device_name(0)
    env["gpu_count"] = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    env["compute_capability"] = f"{props.major}.{props.minor}"
    env["vram_total_mib"] = props.total_memory // (1024*1024)
    # Check PCIe info
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current,memory.total,driver_version",
                       "--format=csv,noheader"], capture_output=True, text=True)
    env["nvidia_smi"] = r.stdout.strip()

# gsplat
env["gsplat"] = gsplat.__version__

# nvcc
r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
env["nvcc"] = r.stdout.strip()

# gcc
r = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
env["gcc"] = r.stdout.split("\n")[0] if r.stdout else "unknown"

# Driver
r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True)
env["nvidia_driver"] = r.stdout.strip().split("\n")[0] if r.stdout else "unknown"

# pip
r = subprocess.run(["pip", "show", "gsplat"], capture_output=True, text=True)
for line in r.stdout.split("\n"):
    if line.lower().startswith("location"):
        env["gsplat_location"] = line.split(":", 1)[1].strip()
    if line.lower().startswith("requires"):
        env["gsplat_requires"] = line.split(":", 1)[1].strip()

# Check git state
try:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__))
    env["git_commit"] = r.stdout.strip()
    r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=os.path.dirname(__file__))
    env["git_dirty"] = len(r.stdout.strip()) > 0
except:
    pass

print(json.dumps(env, indent=2))

os.makedirs("reports/phase-c19", exist_ok=True)
os.makedirs("results/phase-c19", exist_ok=True)
with open("reports/phase-c19/current_environment_fingerprint.json", "w") as f:
    json.dump(env, f, indent=2)
print("Fingerprint saved.")
