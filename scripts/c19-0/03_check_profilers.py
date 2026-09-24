"""Check available profiling tools on this machine."""
import subprocess, shutil, os

# Check ncu (Nsight Compute)
ncu_path = shutil.which("ncu")
print(f"ncu: {ncu_path if ncu_path else 'NOT FOUND'}")

# Check nsys (Nsight Systems)
nsys_path = shutil.which("nsys")
print(f"nsys: {nsys_path if nsys_path else 'NOT FOUND'}")

# Check CUPTI
for path in ["/usr/local/cuda/extras/CUPTI/lib64/libcupti.so",
             "/usr/local/cuda-11/extras/CUPTI/lib64/libcupti.so",
             "/usr/local/cuda-12/extras/CUPTI/lib64/libcupti.so"]:
    print(f"CUPTI at {path}: {'EXISTS' if os.path.exists(path) else 'not found'}")

# Check nvidia-smi
r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True)
print("\nGPU info:")
print(r.stdout if r.stdout else r.stderr)

# Check CUDA version
r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print("\nNVCC:")
print(r.stdout if r.stdout else r.stderr)
