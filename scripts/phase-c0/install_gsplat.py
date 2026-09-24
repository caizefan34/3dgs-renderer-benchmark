import subprocess, os

# Install ninja
result = subprocess.run(["pip3", "install", "ninja"], capture_output=True, text=True, timeout=60)
print("ninja install:", result.stdout[-200:] if result.stdout else "", result.stderr[-200:] if result.stderr else "")

# Check the pre-compiled .so
so_path = "/home/liaoyuanjun/.cache/torch_extensions/py310_cu118/gsplat_cuda/gsplat_cuda.so"
print(f"\nPre-compiled .so exists: {os.path.exists(so_path)}")
if os.path.exists(so_path):
    print(f"  size: {os.path.getsize(so_path)} bytes")

# Check the build directory for gsplat_cuda
build_base = "/home/liaoyuanjun/.cache/torch_extensions/py310_cu118/gsplat_cuda"
if os.path.exists(build_base):
    for f in os.listdir(build_base):
        print(f"  {f}")

# Try to use the old _backend.py.bak approach - check if the old compiled .so 
# can be loaded directly
# First, let's check what the old backup expects
bak = "/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat/cuda/_backend.py.bak"
if os.path.exists(bak):
    with open(bak) as f:
        content = f.read()
    # Find the load_extension function
    if "load_extension" in content:
        idx = content.index("def load_extension")
        print(f"\nOld load_extension function:")
        print(content[idx:idx+800])
