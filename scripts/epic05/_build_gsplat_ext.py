"""Build gsplat CUDA extension with proper CUDA_PATH and MSVC PATH."""
import os
import sys

# Strip any trailing spaces from CUDA_PATH
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3".strip()
os.environ["CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DISTUTILS_USE_SDK"] = "1"

# Make sure cl.exe is on PATH (needed by ninja for .cpp compilation)
_msvc_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
_extra_paths = [_msvc_dir, _cuda_bin]
current_path = os.environ.get("PATH", "")
for p in _extra_paths:
    if p not in current_path:
        current_path = p + os.pathsep + current_path
os.environ["PATH"] = current_path

# Change to gsplat source dir
gsplat_root = r"C:\Users\36570\Documents\Codex\3dgs-renderer-benchmark\artifacts\renderer-sources\gsplat"
os.chdir(gsplat_root)

# Run setup.py via subprocess so __file__ is correct
sys.argv = ["setup.py", "build_ext"]
with open(os.path.join(gsplat_root, "setup.py"), "rb") as f:
    code = f.read()
# Use globals with __file__ set correctly
globals_dict = {"__file__": os.path.join(gsplat_root, "setup.py"), "__name__": "__main__"}
exec(compile(code, os.path.join(gsplat_root, "setup.py"), "exec"), globals_dict)
