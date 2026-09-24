#!/usr/bin/env python3
"""Probe remote env: torch/nvcc/gpu + gsplat bootstrapping for the P3-H build."""
import sys, subprocess, os, shutil, tempfile

def main():
    print("python_exe:", sys.executable)
    import torch
    print("torch:", torch.__version__)
    print("cuda_avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    ess = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print("nvcc:", ((ess.stdout or ess.stderr).strip().splitlines()[-1] if (ess.stdout or ess.stderr) else "MISSING"))

    work = "/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat"
    boot = tempfile.mkdtemp(prefix="p3h_boot_")
    os.symlink(work, os.path.join(boot, "gsplat"))
    sys.path.insert(0, boot)
    try:
        import gsplat
        print("gsplat.__file__:", gsplat.__file__)
        import gsplat.cuda._wrapper
        import gsplat.rendering
        print("gsplat.cuda._wrapper OK")
        from gsplat.experimental.render.functional import gaussian_inference as gi
        print("gaussian_inference:", gi.__file__)
        print("has _native_forward_capture:",
              hasattr(gi._HigsAutogradFunction, "_native_forward_capture"))
        _ = gi._cull_gaussians_batched
        print("has _cull_gaussians_batched: True")
        _ = gi._gather_visible_native
        print("has _gather_visible_native: True")
    except Exception as e:
        print("GSPLAT_IMPORT_ERR:", type(e).__name__, e)

if __name__ == "__main__":
    main()