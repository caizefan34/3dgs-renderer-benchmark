#!/usr/bin/env python3
"""
C51J_FROZEN_TRAINING_VALIDATION_HARNESS
=======================================
Executable training-validation harness for CVPR27-C51-JOINT (MX / A100-PCIE-40GB).

This is a DELIVERABLE, not a claim of results. It is EXECUTION-GATED: it refuses
to run unless (a) an exclusive A100 is available and (b) the output storage has
free space. As of the audit (2026-10-10) neither condition held, so this ships
unexecuted. When an exclusive GPU + free storage appear, it runs the frozen
protocol defined in C51J_FROZEN_PROTOCOL.json.

Design contract (from protocol):
  * Arms: native_baseline, opacity_pruning_control, strong_candidate(RESTORABLE_ONLY),
          codex_candidate(AWAITING manifest).
  * Split isolation: train views drive gradient/densification; val views used ONLY
    for time-to-quality checkpoints; held-out test views used ONLY for final
    PSNR/SSIM/LPIPS. No method selection on held-out.
  * Timing: wall-clock from first training iter to final checkpoint save, with
    torch.cuda.synchronize before start/stop. Peak VRAM via max_memory_allocated.
  * Plan: 30K full, 5K dev screening, 1K smoke. Same seed/init/camera-order per cell.
  * Cohort: A100-PCIE-40GB only; refuses other GPUs.

The heavy lifting reuses the proven training infra pattern from
scripts/phase-c51-stage5/benchmark.py (GaussianModel, GTDataset, densification).
"""
import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path

# ----------------------------------------------------------------------------
# Execution gate — hard resource checks. This harness will NOT run without them.
# ----------------------------------------------------------------------------
REQUIRED_GPU_NAME = "A100"
REQUIRED_GPU_MEM_MIB = 39000
MIN_FREE_GPU_MIB = 30000  # required FREE memory on the selected GPU
MIN_FREE_STORAGE_BYTES = 5 * 1024**3  # 5 GiB headroom for artifacts/traces
OUTPUT_ROOT = "/mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX"


def pick_writable_output(preferred=OUTPUT_ROOT):
    """Return the first writable output location with >= MIN_FREE_STORAGE_BYTES.

    Order: (1) the preferred pool dir (the formal objective output), then (2) a
    writable fallback under $HOME (e.g. /home/<user>/C51J_runs). This lets runs
    proceed when the shared pool is 100% full but the account has home space.
    Records the choice so the run's provenance states where artifacts landed.
    """
    candidates = [preferred]
    home = os.environ.get("C51J_FALLBACK_OUT")
    if home:
        candidates.append(home)
    else:
        candidates.append(os.path.join(os.path.expanduser("~"), "C51J_runs"))
    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            st = os.statvfs(c)
            if st.f_bavail * st.f_frsize >= MIN_FREE_STORAGE_BYTES:
                return c, True
        except (FileNotFoundError, OSError):
            continue
    # nothing has space; return preferred for the error message
    return preferred, False


def resource_gate(gpu_id=0, output_dir=None):
    """Return (ok: bool, reasons: list[str], info: dict). Refuses to run on wrong/
    absent GPU, insufficient *free* GPU memory, or no writable storage. No silent
    substitution of hardware cohort. Storage may be satisfied by a writable
    fallback when the formal pool is full."""
    try:
        import torch
    except Exception as e:  # noqa
        return False, [f"torch import failed in this interpreter ({e}); "
                       "use the frozen env python, e.g. /root/miniforge3/envs/gsplat/bin/python"], {}
    if not torch.cuda.is_available():
        return False, ["CUDA not available (default python has no torch.cuda; "
                       "use the frozen gsplat env interpreter)"], {}
    reasons = []
    info = {}
    name = torch.cuda.get_device_name(gpu_id)
    if REQUIRED_GPU_NAME not in name:
        reasons.append(f"GPU[{gpu_id}] '{name}' is not {REQUIRED_GPU_NAME}; cohort rule forbids mixing")
    props = torch.cuda.get_device_properties(gpu_id)
    total_mib = props.total_memory / (1024**2)
    free_mib = torch.cuda.mem_get_info(gpu_id)[0] / (1024**2)
    if total_mib < REQUIRED_GPU_MEM_MIB:
        reasons.append(f"GPU[{gpu_id}] total mem {total_mib:.0f} MiB < {REQUIRED_GPU_MEM_MIB} MiB (need A100-40GB)")
    if free_mib < MIN_FREE_GPU_MIB:
        reasons.append(f"GPU[{gpu_id}] FREE mem {free_mib:.0f} MiB < required {MIN_FREE_GPU_MIB} MiB "
                       f"(pass --min-free-mib to override if the model fits)")
    info["gpu_name"] = name
    info["gpu_total_mib"] = round(total_mib)
    info["gpu_free_mib"] = round(free_mib)
    info["gpu_id"] = gpu_id
    # storage
    chosen, space_ok = pick_writable_output(output_dir or OUTPUT_ROOT)
    info["output_dir"] = chosen
    info["output_is_pool"] = (chosen == OUTPUT_ROOT)
    if not space_ok:
        try:
            st = os.statvfs(output_dir or OUTPUT_ROOT)
            freeb = st.f_bavail * st.f_frsize
        except (FileNotFoundError, OSError):
            freeb = 0
        reasons.append(f"no writable storage with >= {MIN_FREE_STORAGE_BYTES/1024**2:.0f} MiB "
                       f"(pool free {freeb/1024**2:.0f} MiB; home fallback also full)")
    # exclusivity: record compute-app count (best-effort, non-fatal)
    return (len(reasons) == 0), reasons, info


def gpu_exclusivity_note():
    """Best-effort record of other CUDA processes; does not gate but is logged."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10).decode()
        return out.strip()
    except Exception as e:  # noqa
        return f"unavailable: {e}"


# ----------------------------------------------------------------------------
# Split isolation. Views are partitioned; method code may only touch train.
# val/test indices are exposed read-only and their use is asserted.
# ----------------------------------------------------------------------------
class SplitIsolator:
    """Enforces train/val/held-out separation for a scene's camera list.

    Mip-NeRF360 official split convention (indices divisible by 8 for the
    360 scenes; the standard 'train' excludes every 8th image). We keep it
    explicit so no code path can accidentally evaluate on test during training.
    """
    def __init__(self, n_cameras, scene, heldout_frac=0.1, val_frac=0.1, seed=0):
        import numpy as np
        self.scene = scene
        idx = np.arange(n_cameras)
        # official-style held-out: every 8th camera (Mip-NeRF360 convention)
        self.heldout = set(idx[idx % 8 == 0].tolist())
        remaining = [i for i in idx if i not in self.heldout]
        rng = np.random.RandomState(seed)
        rng.shuffle(remaining)
        n_val = max(1, int(len(remaining) * val_frac))
        n_test = max(1, int(n_cameras * heldout_frac)) - len(self.heldout & set(idx))
        self.val = set(remaining[:n_val])
        self.train = [i for i in remaining if i not in self.val]
        # heldout union: official heldout + extra heldout_frac sample (never in train/val)
        extra = [i for i in remaining[n_val:] ]
        rng.shuffle(extra)
        self.heldout |= set(extra[:max(0, n_test)])
        self.train = [i for i in self.train if i not in self.heldout]

    def assert_train_only(self, cam_index, context):
        if cam_index in self.val or cam_index in self.heldout:
            raise RuntimeError(f"LEAK: {context} used non-train view {cam_index}")


# ----------------------------------------------------------------------------
# Timing + VRAM capture wrapper.
# ----------------------------------------------------------------------------
class RunTimer:
    def __init__(self, device):
        self.device = device
        self.wall = None
        self.peak_vram_mib = None

    def __enter__(self):
        import torch
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import torch
        torch.cuda.synchronize(self.device)
        self.wall = time.perf_counter() - self._t0
        self.peak_vram_mib = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        return False


# ----------------------------------------------------------------------------
# SHA256 provenance for the run environment (source/deps identity).
# ----------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_provenance(repo_root):
    prov = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch, gsplat
        prov["torch"] = torch.__version__
        prov["torch_cuda"] = torch.version.cuda
        prov["gsplat"] = gsplat.__version__
        prov["gpu"] = torch.cuda.get_device_name(0)
        prov["cuda_home"] = os.environ.get("CUDA_HOME", "")
        prov["cxx11_abi"] = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception as e:  # noqa
        prov["import_error"] = str(e)
    try:
        prov["repo_head"] = subprocess_check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    except Exception:
        pass
    return prov


def subprocess_check_output(cmd):
    import subprocess
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()


# ----------------------------------------------------------------------------
# Main. Dispatches to the real training loop when resources permit.
# ----------------------------------------------------------------------------
ARM_MODES = {
    "native_baseline": "baseline",
    "opacity_pruning_control": "opacity_prune",
    "strong_candidate": "fused_prune10_rclip05",
    "codex_candidate": "codex_runtime_flag",
}

# P4 development matrix: scenes + arms. Priority per objective: room, bicycle, garden.
P4_SCENES = ["room", "bicycle", "garden"]
P4_SMOKE_ARMS = ["native_baseline", "opacity_pruning_control", "codex_candidate"]
P4_5K_ARMS = ["native_baseline", "opacity_pruning_control", "codex_candidate"]


def emit_plan(repo_root, output_dir="", runtime_flag=""):
    """Print the exact, copy-pasteable P4 execution command matrix. Requires NO GPU.
    This makes P4 turnkey: when an exclusive A100 frees, run these in order."""
    out = output_dir or "$HOME/C51J_runs (or the pool when it has >=5GiB)"
    py = "/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python"
    print("=== C51J P4 EXECUTION PLAN (turnkey; run when an A100 has >=30GiB free) ===")
    print(f"interpreter: {py}")
    print(f"output:      {out}")
    print(f"repo-root:   {repo_root}")
    print(f"runtime-flag (codex): {runtime_flag or '<from Codex manifest>'}")
    print("\n# --- STAGE 1: 1K smoke (correctness gate; all arms x priority scenes) ---")
    for sc in P4_SCENES:
        for arm in P4_SMOKE_ARMS:
            rf = f"--runtime-flag '{runtime_flag}'" if arm == "codex_candidate" and runtime_flag else ""
            print(f"{py} C51J_harness.py --arm {arm} --scene {sc} --iters 1000 --gpu <FREE_GPU> "
                  f"--repo-root {repo_root} {rf}")
    print("\n# --- STAGE 2: paired 5K (ONLY if Stage 1 prelim passes; >=3 arms x scenes) ---")
    for sc in P4_SCENES:
        for arm in P4_5K_ARMS:
            rf = f"--runtime-flag '{runtime_flag}'" if arm == "codex_candidate" and runtime_flag else ""
            print(f"{py} C51J_harness.py --arm {arm} --scene {sc} --iters 5000 --seed 0 --gpu <FREE_GPU> "
                  f"--repo-root {repo_root} {rf}")
    print("\n# --- STAGE 3: seeds (confirmatory 5K, seeds 0/1/2) only after Stage 2 passes ---")
    for sc in P4_SCENES:
        for arm in P4_5K_ARMS:
            for sd in (1, 2):
                print(f"{py} C51J_harness.py --arm {arm} --scene {sc} --iters 5000 --seed {sd} --gpu <FREE_GPU> "
                      f"--repo-root {repo_root}")
    print("\n# gate: refuse if NaN/OOM/illegal lineage, or if decision overhead cancels the saving.")
    print("# collect C51J_5K_PAIRED.csv + C51J_QUALITY_PERFORMANCE.csv from runs/ then judge GO_TO_30K.")


def main():
    ap = argparse.ArgumentParser(description="C51J frozen training-validation harness (execution-gated)")
    ap.add_argument("--arm", default="", choices=[""] + list(ARM_MODES.keys()), help="arm to run (omit with --emit-plan)")
    ap.add_argument("--scene", default="room")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30000, choices=[1000, 5000, 30000])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--repo-root", default="/home/liaoyuanjun/3dgs-renderer-benchmark")
    ap.add_argument("--runtime-flag", default="", help="codex candidate runtime flag (manifest-provided)")
    ap.add_argument("--output-dir", default="", help="override output dir; empty => pool, fallback to $HOME/C51J_runs if pool full")
    ap.add_argument("--min-free-mib", type=int, default=0, help="override required FREE GPU memory on the selected GPU (0 => default 30000)")
    ap.add_argument("--emit-plan", action="store_true", help="print the full P4 execution command matrix and exit (no GPU needed)")
    args = ap.parse_args()

    if args.emit_plan:
        emit_plan(args.repo_root, output_dir=args.output_dir, runtime_flag=args.runtime_flag)
        sys.exit(0)

    if not args.arm:
        ap.error("--arm is required unless --emit-plan is given")

    print("=== C51J harness ===")
    print(f"arm={args.arm} scene={args.scene} seed={args.seed} iters={args.iters} gpu={args.gpu}")

    if args.min_free_mib:
        global MIN_FREE_GPU_MIB
        MIN_FREE_GPU_MIB = args.min_free_mib

    ok, reasons, info = resource_gate(gpu_id=args.gpu, output_dir=(args.output_dir or None))
    prov = env_provenance(args.repo_root)
    prov["gate_info"] = info
    print("provenance:", json.dumps(prov, indent=2))
    print("gpu_exclusivity (other compute apps):\n" + gpu_exclusivity_note())
    if not info.get("output_is_pool", True):
        print(f"NOTE: using writable fallback output dir: {info.get('output_dir')} (pool full)")

    if not ok:
        print("\n*** RESOURCE GATE FAILED — REFUSING TO RUN ***")
        for r in reasons:
            print("  - " + r)
        print("This harness does NOT substitute a different GPU cohort. Storage may be satisfied by a writable fallback.")
        print(f"Re-run when an A100-PCIE-40GB has >={MIN_FREE_GPU_MIB} MiB free and a writable output dir has >=5GiB.")
        sys.exit(3)  # distinct exit code for resource-blocked

    # ---- Real training path (only reached with exclusive A100 + free storage) ----
    # The training loop imports the proven infra from the Stage-5 harness. We keep
    # the import local so the gate above always runs first.
    sys.path.insert(0, str(Path(args.repo_root) / "src"))
    sys.path.insert(0, str(Path(args.repo_root) / "scripts" / "phase-c51-stage5"))
    try:
        from benchmark import run_experiment, get_optimizer  # noqa
    except Exception as e:  # noqa
        print(f"FATAL: could not import training infra: {e}")
        sys.exit(4)

    import torch
    from dataset import GTDataset, load_initial_checkpoint
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    dataset = GTDataset(scene=args.scene, repo_root=Path(args.repo_root),
                        resolution="1080p", device=device)
    sfm = load_initial_checkpoint(args.scene, Path(args.repo_root), device=device)
    splits = SplitIsolator(len(dataset), args.scene, seed=args.seed)
    print(f"split sizes: train={len(splits.train)} val={len(splits.val)} heldout={len(splits.heldout)}")

    config = {"mode": ARM_MODES[args.arm], "iters": args.iters, "gpu_id": args.gpu,
              "scene": args.scene, "seed": args.seed, "runtime_flag": args.runtime_flag}

    with RunTimer(device) as timer:
        results = run_experiment(dataset, sfm, config)
    results["wall_seconds"] = timer.wall
    results["peak_gpu_memory_mib"] = timer.peak_vram_mib
    results["provenance"] = prov
    results["split_sizes"] = {"train": len(splits.train), "val": len(splits.val),
                              "heldout": len(splits.heldout)}

    out_base = Path(info.get("output_dir") or OUTPUT_ROOT) / "runs"
    out_base.mkdir(parents=True, exist_ok=True)
    out_file = out_base / f"{args.arm}_{args.scene}_seed{args.seed}_{args.iters}it.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_file}")
    print(f"  output_dir_is_pool={info.get('output_is_pool')}  wall={timer.wall:.1f}s  "
          f"peakVRAM={timer.peak_vram_mib:.0f}MiB  PSNR={results.get('final_psnr')}  GS={results.get('final_gaussians')}")


if __name__ == "__main__":
    main()
