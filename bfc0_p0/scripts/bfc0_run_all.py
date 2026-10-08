#!/usr/bin/env python3
"""BFC0 P0: Main orchestration script.

Executes all DSH steps in order:
- DSH-0: Environment capture
- DSH-1: Prior-art audit (manual, documented)
- DSH-2: Source audit
- DSH-3: Trace collection
- DSH-4/5: Oracle computation
- DSH-6: Timing/economics
- DSH-7/8: Gate evaluation and verdict
"""
import os, sys, json, subprocess

def run_script(script_name, args=None):
    """Run a Python script and return success status."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

def main():
    out_dir = os.environ.get("BFC0_OUT_DIR", "C:\\Users\\36570\\3dgs-renderer-benchmark\\bfc0_p0\\output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Set environment variable for scripts
    env = os.environ.copy()
    env["BFC0_OUT_DIR"] = out_dir
    env["BFC0_REPO_DIR"] = "C:\\Users\\36570\\3dgs-renderer-benchmark"
    
    # DSH-0: Environment capture
    print("\n" + "="*60)
    print("DSH-0: Environment Capture")
    print("="*60)
    run_script("bfc0_env.py")
    
    # DSH-1: Prior-art audit (manual)
    print("\n" + "="*60)
    print("DSH-1: Prior-Art Audit")
    print("="*60)
    print("Manual audit documented in BFC0_PRIOR_ART_MATRIX.md")
    print("No direct collision found; PASS assumed for automated execution.")
    
    # DSH-2: Source audit
    print("\n" + "="*60)
    print("DSH-2: Source Audit")
    print("="*60)
    run_script("bfc0_source_audit.py")
    
    # DSH-3: Trace collection
    print("\n" + "="*60)
    print("DSH-3: Trace Collection")
    print("="*60)
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else "C:\\Users\\36570\\3dgs-renderer-benchmark\\data\\datasets\\mipnerf360\\room"
    run_script("bfc0_trace_collector.py", [scene_dir])
    
    # DSH-4/5: Oracle
    print("\n" + "="*60)
    print("DSH-4/5: Oracle Computation")
    print("="*60)
    run_script("bfc0_oracle.py")
    
    # DSH-6: Timing
    print("\n" + "="*60)
    print("DSH-6: Timing/Economics")
    print("="*60)
    run_script("bfc0_timing.py", [scene_dir])
    
    # DSH-7/8: Gates and verdict
    print("\n" + "="*60)
    print("DSH-7/8: Gate Evaluation and Verdict")
    print("="*60)
    run_script("bfc0_gates.py")
    
    print("\n" + "="*60)
    print("BFC0 P0 Execution Complete")
    print("="*60)
    print(f"Output directory: {out_dir}")

if __name__ == "__main__":
    main()
