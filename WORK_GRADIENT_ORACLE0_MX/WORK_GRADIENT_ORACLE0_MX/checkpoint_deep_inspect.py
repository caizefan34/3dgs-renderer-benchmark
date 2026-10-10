#!/usr/bin/env python3
"""
Deep checkpoint inspection to extract N (Gaussian count) and iteration.
"""
import os
import sys
import torch

def inspect_checkpoint_deep(path):
    """Deep inspection to extract tensor shapes and metadata."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        result = {
            "status": "LOADED",
            "top_level_keys": list(ckpt.keys()) if isinstance(ckpt, dict) else [],
        }
        
        if not isinstance(ckpt, dict):
            result["type"] = type(ckpt).__name__
            return result
        
        # Look for Gaussian parameters (common 3DGS checkpoint structure)
        param_keys = ['means', 'quats', 'scales', 'opacities', 'sh']
        found_params = {}
        
        for key in param_keys:
            if key in ckpt:
                tensor = ckpt[key]
                if hasattr(tensor, 'shape'):
                    found_params[key] = list(tensor.shape)
        
        result["gaussian_params"] = found_params
        
        # Extract N from means
        if 'means' in found_params:
            result["N"] = found_params['means'][0]
        
        # Look for iteration counter
        iter_keys = ['iteration', 'iter', 'step', 'global_step']
        for key in iter_keys:
            if key in ckpt:
                result['iteration'] = ckpt[key]
                break
        
        # Look for metadata sub-dict
        if 'meta' in ckpt and isinstance(ckpt['meta'], dict):
            result['meta_keys'] = list(ckpt['meta'].keys())
        
        # Check for config/args
        for cfg_key in ['config', 'args', 'training_args', 'cfg']:
            if cfg_key in ckpt:
                result['has_config'] = True
                if isinstance(ckpt[cfg_key], dict):
                    result['config_keys'] = list(ckpt[cfg_key].keys())[:10]  # First 10
        
        return result
        
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "error_type": type(e).__name__
        }

def main():
    checkpoints = [
        ("garden", "EARLY_5K", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints/0005000.pt"),
        ("garden", "MIDDLE_10K", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints/0010000.pt"),
        ("garden", "LATE_FINAL", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints/final.pt"),
        ("bicycle", "EARLY_5K", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"),
        ("bicycle", "MIDDLE_10K", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0010000.pt"),
        ("bicycle", "LATE_FINAL", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/final.pt"),
    ]
    
    print("=== DEEP CHECKPOINT INSPECTION ===\n")
    
    for scene, stage, path in checkpoints:
        print(f"{scene.upper()} - {stage}")
        print("-" * 60)
        
        if not os.path.exists(path):
            print(f"  NOT FOUND: {path}\n")
            continue
        
        result = inspect_checkpoint_deep(path)
        
        print(f"  Status: {result['status']}")
        
        if result['status'] == 'LOADED':
            print(f"  Top-level keys: {result.get('top_level_keys', [])[:10]}")
            
            if 'N' in result:
                print(f"  ✓ N (Gaussians): {result['N']}")
            
            if 'iteration' in result:
                print(f"  ✓ Iteration: {result['iteration']}")
            
            if 'gaussian_params' in result:
                print(f"  Gaussian params found:")
                for k, v in result['gaussian_params'].items():
                    print(f"    {k}: {v}")
            
            if 'has_config' in result:
                print(f"  Config keys: {result.get('config_keys', [])}")
        
        print()
    
    print("=== INSPECTION COMPLETE ===")

if __name__ == "__main__":
    main()
