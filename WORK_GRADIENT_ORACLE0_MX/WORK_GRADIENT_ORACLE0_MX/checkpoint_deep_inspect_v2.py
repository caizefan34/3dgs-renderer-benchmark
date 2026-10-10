#!/usr/bin/env python3
"""
Deep checkpoint inspection to extract N (Gaussian count) and iteration from model_state_dict.
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
        
        # Extract iteration
        if 'num_iterations_trained' in ckpt:
            result['iteration'] = ckpt['num_iterations_trained']
        
        # Extract metadata
        if 'model_name' in ckpt:
            result['model_name'] = ckpt['model_name']
        
        if 'SH_DEGREE' in ckpt:
            result['sh_degree'] = ckpt['SH_DEGREE']
        
        if 'creation_date' in ckpt:
            result['creation_date'] = ckpt['creation_date']
        
        # Extract Gaussian parameters from model_state_dict
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            result['state_dict_keys'] = list(state_dict.keys())[:15]  # First 15 keys
            
            # Look for Gaussian parameter keys
            param_patterns = ['means', 'xyz', 'positions', 'quats', 'rotations', 
                            'scales', 'opacities', 'opacity', 'sh', 'colors']
            
            found_params = {}
            for key in state_dict.keys():
                for pattern in param_patterns:
                    if pattern in key.lower():
                        tensor = state_dict[key]
                        if hasattr(tensor, 'shape'):
                            found_params[key] = list(tensor.shape)
                        break
            
            result["gaussian_params"] = found_params
            
            # Extract N from means/xyz/positions
            for key in found_params:
                if any(x in key.lower() for x in ['means', 'xyz', 'positions']):
                    result['N'] = found_params[key][0]
                    break
        
        return result
        
    except Exception as e:
        import traceback
        return {
            "status": "ERROR",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
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
    
    print("=== DEEP CHECKPOINT INSPECTION (model_state_dict) ===\n")
    
    results = []
    
    for scene, stage, path in checkpoints:
        print(f"{scene.upper()} - {stage}")
        print("-" * 60)
        
        if not os.path.exists(path):
            print(f"  NOT FOUND: {path}\n")
            results.append({"scene": scene, "stage": stage, "status": "NOT_FOUND"})
            continue
        
        result = inspect_checkpoint_deep(path)
        results.append({"scene": scene, "stage": stage, **result})
        
        print(f"  Status: {result['status']}")
        
        if result['status'] == 'LOADED':
            if 'N' in result:
                print(f"  ✓ N (Gaussians): {result['N']:,}")
            
            if 'iteration' in result:
                print(f"  ✓ Iteration: {result['iteration']:,}")
            
            if 'model_name' in result:
                print(f"  Model: {result['model_name']}")
            
            if 'sh_degree' in result:
                print(f"  SH degree: {result['sh_degree']}")
            
            if 'gaussian_params' in result:
                print(f"  Gaussian params:")
                for k, v in result['gaussian_params'].items():
                    print(f"    {k}: {v}")
        
        elif result['status'] == 'ERROR':
            print(f"  Error: {result.get('error')}")
        
        print()
    
    # Save results
    import json
    output_path = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/CHECKPOINT_DEEP_RESULTS.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
    
    print("=== INSPECTION COMPLETE ===")

if __name__ == "__main__":
    main()
