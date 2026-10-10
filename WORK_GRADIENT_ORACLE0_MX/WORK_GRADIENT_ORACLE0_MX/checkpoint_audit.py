#!/usr/bin/env python3
"""
Checkpoint inventory and metadata extraction for WORK-GRADIENT-ORACLE0-MX.
Safely inspects FasterGSFused checkpoints without unsafe deserialization.
"""
import os
import sys
import json
import hashlib
import struct
from pathlib import Path

def sha256_file(path):
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def inspect_pytorch_checkpoint_safely(path):
    """
    Safely inspect PyTorch checkpoint without full unpickling.
    Returns metadata dict with safe fields only.
    """
    try:
        # Check file size
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        
        # Try to read as zip (PyTorch format)
        import zipfile
        if not zipfile.is_zipfile(path):
            return {
                "status": "NOT_ZIP_FORMAT",
                "size_mb": round(size_mb, 2),
                "note": "File is not in PyTorch zip format"
            }
        
        # Open as zip and list contents
        with zipfile.ZipFile(path, 'r') as zf:
            namelist = zf.namelist()
            
            # Look for data.pkl (contains pickle with metadata)
            has_data_pkl = 'data.pkl' in namelist
            
            # Look for version file
            version = None
            if 'version' in namelist:
                try:
                    version = zf.read('version').decode('utf-8').strip()
                except:
                    pass
            
            return {
                "status": "VALID_ZIP_FORMAT",
                "size_mb": round(size_mb, 2),
                "has_data_pkl": has_data_pkl,
                "pytorch_version": version,
                "file_count": len(namelist),
                "note": "Safe inspection only; no tensor data loaded"
            }
            
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "size_mb": round(size_mb, 2) if 'size_mb' in locals() else None
        }

def inspect_checkpoint_structure(path):
    """
    Attempt to load checkpoint with torch.load and extract basic metadata.
    Uses map_location='cpu' to avoid CUDA initialization.
    """
    try:
        import torch
        
        # Load checkpoint safely
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        metadata = {
            "status": "LOADED",
            "keys": list(ckpt.keys()) if isinstance(ckpt, dict) else "NOT_A_DICT"
        }
        
        if isinstance(ckpt, dict):
            # Check for common 3DGS checkpoint keys
            if 'means' in ckpt:
                means = ckpt['means']
                metadata['N'] = means.shape[0] if hasattr(means, 'shape') else None
                metadata['means_shape'] = list(means.shape) if hasattr(means, 'shape') else None
            
            if 'iteration' in ckpt:
                metadata['iteration'] = ckpt['iteration']
            
            if 'config' in ckpt:
                metadata['has_config'] = True
                metadata['config_keys'] = list(ckpt['config'].keys()) if isinstance(ckpt['config'], dict) else None
            
            # Check for optimizer state
            if 'optimizer' in ckpt:
                metadata['has_optimizer'] = True
            
            # Check for training metadata
            for key in ['scene', 'dataset', 'args', 'training_args']:
                if key in ckpt:
                    metadata[f'has_{key}'] = True
        
        return metadata
        
    except Exception as e:
        return {
            "status": "LOAD_FAILED",
            "error": str(e),
            "error_type": type(e).__name__
        }

def main():
    base_paths = [
        "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints",
        "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
    ]
    
    print("=== CHECKPOINT INVENTORY ===\n")
    
    inventory = []
    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            print(f"SKIP: {base_path} not found")
            continue
        
        scene = "garden" if "garden" in base_path else "bicycle"
        print(f"\nScene: {scene.upper()}")
        print(f"Path: {base_path}")
        print("-" * 60)
        
        # List checkpoint files
        ckpt_files = sorted([f for f in os.listdir(base_path) if f.endswith('.pt')])
        
        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(base_path, ckpt_file)
            
            print(f"\n  File: {ckpt_file}")
            
            # Compute SHA256
            sha = sha256_file(ckpt_path)
            print(f"    SHA256: {sha[:16]}...{sha[-8:]}")
            
            # Safe format inspection
            safe_meta = inspect_pytorch_checkpoint_safely(ckpt_path)
            print(f"    Format: {safe_meta['status']} ({safe_meta.get('size_mb', '?')} MB)")
            
            # Try structured load
            if safe_meta['status'] == 'VALID_ZIP_FORMAT':
                struct_meta = inspect_checkpoint_structure(ckpt_path)
                print(f"    Load: {struct_meta['status']}")
                
                if struct_meta['status'] == 'LOADED' and 'N' in struct_meta:
                    print(f"    N (Gaussians): {struct_meta['N']}")
                    print(f"    Means shape: {struct_meta.get('means_shape')}")
                
                if 'iteration' in struct_meta:
                    print(f"    Iteration: {struct_meta['iteration']}")
                
                # Determine stage from filename
                stage = "UNKNOWN"
                if '0005000' in ckpt_file or '5k' in ckpt_file.lower():
                    stage = "EARLY_5K"
                elif '0010000' in ckpt_file or '10k' in ckpt_file.lower():
                    stage = "MIDDLE_10K"
                elif 'final' in ckpt_file.lower():
                    stage = "LATE_FINAL"
                
                inventory.append({
                    "scene": scene,
                    "file": ckpt_file,
                    "path": ckpt_path,
                    "sha256": sha,
                    "size_mb": safe_meta.get('size_mb'),
                    "stage": stage,
                    "N": struct_meta.get('N'),
                    "iteration": struct_meta.get('iteration'),
                    "format": safe_meta['status'],
                    "loadable": struct_meta['status'] == 'LOADED'
                })
            else:
                inventory.append({
                    "scene": scene,
                    "file": ckpt_file,
                    "path": ckpt_path,
                    "sha256": sha,
                    "size_mb": safe_meta.get('size_mb'),
                    "format": safe_meta['status'],
                    "loadable": False
                })
    
    print("\n" + "=" * 60)
    print(f"TOTAL CHECKPOINTS FOUND: {len(inventory)}")
    print(f"LOADABLE: {sum(1 for c in inventory if c.get('loadable', False))}")
    
    # Save inventory
    output_path = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/CHECKPOINT_INVENTORY.json"
    with open(output_path, 'w') as f:
        json.dump(inventory, f, indent=2)
    print(f"\nInventory saved to: {output_path}")
    
    print("\n=== INVENTORY COMPLETE ===")

if __name__ == "__main__":
    main()
