#!/usr/bin/env python3
"""Fix ptxas extraction with correct CDIM=3 parsing and occupancy computation."""
import sys
import re
import json
import math
from pathlib import Path

def parse_ptxas_for_kernel(log_text: str, kernel_pattern: str, cdim: int):
    """Parse ptxas output for a specific kernel+CDIM.
    
    The mangled name for CDIM=3 contains 'ILj3EfEE' where Lj3E = template param 3.
    """
    lines = log_text.split('\n')
    
    # Pattern: the kernel name with the specific CDIM template parameter
    # e.g. rasterize_to_pixels_3dgs_bwd_kernelILj3EfEE
    target = f"{kernel_pattern}ILj{cdim}EfEE"
    
    for i, line in enumerate(lines):
        if target in line and "Compiling entry function" in line:
            # Found the kernel - look at next few lines for resource info
            info = {'function': target, 'cdim': cdim}
            for j in range(i+1, min(i+5, len(lines))):
                l = lines[j].strip()
                if "stack frame" in l and "spill" in l:
                    m = re.search(r'(\d+)\s+bytes\s+stack\s+frame', l)
                    if m: info['stack_frame_bytes'] = int(m.group(1))
                    m = re.search(r'(\d+)\s+bytes\s+spill\s+stores', l)
                    if m: info['spill_stores_bytes'] = int(m.group(1))
                    m = re.search(r'(\d+)\s+bytes\s+spill\s+loads', l)
                    if m: info['spill_loads_bytes'] = int(m.group(1))
                elif "Used" in l and "registers" in l:
                    m = re.search(r'Used\s+(\d+)\s+registers', l)
                    if m: info['registers'] = int(m.group(1))
                    m = re.search(r'(\d+)\s+bytes\s+cmem', l)
                    if m: info['cmem_bytes'] = int(m.group(1))
                    # smem is not reported by ptxas for dynamic shared memory
                elif "Compiling entry function" in l:
                    break  # Next kernel started
            return info
    return None


def compute_occupancy_a100(registers: int, threads_per_block: int = 256, smem_bytes: int = 0):
    """Compute theoretical occupancy for A100 (SM 8.0) using whole-block residency.

    A100 SM 8.0 specs:
    - Max 2048 threads per SM (64 warps)
    - Max 32 blocks per SM
    - 65536 registers per SM
    - Register allocation unit: 256 registers per warp
    - 164KB shared memory (99KB = 101376 bytes default without opt-in)

    Whole-block residency: the number of resident warps must be a multiple
    of warps_per_block (8 for 256 threads). We compute blocks_per_sm from
    each limiter, take the min, then multiply by warps_per_block.
    """
    MAX_THREADS_PER_SM = 2048
    MAX_WARPS_PER_SM = 64
    MAX_BLOCKS_PER_SM = 32
    MAX_REGS_PER_SM = 65536
    REG_ALLOC_UNIT = 256  # registers per warp allocation unit
    DEFAULT_SMEM_PER_BLOCK = 101376  # 99KB default without opt-in

    warps_per_block = math.ceil(threads_per_block / 32)  # 8 for 256 threads

    # Register limit: max blocks based on register file
    regs_per_warp = max(1, math.ceil(registers * 32 / REG_ALLOC_UNIT)) * REG_ALLOC_UNIT
    blocks_per_sm_from_regs = MAX_REGS_PER_SM // (regs_per_warp * warps_per_block)

    # Thread/block limit
    blocks_per_sm_from_threads = MAX_THREADS_PER_SM // (warps_per_block * 32)

    # Shared memory limit (dynamic shared memory set at runtime)
    if smem_bytes > 0:
        blocks_per_sm_from_smem = DEFAULT_SMEM_PER_BLOCK // smem_bytes
    else:
        blocks_per_sm_from_smem = MAX_BLOCKS_PER_SM

    # Final occupancy: min blocks across all limiters
    blocks_per_sm = min(blocks_per_sm_from_regs, blocks_per_sm_from_threads,
                        blocks_per_sm_from_smem, MAX_BLOCKS_PER_SM)
    warps_per_sm = blocks_per_sm * warps_per_block
    occupancy_pct = warps_per_sm / MAX_WARPS_PER_SM * 100

    return {
        'registers': registers,
        'regs_per_warp': regs_per_warp,
        'warps_per_block': warps_per_block,
        'blocks_per_sm_from_regs': blocks_per_sm_from_regs,
        'blocks_per_sm_from_threads': blocks_per_sm_from_threads,
        'blocks_per_sm_from_smem': blocks_per_sm_from_smem,
        'blocks_per_sm': blocks_per_sm,
        'warps_per_sm': warps_per_sm,
        'max_warps_per_sm': MAX_WARPS_PER_SM,
        'occupancy_pct': round(occupancy_pct, 2),
        'smem_bytes': smem_bytes,
        'threads_per_block': threads_per_block,
    }


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "3dgs-renderer-benchmark/results/n2_cpcb"
    baseline_log = (results_dir / "build_logs/baseline_build.log").read_text()
    cpcb_log = (results_dir / "build_logs/cpcb_build.log").read_text()
    
    # Shared memory for tile_size=16, CDIM=3:
    # shmem = 16*16 * (sizeof(int32) + sizeof(vec3) + sizeof(vec3) + sizeof(float)*CDIM)
    # = 256 * (4 + 12 + 12 + 4*3) = 256 * 40 = 10240
    smem_cdim3 = 256 * (4 + 12 + 12 + 4 * 3)  # 256 * 40 = 10240 bytes
    smem_cdim8 = 256 * (4 + 12 + 12 + 4 * 8)  # 256 * 60 = 15360 bytes
    
    result = {}
    
    for cdim, smem in [(3, smem_cdim3), (8, smem_cdim8)]:
        base_kern = parse_ptxas_for_kernel(baseline_log, "rasterize_to_pixels_3dgs_bwd_kernel", cdim)
        cpcb_kern = parse_ptxas_for_kernel(cpcb_log, "rasterize_to_pixels_3dgs_bwd_kernel", cdim)
        
        if base_kern:
            base_kern['smem_bytes'] = smem
            base_occ = compute_occupancy_a100(base_kern['registers'], 256, smem)
            base_kern['occupancy'] = base_occ
        else:
            base_kern = {'error': 'not found'}
            base_occ = {}
            
        if cpcb_kern:
            cpcb_kern['smem_bytes'] = smem
            cpcb_occ = compute_occupancy_a100(cpcb_kern['registers'], 256, smem)
            cpcb_kern['occupancy'] = cpcb_occ
        else:
            cpcb_kern = {'error': 'not found'}
            cpcb_occ = {}
        
        result[f'cdim{cdim}'] = {
            'baseline': base_kern,
            'cpcb': cpcb_kern,
        }
        
        if 'registers' in base_kern and 'registers' in cpcb_kern:
            delta = cpcb_kern['registers'] - base_kern['registers']
            result[f'cdim{cdim}']['register_delta'] = delta
            result[f'cdim{cdim}']['occupancy_change'] = {
                'baseline_pct': base_occ.get('occupancy_pct'),
                'cpcb_pct': cpcb_occ.get('occupancy_pct'),
                'delta_pct': (cpcb_occ.get('occupancy_pct', 0) - base_occ.get('occupancy_pct', 0)),
                'baseline_warps': base_occ.get('warps_per_sm'),
                'cpcb_warps': cpcb_occ.get('warps_per_sm'),
                'warps_delta': (cpcb_occ.get('warps_per_sm', 0) - base_occ.get('warps_per_sm', 0)),
            }
    
    # Summary table
    result['summary'] = {
        'metric': ['registers/thread', 'spill_stores', 'spill_loads', 'stack_frame', 
                    'smem_bytes', 'theoretical_occupancy', 'warps_per_sm'],
    }
    
    for cdim in [3, 8]:
        bk = result.get(f'cdim{cdim}', {}).get('baseline', {})
        ck = result.get(f'cdim{cdim}', {}).get('cpcb', {})
        bo = bk.get('occupancy', {})
        co = ck.get('occupancy', {})
        result['summary'][f'baseline_cdim{cdim}'] = [
            bk.get('registers'), bk.get('spill_stores_bytes', 0), bk.get('spill_loads_bytes', 0),
            bk.get('stack_frame_bytes', 0), bk.get('smem_bytes'), bo.get('occupancy_pct'),
            bo.get('warps_per_sm')
        ]
        result['summary'][f'cpcb_cdim{cdim}'] = [
            ck.get('registers'), ck.get('spill_stores_bytes', 0), ck.get('spill_loads_bytes', 0),
            ck.get('stack_frame_bytes', 0), ck.get('smem_bytes'), co.get('occupancy_pct'),
            co.get('warps_per_sm')
        ]
        result['summary'][f'delta_cdim{cdim}'] = [
            (ck.get('registers', 0) - bk.get('registers', 0)),
            (ck.get('spill_stores_bytes', 0) - bk.get('spill_stores_bytes', 0)),
            (ck.get('spill_loads_bytes', 0) - bk.get('spill_loads_bytes', 0)),
            (ck.get('stack_frame_bytes', 0) - bk.get('stack_frame_bytes', 0)),
            0,  # smem unchanged
            (co.get('occupancy_pct', 0) - bo.get('occupancy_pct', 0)),
            (co.get('warps_per_sm', 0) - bo.get('warps_per_sm', 0)),
        ]
    
    result['note'] = (
        "A100 SM 8.0: 65536 regs/SM, 2048 max threads/SM (64 warps), "
        "256 regs/warp allocation unit, 99KB default smem/block. "
        "tile_size=16, threads_per_block=256, warps_per_block=8. "
        "smem = tile_size^2 * (sizeof(int32) + sizeof(vec3) + sizeof(vec3) + sizeof(float)*CDIM) "
        "= 256*(4+12+12+4*CDIM) bytes. "
        "Occupancy uses whole-block residency: blocks_per_sm = min(regs, threads, smem), "
        "warps_per_sm = blocks_per_sm * warps_per_block. "
        "No fractional/non-block-multiple warp counts."
    )
    
    out_path = results_dir / "compiler_resources.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")
    
    # Print summary
    print("\n=== N2-R1 RESOURCE AUDIT ===")
    for cdim in [3, 8]:
        bk = result.get(f'cdim{cdim}', {}).get('baseline', {})
        ck = result.get(f'cdim{cdim}', {}).get('cpcb', {})
        bo = bk.get('occupancy', {})
        co = ck.get('occupancy', {})
        print(f"\nCDIM={cdim}:")
        print(f"  {'Metric':<25} {'Baseline':>10} {'CPCB':>10} {'Delta':>10}")
        print(f"  {'Registers/thread':<25} {bk.get('registers','?'):>10} {ck.get('registers','?'):>10} {(ck.get('registers',0)-bk.get('registers',0)):>+10}")
        print(f"  {'Spill stores (bytes)':<25} {bk.get('spill_stores_bytes',0):>10} {ck.get('spill_stores_bytes',0):>10} {0:>+10}")
        print(f"  {'Spill loads (bytes)':<25} {bk.get('spill_loads_bytes',0):>10} {ck.get('spill_loads_bytes',0):>10} {0:>+10}")
        print(f"  {'Stack frame (bytes)':<25} {bk.get('stack_frame_bytes',0):>10} {ck.get('stack_frame_bytes',0):>10} {0:>+10}")
        print(f"  {'Shared mem (bytes)':<25} {bk.get('smem_bytes','?'):>10} {ck.get('smem_bytes','?'):>10} {0:>+10}")
        print(f"  {'Theoretical occupancy':<25} {str(bo.get('occupancy_pct','?'))+'%':>10} {str(co.get('occupancy_pct','?'))+'%':>10} {(co.get('occupancy_pct',0)-bo.get('occupancy_pct',0)):>+10}")
        print(f"  {'Warps per SM':<25} {bo.get('warps_per_sm','?'):>10} {co.get('warps_per_sm','?'):>10} {(co.get('warps_per_sm',0)-bo.get('warps_per_sm',0)):>+10}")


if __name__ == "__main__":
    main()
