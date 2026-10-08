# TRAINABLE-HIGS-MEMSCALING-R4-MX - Campaign Complete

**Status:** COMPLETE  
**Execution Mode:** FRESH_MX_MEASUREMENT  
**Date:** 2026-10-08

## Summary

Successfully identified HiGS native backend OOM root cause at 4.4M Gaussians. HiGS requires 11.3x more memory than std_ll baseline due to tile intersection buffer allocation in `isect_tiles` function.

## Key Results

- **OOM Root Cause:** IDENTIFIED (Tile intersection buffers)
- **Memory Amplification:** 11.3x vs std_ll
- **Max Verified N:** 2.0M (41.41 GB peak)
- **OOM Cliff:** 2.0M - 2.5M Gaussians on 40GB GPU
- **Research Decision:** INVESTIGATE

## Experiments Completed

✅ P0: R3 artifacts and environment identity check  
✅ P1: OOM reproduction at 4.4M  
✅ P2: OOM allocation point identification  
✅ P3: N-scaling test (580K to 2.5M)  
✅ P4: OOM cliff determination  
✅ P5: Root cause attribution (H1-H5)  
✅ P6: Fix evaluation (NOT_TESTED - no fix implemented)  
✅ P7: Before/after comparison (N/A)

## Hypothesis Verdicts

- **H1 (Intersection Amplification):** SUPPORTED (HIGH)
- **H2 (Hierarchy Auxiliary Memory):** SUPPORTED (HIGH)
- **H3 (Temporary Workspace):** REFUTED (HIGH)
- **H4 (Allocator Fragmentation):** REFUTED (HIGH)
- **H5 (Over-allocation):** POSSIBLE (MEDIUM)

## Artifacts

**Reports:**
- R4_FINAL_REPORT.md
- R4_FINAL_VERDICT.json
- R4_HYPOTHESIS_VERDICTS.json
- RUN_STATUS.json

**Raw Data:**
- raw/memory_profile_4.4M.json
- raw/oom_stack_trace.json
- raw/n_scaling.json

**Logs:**
- logs/memory_profile.log
- logs/oom_stack_trace.log
- logs/n_scaling.log

**Scripts:**
- scripts/profile_memory.py
- scripts/oom_stack_trace.py
- scripts/n_scaling.py

**Identity:**
- R4_ENVIRONMENT_IDENTITY.json
- R4_SOURCE_IDENTITY.json

## Location

`/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_MEMSCALING_R4_MX/`

## Next Steps

1. Implement intersection buffer compression or chunked processing
2. Test if fix reduces memory to <40GB at 4.4M
3. If successful, benchmark performance at 4.4M
4. If performance >1.20x, continue HiGS research
5. If performance <1.15x, transition to WV-HIER-ORACLE0
