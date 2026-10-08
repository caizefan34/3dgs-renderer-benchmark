#!/usr/bin/env python3
"""Definitive check: does R1 actually allocate + use the Gaussian cache?
memory_allocated delta R0->R1 forward reveals cache = n_buckets*32*256*4 bytes.
"""
import sys, os, json
BWD1_DIR = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN"
sys.path.insert(0, os.path.join(BWD1_DIR, "rasterization"))
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")
import torch
import mx_probe as MP
import mx_perf as PERF

def set_mode(mode):
    if mode == "r1": os.environ["BWD1_CACHE_ENABLE"] = "1"
    else: os.environ.pop("BWD1_CACHE_ENABLE", None)

def main():
    cfg, ckpt = sys.argv[1], sys.argv[2]
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    trainer, dataset, train_ds, sampler, ad, g = MP.build(cfg, ckpt)
    view = list(train_ds)[0]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    set_mode("r0")
    img0 = PERF.h_fwd(g, view); torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    set_mode("r1")
    img1 = PERF.h_fwd(g, view); torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated()
    set_mode("r0")
    delta = m1 - m0
    print("MEM_R0_alloc_MB=%.1f MEM_R1_alloc_MB=%.1f DELTA_MB=%.1f" % (m0/1e6, m1/1e6, delta/1e6))
    print("CACHE_EXPECTED_MIB", round(delta/1048576,1))
    print("IMG_BIT_IDENTICAL", torch.equal(img0, img1), "MAXABS", float((img0-img1).abs().max()))

if __name__ == "__main__":
    main()
