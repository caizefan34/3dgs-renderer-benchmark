#!/usr/bin/env python3
"""
ReproTopo0 baseline-audit driver (packet Sec.2).

Audits the ORIGINAL C0 topology event for determinism:
  - same-GPU: run the original event 32x on one GPU (identical RNG seed each
    time), hash full state -> unique-hash count.
  - cross-A100: run once on GPU 0/1/2/3 -> unique-hash count.
  - permutation: run on 8 frozen row permutations -> unique-hash count. The
    original is EXPECTED NOT to be invariant (torch.normal draw order follows
    row order), which is the determinism gap ReproTopo closes (Sec.5C).

Baseline result feeds:
  baseline_already_full_deterministic = same_gpu_pass AND cross_gpu_pass
  candidate_adds_stronger_property    = candidate has permutation/tie
                                        invariance that the original lacks.
"""
import sys
import gc
import json
import argparse
import torch

sys.path.insert(0, "/root/reprotopo0_src")
import rt0_common as C
import rt0_event as E


def make_perms(N, n=8, seed=20260930):
    g = torch.Generator()
    g.manual_seed(seed)
    return [torch.randperm(N, generator=g) for _ in range(n)]


def run_orig(snap, device, seed=42):
    model = E.restore_original_model(snap, device=device)
    post, actions = E.run_original_event(model, snap, device=device, seed=seed)
    h, n = E.full_state_hash(post)
    del model, post
    torch.cuda.empty_cache()
    gc.collect()
    return h, actions, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_base", type=int, default=0)
    ap.add_argument("--n_rep", type=int, default=32)
    ap.add_argument("--n_perm", type=int, default=8)
    args = ap.parse_args()
    args.device = "cuda:%d" % args.gpu_base

    snap = torch.load(args.snap, map_location="cpu", weights_only=False)
    scene = snap["scene"]; iteration = snap["iteration"]

    # warmup (prime CUDA + bootstrap C0 once)
    run_orig(snap, args.device, 42)

    # ---- same-GPU repetition (seed reset each run) ----
    hashes = []
    for _ in range(args.n_rep):
        h, a, n = run_orig(snap, args.device, 42)
        hashes.append(h)
    same_gpu_unique = len(set(hashes))

    # ---- cross-A100 (GPU base..base+3) ----
    cross = {}
    for g in range(4):
        dev = "cuda:%d" % (args.gpu_base + g)
        h, a, n = run_orig(snap, dev, 42)
        cross[g] = h
    cross_unique = len(set(cross.values()))

    # ---- permutation (original is EXPECTED to be order-sensitive) ----
    perms = make_perms(int(snap["N"]), n=args.n_perm)
    perm_hashes = []
    for p in perms:
        psnap = E.permute_snapshot(snap, p, device=args.device)
        h, a, n = run_orig(psnap, args.device, 42)
        perm_hashes.append(h)
        del psnap
    perm_unique = len(set(perm_hashes))

    result = {
        "scene": scene, "iteration": iteration, "N_in": int(snap["N"]),
        "same_gpu_repeats": args.n_rep,
        "same_gpu_unique_hashes": same_gpu_unique,
        "same_gpu_determinism_pass": same_gpu_unique == 1,
        "n_gpu_cross": 4,
        "cross_unique_hashes": cross_unique,
        "cross_a100_determinism_pass": cross_unique == 1,
        "cross_hashes": {str(k): v for k, v in cross.items()},
        "n_perm": args.n_perm,
        "perm_unique_hashes": perm_unique,
        "permutation_determinism_pass": perm_unique == 1,
        "permutation_invariant": perm_unique == 1,
        "gpu_base": args.gpu_base,
    }
    C.emit_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
