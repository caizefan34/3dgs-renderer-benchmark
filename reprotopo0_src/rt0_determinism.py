#!/usr/bin/env python3
"""
ReproTopo0 determinism-stress driver (packet Sec.5).

A. Repetition     : candidate event N times on one GPU -> one unique hash.
B. Cross A100     : candidate event on GPU 0/1/2/3 -> same full-state hash.
C. Permutation    : 8 frozen random row permutations of the SAME logical
                    records; after the candidate event, sort by logical id
                    (full_state_hash already does) -> byte-identical tensors.
D. Tie-stress     : diagnostic copy with ~1% of rows forced to the EXACT
                    decision-score (grad == threshold, threshold unchanged);
                    8 permutations still yield one unique result.

The candidate is pure-torch + device-independent counter RNG, so it should be
invariant under input-row permutation and GPU choice. (The original is NOT,
because torch.normal draw order follows row order -- see rt0_baseline.py.)
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


def make_tie_snapshot(snap):
    """Force ~1% of rows to the EXACT decision-score (grad == threshold)."""
    tsnap = dict(snap)
    cfg = snap["config"]
    thresh = float(cfg["densify_grad_threshold"])
    accum = snap["xyz_gradient_accum"].clone()
    denom = snap["denom"].clone()
    grads = (accum / denom).squeeze(-1)
    grads = torch.nan_to_num(grads, nan=0.0)
    N = int(snap["N"])
    eps = max(1e-6, 1e-3 * thresh)
    near = torch.where(torch.abs(grads - thresh) <= eps)[0].tolist()
    k = max(1, int(0.01 * N))
    if len(near) >= k:
        g = torch.Generator().manual_seed(7)
        sel = torch.as_tensor(near, dtype=torch.long)[
            torch.randperm(len(near), generator=g)[:k]]
    else:
        # not enough near-threshold rows: take all near + fill with random rows
        g = torch.Generator().manual_seed(7)
        pool = list(range(N))
        perm = torch.randperm(N, generator=g).tolist()
        picked = set(near)
        sel = list(near)
        for r in perm:
            if len(sel) >= k:
                break
            if r not in picked:
                sel.append(r)
        sel = torch.as_tensor(sel[:k], dtype=torch.long)
    # exact tie: accum/denom == thresh  (denom=1, accum=thresh -> grad==thresh)
    denom[sel, 0] = 1.0
    accum[sel, 0] = thresh
    tsnap["xyz_gradient_accum"] = accum
    tsnap["denom"] = denom
    tsnap["_tie_rows"] = int(sel.numel())
    return tsnap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_rep", type=int, default=64)
    ap.add_argument("--n_perm", type=int, default=8)
    ap.add_argument("--n_gpu", type=int, default=4)
    ap.add_argument("--gpu_base", type=int, default=0)
    args = ap.parse_args()
    args.device = "cuda:%d" % args.gpu_base

    snap = torch.load(args.snap, map_location="cpu", weights_only=False)
    scene = snap["scene"]; iteration = snap["iteration"]

    # warmup (prime CUDA context / allocator)
    E.run_reprotopo_event(snap, device=args.device)
    torch.cuda.synchronize()

    # ---- A. same-GPU repetition ----
    hashes = []
    for _ in range(args.n_rep):
        post, meta, actions = E.run_reprotopo_event(snap, device=args.device)
        h, _ = E.full_state_hash(post)
        hashes.append(h)
        del post
    same_gpu_unique = len(set(hashes))
    base_hash = hashes[0]

    # ---- B. cross-A100 (GPU 0..n_gpu-1) ----
    cross_hashes = {}
    for g in range(args.n_gpu):
        dev = "cuda:%d" % (args.gpu_base + g)
        post, meta, actions = E.run_reprotopo_event(snap, device=dev)
        h, _ = E.full_state_hash(post)
        cross_hashes[g] = h
        del post
        torch.cuda.empty_cache()
    cross_unique = len(set(cross_hashes.values()))

    # ---- C. input-row permutation ----
    perms = make_perms(int(snap["N"]), n=args.n_perm)
    perm_hashes = []
    for p in perms:
        psnap = E.permute_snapshot(snap, p, device=args.device)
        post, meta, actions = E.run_reprotopo_event(psnap, device=args.device)
        h, _ = E.full_state_hash(post)
        perm_hashes.append(h)
        del post, psnap
    perm_unique = len(set(perm_hashes))
    perm_all_equal_base = all(h == base_hash for h in perm_hashes)

    # ---- D. exact-tie stress ----
    tsnap = make_tie_snapshot(snap)
    tie_hashes = []
    for p in perms:
        psnap = E.permute_snapshot(tsnap, p, device=args.device)
        post, meta, actions = E.run_reprotopo_event(psnap, device=args.device)
        h, _ = E.full_state_hash(post)
        tie_hashes.append(h)
        del post, psnap
    tie_unique = len(set(tie_hashes))

    # representative run for collision / nan / count
    post, meta, actions = E.run_reprotopo_event(snap, device=args.device)
    collision_count = int(meta["collision_count"])
    nan_inf = int(meta["n_nan_inf"])
    N_out = int(meta["N_out"])
    del post

    result = {
        "scene": scene, "iteration": iteration,
        "N_in": int(snap["N"]), "N_out": N_out,
        "same_gpu_repeats": args.n_rep,
        "same_gpu_unique_hashes": same_gpu_unique,
        "same_gpu_determinism_pass": same_gpu_unique == 1,
        "n_gpu_cross": args.n_gpu,
        "gpu_base": args.gpu_base,
        "cross_unique_hashes": cross_unique,
        "cross_hashes": {str(k): v for k, v in cross_hashes.items()},
        "cross_a100_determinism_pass": cross_unique == 1,
        "n_perm": args.n_perm,
        "perm_unique_hashes": perm_unique,
        "perm_all_equal_to_base": perm_all_equal_base,
        "permutation_determinism_pass": (perm_unique == 1) and perm_all_equal_base,
        "tie_rows": int(tsnap.get("_tie_rows", 0)),
        "tie_unique_hashes": tie_unique,
        "tie_stress_pass": tie_unique == 1,
        "collision_count": collision_count,
        "nan_inf_count": nan_inf,
        "base_hash": base_hash,
    }
    C.emit_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
