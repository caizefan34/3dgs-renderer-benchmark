#!/usr/bin/env python3
"""Characterise the off-vs-pristine SASS difference of the rasterizer backward.

Consumes the normalised dumps written by p10_sasshash.py (--keep-instrs) and
answers, per matched instantiation:
  - instruction counts
  - whether the *multiset* of full instruction lines is identical (pure
    permutation => scheduling / register-allocation difference only)
  - per-opcode histogram delta (an FFMA/FMUL swing => contraction difference,
    which changes fp rounding)
  - the raw text of the first differing instructions
"""
import argparse
import collections
import json

MARKER = {"off": "Lb0E", "on": "Lb1E"}
KERNEL = "rasterize_to_pixels_3dgs_bwd_kernel"


def stem(instr):
    for t in instr.split():
        if not t.startswith("@"):
            return t.split(".")[0]
    return "?"


def index(functions, arm):
    want = MARKER.get(arm)
    out = {}
    for f, d in functions.items():
        if want and want not in f:
            continue
        if not want and any(m in f for m in MARKER.values()):
            continue
        out[f.replace("Lb0E", "").replace("Lb1E", "")] = (f, d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pri", required=True)
    ap.add_argument("--h8", required=True)
    ap.add_argument("--kernel", default=KERNEL)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pri = json.load(open(args.pri))
    h8 = json.load(open(args.h8))
    P = index(pri["functions"], pri["arm"])
    H = index(h8["functions"], h8["arm"])

    res = {"pri_arm": pri["arm"], "h8_arm": h8["arm"], "kernel": args.kernel, "functions": []}
    for c in sorted(set(P) & set(H)):
        if args.kernel not in c:
            continue
        pf, pd = P[c]
        hf, hd = H[c]
        a, b = pd["instrs"] or [], hd["instrs"] or []
        ca, cb = collections.Counter(a), collections.Counter(b)
        oa = collections.Counter(stem(x) for x in a)
        ob = collections.Counter(stem(x) for x in b)
        delta = {k: ob[k] - oa[k] for k in set(oa) | set(ob) if ob[k] != oa[k]}
        only_pri = sum((ca - cb).values())
        only_h8 = sum((cb - ca).values())
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        res["functions"].append({
            "canonical": c,
            "n_instr_pri": len(a), "n_instr_h8": len(b),
            "multiset_identical": only_pri == 0 and only_h8 == 0,
            "lines_only_pri": only_pri, "lines_only_h8": only_h8,
            "opcode_histogram_identical": not delta,
            "opcode_delta_top": dict(sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:12]),
            "fp_delta": {k: delta.get(k, 0) for k in ("FFMA", "FMUL", "FADD", "FSETP", "FSEL", "FMNMX", "MUFU")},
            "atom_delta": {k: delta.get(k, 0) for k in ("RED", "ATOM", "ATOMG", "REDG", "ATOMS", "REDS", "ATOMSR")},
            "ldst_delta": {k: delta.get(k, 0) for k in ("LDG", "STG", "LDS", "STS", "LDL", "STL")},
            "first_diff_index": first,
            "diff_context": [{"i": i, "pri": a[i] if i < len(a) else None,
                              "h8": b[i] if i < len(b) else None}
                             for i in range(max(0, first - 3), min(first + args.context, max(len(a), len(b))))],
        })
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"kernel={args.kernel}: {len(res['functions'])} matched instantiations")
    n_multiset = sum(1 for r in res["functions"] if r["multiset_identical"])
    n_hist = sum(1 for r in res["functions"] if r["opcode_histogram_identical"])
    print(f"  multiset-identical: {n_multiset}/{len(res['functions'])}  "
          f"opcode-histogram-identical: {n_hist}/{len(res['functions'])}")
    for r in res["functions"][:3]:
        print(f"  {r['canonical'][:70]}...")
        print(f"    n_instr {r['n_instr_pri']}->{r['n_instr_h8']}  lines only-pri={r['lines_only_pri']} "
              f"only-h8={r['lines_only_h8']}  multiset_eq={r['multiset_identical']} hist_eq={r['opcode_histogram_identical']}")
        print(f"    opcode delta: {r['opcode_delta_top']}")
        print(f"    fp delta: {r['fp_delta']}  atom delta: {r['atom_delta']}  ldst delta: {r['ldst_delta']}")
        for d in r["diff_context"][:5]:
            print(f"      [{d['i']}] pri: {d['pri']}")
            print(f"      [{d['i']}] h8 : {d['h8']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
