#!/usr/bin/env python3
"""FP-dataflow comparison of the stored SASS instruction streams (from
p10_sasshash.py) for pristine vs patched-off.

Decides whether the patched build's *arithmetic* is identical to the pristine
build's when the switch is off: opcode multiset, opcode sequence, FP-only
sequence, and the FP dataflow after alpha-renaming of registers (so register
re-coloring and MOV scaffolding are normalized away, while a changed operation
order / FMA contraction / operand role is not).
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, "/root/epic05_p10")
from p10_sass_cmp import canonical, fp_stream, first_diff, INSTR_RE  # noqa: E402

MARK = {"off": "Lb0E", "on": "Lb1E"}


def load(path, arm):
    with open(path) as f:
        d = json.load(f)
    funcs = d["functions"]
    out = {}
    skipped = 0
    for name, rec in funcs.items():
        has = [m for m in MARK.values() if m in name]
        if arm == "off" and has and MARK["off"] not in name:
            skipped += 1
            continue
        if arm == "on" and MARK["on"] not in name:
            skipped += 1
            continue
        out[canonical(name)] = rec
    return out, skipped


def instrs_of(rec):
    out = []
    for line in rec["instrs"]:
        m = INSTR_RE.match(line) if line.startswith(" ") else None
        if m:
            out.append((None, m.group(2), line))
        else:
            m2 = __import__("re").match(r"^\s*(@!?U?P[0-9T]\s+)?([A-Z][A-Z0-9._]*)", line)
            if m2:
                out.append((None, m2.group(2), line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pri", required=True)
    ap.add_argument("--h8", required=True)
    ap.add_argument("--arm", choices=["off", "on"], default="off")
    ap.add_argument("--family", default="rasterize_to_pixels_3dgs_bwd_kernel")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    P, sp = load(a.pri, a.arm)
    H, sh = load(a.h8, a.arm)
    rows = []
    for k in sorted(P):
        if a.family not in k:
            continue
        if k not in H:
            rows.append({"symbol": k, "matched": False})
            continue
        ip, ih = instrs_of(P[k]), instrs_of(H[k])
        ops_p = [o for _, o, _ in ip]
        ops_h = [o for _, o, _ in ih]
        fp_p, fp_h = fp_stream(ip), fp_stream(ih)
        multiset_same = collections.Counter(ops_p) == collections.Counter(ops_h)
        row = {
            "symbol": k, "matched": True,
            "n_instructions": {"pri": len(ops_p), "h8": len(ops_h)},
            "opcode_multiset_identical": multiset_same,
            "opcode_sequence_identical": ops_p == ops_h,
            "fp_sequence_identical": [o for o in ops_p if o in __import__("p10_sass_cmp").FP_OPS]
                                    == [o for o in ops_h if o in __import__("p10_sass_cmp").FP_OPS],
            "fp_dataflow_identical": fp_p == fp_h,
            "n_fp_instructions": len(fp_p),
        }
        if not multiset_same:
            cp, ch = collections.Counter(ops_p), collections.Counter(ops_h)
            row["opcode_delta"] = {o: ch.get(o, 0) - cp.get(o, 0) for o in sorted(set(cp) | set(ch))
                                   if ch.get(o, 0) != cp.get(o, 0)}
        d = first_diff(ops_p, ops_h)
        row["first_opcode_diff"] = None if d is None else {"index": d[0], "pri": d[1], "h8": d[2]}
        d2 = first_diff(fp_p, fp_h)
        row["first_fp_dataflow_diff"] = None if d2 is None else {
            "index": d2[0], "pri": d2[1], "h8": d2[2],
            "context_pri": fp_p[max(0, d2[0] - 4):d2[0] + 4],
            "context_h8": fp_h[max(0, d2[0] - 4):d2[0] + 4]}
        rows.append(row)

    out = {"family": a.family, "arm": a.arm, "pri": a.pri, "h8": a.h8,
           "n_pri_skipped_instantiations": sp, "n_h8_skipped_instantiations": sh,
           "n_matched": sum(1 for r in rows if r["matched"]),
           "all_fp_dataflow_identical": bool(rows) and all(r.get("fp_dataflow_identical") for r in rows),
           "all_fp_sequence_identical": bool(rows) and all(r.get("fp_sequence_identical") for r in rows),
           "all_opcode_multiset_identical": bool(rows) and all(r.get("opcode_multiset_identical") for r in rows),
           "n_fp_dataflow_mismatched": sum(1 for r in rows if r.get("matched") and not r.get("fp_dataflow_identical")),
           "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{a.family} arm={a.arm}: matched={out['n_matched']} "
          f"fp_dataflow_identical={out['all_fp_dataflow_identical']} "
          f"fp_seq_identical={out['all_fp_sequence_identical']} "
          f"multiset_identical={out['all_opcode_multiset_identical']} "
          f"mismatched={out['n_fp_dataflow_mismatched']} -> {a.out}")
    for r in rows[:3]:
        if r["matched"] and not r["fp_dataflow_identical"]:
            print("  first mismatch:", r["symbol"][:60], r["first_fp_dataflow_diff"]["index"])
            for c in r["first_fp_dataflow_diff"]["context_pri"]:
                print("    pri:", c)
            for c in r["first_fp_dataflow_diff"]["context_h8"]:
                print("    h8 :", c)
            break


if __name__ == "__main__":
    main()
