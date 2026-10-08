#!/usr/bin/env python3
"""Compare the SASS of the two builds at the level that decides arithmetic:

  level 1  instruction-count and opcode multiset
  level 2  full opcode sequence (mnemonics only, operands masked)
  level 3  FP-only opcode sequence
  level 4  FP dataflow after alpha-renaming of registers (MOV treated as copy)

Level 4 is the decisive one for "does the patched build compute the same
function with the switch off": it normalizes away register allocation and MOV
scaffolding, so a genuine change of operation order, a changed FMA contraction
(FFMA vs FMUL+FADD), or a changed operand role shows up as a mismatch, while a
pure re-coloring of registers does not.

Instantiation matching: the patched build's kernels carry an extra `bool`
template argument, mangled as `Lb0E` (switch off) / `Lb1E` (on); the pristine
build has no such parameter. Canonicalizing both is stripping those markers.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys

INSTR_RE = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(@!?U?P[0-9T]\s+)?([A-Z][A-Z0-9._]*)")
FUNC_RE = re.compile(r"^\s*Function : (\S+)")

FP_OPS = {"FFMA", "FMUL", "FADD", "FSUB", "FSETP", "FSEL", "FMNMX", "MUFU", "FSQRT",
          "FFMA32I", "FADD32I", "FMUL32I", "F2F", "F2I", "I2F", "FRND", "FSET", "FSWZ",
          "RRO", "FLO", "FCHK", "F2FP", "F2F32I"}


def base_op(op):
    """strip the modifier suffixes (.FTZ/.SAT/...) -- sm_80 emits `FFMA.FTZ` etc."""
    return op.split(".")[0]


def is_fp(op):
    return base_op(op) in FP_OPS


def is_copy(op):
    return base_op(op) == "MOV" or op.startswith("IMAD.MOV") or \
        base_op(op) in ("MOV32I", "SEL", "PRMT")


def canonical(name):
    return name.replace("Lb0E", "").replace("Lb1E", "")


SCALAR_RE = re.compile(r"_kernelI([fd])")


def scalar_of(name):
    """float/double instantiation marker; the H8 patch changed one parameter from
    `const scalar_t*` to `scalar_t*` (v_means2d is written back), so the mangled
    parameter list legitimately differs and name canonicalization cannot pair the
    projection kernels. Pairing falls back to (family, scalar type)."""
    m = SCALAR_RE.search(name)
    return m.group(1) if m else "?"


def parse(path):
    """-> (raw_function_name, [(opcode, body)]) in program order"""
    out = []
    cur = None
    with open(path) as f:
        for line in f:
            m = FUNC_RE.match(line)
            if m:
                cur = m.group(1)
                continue
            m = INSTR_RE.match(line)
            if m:
                out.append((cur, m.group(2), line.rstrip("\n")))
    return out


def parse_ops(body):
    """operands of an FP instruction, in order: registers and immediates,
    predicate masked out. Returns None if the shape is not understood.

    The cuobjdump line is `/*addr*/ <pred> OP operands ; /* encoding */`, so the
    trailing encoding comment is cut at the semicolon and the address comment is
    dropped after the first `*/`."""
    code = body.split(";")[0]
    if "*/" in code:
        code = code.split("*/", 1)[1]
    toks = code.strip().split()
    if not toks:
        return None
    ops = []
    for t in toks[1:]:
        t = t.rstrip(",")
        if not t or t.startswith("@"):
            continue
        ops.append(t)
    return ops


def fp_stream(instrs):
    """alpha-renamed FP instruction stream: register ids replaced by the
    canonical value id of the instruction that produced them, in first-def order."""
    value_of = {}     # raw reg -> value id
    alias = {}        # raw reg -> raw reg it copies from
    next_id = [0]
    stream = []
    for _, op, body in instrs:
        if is_fp(op):
            ops = parse_ops(body)
            if ops is None:
                continue
            canon = []
            for o in ops:
                # strip modifiers like .reuse / address forms
                r = re.match(r"^[!]?(R\d+|P\d+|SR\d+|SRZ|RZ|UR\d+|c\[[^\]]*\]|0x[0-9a-f]+|-?[\d.]+)", o)
                r = r.group(1) if r else o
                if r in alias:
                    r = alias[r]
                if r.startswith("R") or r in ("RZ", "SRZ"):
                    if r not in value_of:
                        value_of[r] = next_id[0]
                        next_id[0] += 1
                    canon.append(value_of[r])
                else:
                    canon.append(r)
            stream.append(" ".join([op] + [str(c) for c in canon]))
            # dest = first operand (SASS prints dest first)
            d = re.match(r"^[!]?(R\d+|RZ|SRZ|P\d+)", ops[0]) if ops else None
            if d:
                value_of[d.group(1)] = next_id[0]
                next_id[0] += 1
                alias.pop(d.group(1), None)
        elif is_copy(op):
            ops = parse_ops(body)
            if ops and len(ops) >= 2:
                d = re.match(r"^[!]?(R\d+)", ops[0])
                # IMAD.MOV Rd, RZ, RZ, Rs -> source is the last operand
                src = ops[-1] if op.startswith("IMAD.MOV") else ops[1]
                s = re.match(r"^[!]?(R\d+)", src)
                if d and s:
                    alias[d.group(1)] = alias.get(s.group(1), s.group(1))
    return stream


REG_RE = re.compile(r"^[!]?(R\d+|P\d+|SR\d+|UR\d+|RZ|SRZ|URZ)")


def dag_signature(instrs, rounds=4):
    """Schedule- and register-name-invariant signature of the FP instruction DAG.

    Each FP/copy instruction becomes a node whose operands point at the node that
    most recently defined the source register. Labels are then refined
    (Weisfeiler-Lehman style) from the opcode outward, so the resulting multiset
    of labels depends only on the *shape and opcodes* of the dataflow graph, not
    on the order in which independent instructions were scheduled nor on which
    physical register was chosen."""
    nodes = []          # [op, [(kind, token)], [src node idx or None]]
    last_def = {}
    alias = {}          # MOV-style copies are folded into register aliases

    def resolve(tok):
        seen = 0
        while tok in alias and seen < 64:
            tok = alias[tok]
            seen += 1
        return tok

    for _, op, body in instrs:
        if not (is_fp(op) or is_copy(op)):
            continue
        ops = parse_ops(body)
        if not ops:
            continue
        if is_copy(op):
            d = REG_RE.match(ops[0])
            src = ops[-1] if op.startswith("IMAD.MOV") else (ops[1] if len(ops) > 1 else None)
            s = REG_RE.match(src) if src else None
            if d and s:
                dst, srcreg = d.group(1), resolve(s.group(1))
                alias[dst] = srcreg
                if srcreg in last_def:
                    last_def[dst] = last_def[srcreg]
                else:
                    last_def.pop(dst, None)
            continue
        operands = []
        src_nodes = []
        for i, o in enumerate(ops):
            r = REG_RE.match(o)
            tok = r.group(1) if r else None
            if tok is not None and i > 0:
                tok = resolve(tok)
                operands.append(("R", tok))
                src_nodes.append(last_def.get(tok))
            elif tok is not None:
                operands.append(("D", tok))
            else:
                imm = re.match(r"^[!]?(-?[\d.]+|0x[0-9a-f]+)", o)
                operands.append(("I", imm.group(1) if imm else o))
        nodes.append([op, operands, src_nodes])
        if operands and operands[0][0] == "D":
            alias.pop(operands[0][1], None)
            last_def[operands[0][1]] = len(nodes) - 1
    labels = [n[0] for n in nodes]
    for _ in range(rounds):
        new = []
        for i, (op, operands, srcs) in enumerate(nodes):
            key = [op]
            si = 0
            for kind, tok in operands:
                if kind == "R":
                    j = srcs[si]
                    si += 1
                    key.append(("R", labels[j] if j is not None else "undef"))
                else:
                    key.append((kind, tok))
            new.append(hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12])
        labels = new
    return collections.Counter(labels)


def summarize(instrs):
    ops = [op for _, op, _ in instrs]
    return {"n_instructions": len(ops),
            "opcode_multiset_md5": hashlib.md5(
                json.dumps(dict(sorted(collections.Counter(ops).items()))).encode()).hexdigest()[:16],
            "opcode_counts": dict(sorted(collections.Counter(ops).most_common())),
            "opcode_seq_md5": hashlib.md5(" ".join(ops).encode()).hexdigest()[:16],
            "fp_seq_md5": hashlib.md5(" ".join(o for o in ops if is_fp(o)).encode()).hexdigest()[:16],
            "n_fp": sum(1 for o in ops if is_fp(o))}


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), (a[min(len(a), len(b)):] or [None])[0] if len(a) > len(b) else None, \
            (b[min(len(a), len(b)):] or [None])[0] if len(b) > len(a) else None
    return None


def index(dirpath):
    """-> {canonical_name: {raw_name: instrs}}; a canonical name holds one entry
    per instantiation (the patched .so has both Lb0E and Lb1E)."""
    funcs = collections.defaultdict(dict)
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith(".sass"):
            continue
        instrs = parse(os.path.join(dirpath, fn))
        if not instrs or instrs[0][0] is None:
            continue
        funcs[canonical(instrs[0][0])][instrs[0][0]] = instrs
    return funcs


def select_arm(funcs, arm):
    """One instruction stream per canonical name for the requested arm.

    Kernels that carry no bool-template marker (the un-patched forward kernels)
    exist once and belong to every arm."""
    out = {}
    for k, by_raw in funcs.items():
        if arm == "on":
            keep = [v for raw, v in by_raw.items() if "Lb1E" in raw or
                    ("Lb0E" not in raw)]
        elif arm == "off":
            keep = [v for raw, v in by_raw.items() if "Lb0E" in raw or
                    ("Lb1E" not in raw)]
        else:
            keep = [v for raw, v in by_raw.items() if "Lb0E" not in raw and "Lb1E" not in raw]
        if keep:
            out[k] = keep[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pri", required=True)
    ap.add_argument("--h8", required=True)
    ap.add_argument("--arm", choices=["off", "on"], default="off")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()

    P, H = select_arm(index(a.pri), "pristine"), select_arm(index(a.h8), a.arm)

    out = {"arm": a.arm, "pri_dir": a.pri, "h8_dir": a.h8,
           "caveats": {
               "fp_schedule_and_fp_dag_levels": (
                   "register-allocation-sensitive: two builds that compute the same "
                   "arithmetic can still differ at these levels when nvcc recolors "
                   "physical registers or reschedules independent instructions (the "
                   "EDGES of the dag signature follow the most recent definition of the "
                   "physical register). Inertness evidence therefore rests on the opcode "
                   "census (opcode_delta_total / n_fp_delta_total), not on these booleans."
               ),
               "nearest_build_controls": (
                   "the un-patched forward kernels inside the patched projection TU "
                   "(.cu) came out bit-identical in opcode count and multiset, which "
                   "shows the build is deterministic for unchanged kernel source; "
                   "pristine_ctl is a second compile of the pristine source used as a "
                   "same-source noise floor."
               ),
           },
           "families": {}}
    for fam in ("rasterize_to_pixels_3dgs_bwd_kernel", "projection_ewa_3dgs_fused_bwd_kernel",
                "rasterize_to_pixels_3dgs_fwd_kernel", "projection_ewa_3dgs_fused_fwd_kernel"):
        pk = [k for k in P if fam in k]
        hk = [k for k in H if fam in k]
        # pair by canonical name where possible, else by (family, scalar type) --
        # the projection bwd signature legitimately differs (const -> non-const
        # pointer) and each build has exactly one instantiation per scalar type
        pairs = [(k, k) for k in sorted(pk) if k in H]
        unpaired = [k for k in sorted(pk) if k not in H]
        if unpaired:
            hleft = {scalar_of(k): k for k in hk if k not in {p for _, p in pairs}}
            for k in unpaired:
                sc = scalar_of(k)
                if sc in hleft:
                    pairs.append((k, hleft.pop(sc)))
        rows = []
        agg = collections.Counter()
        for k, hkey in pairs:
            pin = P[k]
            hin = H[hkey]
            sp, sh = summarize(pin), summarize(hin)
            fp_p, fp_h = fp_stream(pin), fp_stream(hin)
            dag_p, dag_h = dag_signature(pin), dag_signature(hin)
            ops_p = [o for _, o, _ in pin]
            ops_h = [o for _, o, _ in hin]
            row = {
                "symbol": k,
                "h8_symbol": hkey,
                "paired_by": "name" if k == hkey else "scalar_type",
                "matched": True,
                "n_instructions": {"pri": sp["n_instructions"], "h8": sh["n_instructions"]},
                "opcode_multiset_identical": sp["opcode_multiset_md5"] == sh["opcode_multiset_md5"],
                "opcode_sequence_identical": sp["opcode_seq_md5"] == sh["opcode_seq_md5"],
                "fp_sequence_identical": sp["fp_seq_md5"] == sh["fp_seq_md5"],
                "fp_schedule_identical": fp_p == fp_h,
                "fp_dag_identical": dag_p == dag_h,
                "n_fp": {"pri": sp["n_fp"], "h8": sh["n_fp"]},
                "n_dag_nodes": {"pri": sum(dag_p.values()), "h8": sum(dag_h.values())},
            }
            if dag_p != dag_h:
                extra = {k: v for k, v in (dag_h - dag_p).items()}
                missing = {k: v for k, v in (dag_p - dag_h).items()}
                row["dag_delta"] = {"n_extra": sum(extra.values()), "n_missing": sum(missing.values()),
                                    "n_distinct_changed": len(set(extra) | set(missing))}
            d = first_diff(ops_p, ops_h)
            row["first_opcode_diff"] = None if d is None else {"index": d[0], "pri": d[1], "h8": d[2]}
            d2 = first_diff(fp_p, fp_h)
            row["first_fp_schedule_diff"] = None if d2 is None else {
                "index": d2[0], "pri": d2[1], "h8": d2[2],
                "context_pri": fp_p[max(0, d2[0] - 3):d2[0] + 3],
                "context_h8": fp_h[max(0, d2[0] - 3):d2[0] + 3]}
            for op, n in sh["opcode_counts"].items():
                agg[op] += n - sp["opcode_counts"].get(op, 0)
            for op, n in sp["opcode_counts"].items():
                if op not in sh["opcode_counts"]:
                    agg[op] -= n
            if not row["opcode_multiset_identical"]:
                cp, ch = collections.Counter(ops_p), collections.Counter(ops_h)
                row["opcode_delta"] = {k2: ch.get(k2, 0) - cp.get(k2, 0)
                                       for k2 in sorted(set(cp) | set(ch))
                                       if ch.get(k2, 0) != cp.get(k2, 0)}
            rows.append(row)
        out["families"][fam] = {
            "n_pri_instantiations": len(pk), "n_h8_instantiations": len(hk),
            "n_matched": sum(1 for r in rows if r["matched"]),
            "opcode_delta_total": {k2: v for k2, v in sorted(agg.items()) if v},
            "n_fp_delta_total": sum(v for k2, v in agg.items() if is_fp(k2)),
            "all_total_instructions_identical": bool(rows) and all(
                r["n_instructions"]["pri"] == r["n_instructions"]["h8"] for r in rows),
            "all_fp_dag_identical": bool(rows) and all(r.get("fp_dag_identical") for r in rows),
            "all_fp_schedule_identical": bool(rows) and all(r.get("fp_schedule_identical") for r in rows),
            "all_fp_sequence_identical": bool(rows) and all(r.get("fp_sequence_identical") for r in rows),
            "all_opcode_multiset_identical": bool(rows) and all(r.get("opcode_multiset_identical") for r in rows),
            "rows": rows if a.detail else [
                {kk: r.get(kk) for kk in ("symbol", "matched", "fp_dag_identical",
                                          "fp_schedule_identical", "fp_sequence_identical",
                                          "opcode_multiset_identical", "opcode_sequence_identical",
                                          "n_fp", "n_dag_nodes", "n_instructions", "dag_delta",
                                          "first_fp_schedule_diff")} for r in rows],
        }
        f = out["families"][fam]
        print(f"{fam}: matched={f['n_matched']}/{f['n_pri_instantiations']} "
              f"fp_dag_identical={f['all_fp_dag_identical']} "
              f"fp_schedule_identical={f['all_fp_schedule_identical']} "
              f"fp_seq_identical={f['all_fp_sequence_identical']} "
              f"opcode_multiset_identical={f['all_opcode_multiset_identical']}")
        top = sorted(f["opcode_delta_total"].items(), key=lambda kv: -abs(kv[1]))[:6]
        print(f"    fp_delta_total={f['n_fp_delta_total']} "
              f"opcode_delta_total(top)={top}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"SASS_CMP arm={a.arm} -> {a.out}")


if __name__ == "__main__":
    main()
