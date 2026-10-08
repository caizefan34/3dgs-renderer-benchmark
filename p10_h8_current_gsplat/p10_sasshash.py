#!/usr/bin/env python3
"""Normalised SASS hash comparison: pristine binary vs the patched binary's
switch-off (H8_MR=false) instantiation.

The inert control compares the pristine binary against the patched binary with
the switch off. Both run baseline arithmetic; the only way a *deterministic*
elementwise offset can appear is a codegen difference between the two
compilations. This tool settles that question directly from the cubins:

  hash    --so SO --arm pristine|off|on --out F
          streams cuobjdump -sass restricted to the four kernel families of
          interest; per function stores a SHA-256 of the address-stripped
          instruction text (exact code opcodes + registers + predicates),
          plus a res-usage record.

  compare --pri F --h8 F --out F
          canonicalises the mangled names (strips the bool template argument
          Lb0E / Lb1E the patch added), then matches every pristine function
          against the patched switch-off function and reports hash equality,
          instruction counts and the first differing instruction with context.
          Also compares the untouched forward kernels as a toolchain control
          (unchanged code must compile identically in both trees).
"""
import argparse
import collections
import hashlib
import json
import os
import re
import subprocess

KERNELS = ["rasterize_to_pixels_3dgs_bwd_kernel", "projection_ewa_3dgs_fused_bwd_kernel",
           "rasterize_to_pixels_3dgs_fwd_kernel", "projection_ewa_3dgs_fused_fwd_kernel"]

RES_FUNC_RE = re.compile(r"\s*Function (\S+):\s*$")
SASS_FUNC_RE = re.compile(r"\s*Function : (\S+)")
USAGE_RE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")
INSTR_RE = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(.*)$")
# control-flow targets are absolute code addresses: the patched .so carries two
# instantiations of every backward kernel, so symbols shift and *only* the
# targets of BRA/CALL differ for otherwise identical code. Mask them; the
# opcode, predicates, registers and all other operands stay semantic.
TGT_OPS = {"BRA", "CALL", "JMP", "BRX", "JCAL", "PREEXIT"}
TGT_RE = re.compile(r"^0x[0-9a-f]+;?$")


def norm_body(body):
    toks = body.split()
    for i, t in enumerate(toks):
        if t.startswith("@"):
            continue
        if t.split(".")[0] in TGT_OPS:
            for j in range(i + 1, len(toks)):
                if TGT_RE.match(toks[j]):
                    toks[j] = "<TGT>"
        break
    return " ".join(toks)


def cuobjdump():
    p = "/usr/local/cuda/bin/cuobjdump"
    return p if os.path.exists(p) else "cuobjdump"


def select_names(so):
    res = subprocess.run([cuobjdump(), "-arch", "sm_80", "-res-usage", so],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cuobjdump -res-usage failed: {res.stderr[-2000:]}")
    names, usage, cur = [], {}, None
    for ln in res.stdout.splitlines():
        m = RES_FUNC_RE.match(ln)
        if m:
            cur = m.group(1)
            continue
        m = USAGE_RE.search(ln)
        if m and cur:
            usage[cur] = {"regs": int(m.group(1)), "stack": int(m.group(2)),
                          "shared": int(m.group(3)), "local": int(m.group(4))}
    for ln in res.stdout.splitlines():
        m = RES_FUNC_RE.match(ln)
        if m and any(k in m.group(1) for k in KERNELS):
            names.append(m.group(1))
    return names, usage


def canonical(name):
    """Drop the bool template argument the patch added (Itanium-style Lb0E/Lb1E)."""
    return name.replace("Lb0E", "").replace("Lb1E", "")


def cmd_hash(args):
    names, usage = select_names(args.so)
    if not names:
        raise RuntimeError("no matching kernels in .so")
    cmd = [cuobjdump(), "-arch", "sm_80", "-sass"]
    for n in names:
        cmd += ["-fun", n]
    cmd.append(args.so)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    funcs = collections.OrderedDict()
    cur = None
    for line in p.stdout:
        m = SASS_FUNC_RE.match(line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, [])
            continue
        if cur is None:
            continue
        m = INSTR_RE.match(line)
        if m:
            body = norm_body(m.group(1).split("/*")[0].strip())
            if body:
                funcs[cur].append(body)
    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"cuobjdump -sass failed rc={rc}")

    out = {"so": args.so, "arm": args.arm, "functions": {}, "n_matched_names": len(names)}
    for f, instrs in funcs.items():
        text = "\n".join(instrs).encode()
        out["functions"][f] = {
            "canonical": canonical(f),
            "n_instructions": len(instrs),
            "sha256": hashlib.sha256(text).hexdigest(),
            "usage": usage.get(f),
            "instrs": instrs if args.keep_instrs else None,
        }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"HASH {args.arm} funcs={len(funcs)} -> {args.out}")


def _load(path):
    with open(path) as f:
        return json.load(f)


MARKER = {"off": "Lb0E", "on": "Lb1E"}


def _index(functions, arm):
    """canonical name -> (mangled, record), selecting the arm's instantiation.

    The patched .so contains both Lb0E (off) and Lb1E (on) instantiations, which
    canonicalise to the same name; the pristine .so contains neither marker.
    """
    by_c, n_alt = {}, 0
    for f, d in functions.items():
        by_c.setdefault(canonical(f), []).append((f, d))
    chosen = {}
    want = MARKER.get(arm)
    for c, lst in by_c.items():
        if want:
            match = [(f, d) for f, d in lst if want in f]
        else:
            match = [(f, d) for f, d in lst if not any(m in f for m in MARKER.values())]
        if not match:  # marker absent -> single instantiation, nothing to pick
            match = lst
        chosen[c] = match[0]
        n_alt += len(lst) - len(match) if len(lst) > len(match) else 0
    return chosen, n_alt


def cmd_compare(args):
    pri, h8 = _load(args.pri), _load(args.h8)
    pri_by_c, pri_alt = _index(pri["functions"], pri["arm"])
    h8_by_c, h8_alt = _index(h8["functions"], h8["arm"])

    def family(name):
        for k in KERNELS:
            if k in name:
                return k.replace("_kernel", "")
        return "?"

    rows, only_pri, only_h8 = [], [], []
    for c, plist in sorted(pri_by_c.items()):
        hl = h8_by_c.get(c)
        if not hl:
            only_pri.append((family(c), c))
            continue
        pf, pd = plist
        hf, hd = hl
        row = {"family": family(c), "canonical": c,
               "n_instructions": {"pri": pd["n_instructions"], "h8": hd["n_instructions"]},
               "regs": {"pri": (pd["usage"] or {}).get("regs"), "h8": (hd["usage"] or {}).get("regs")},
               "spill_local": {"pri": (pd["usage"] or {}).get("local"), "h8": (hd["usage"] or {}).get("local")},
               "identical": pd["sha256"] == hd["sha256"]}
        if not row["identical"] and pd["instrs"] and hd["instrs"]:
            a, b = pd["instrs"], hd["instrs"]
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            ndiff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
            row["first_diff_index"] = first
            row["n_differing_instructions"] = ndiff
            row["diff_context"] = [
                {"i": i, "pri": a[i] if i < len(a) else None, "h8": b[i] if i < len(b) else None}
                for i in range(max(0, first - 2), min(first + 5, max(len(a), len(b))))]
        rows.append(row)
    for c, hl in sorted(h8_by_c.items()):
        if c not in pri_by_c:
            only_h8.append((family(c), c))

    out = {"pri_so": pri["so"], "pri_arm": pri["arm"], "h8_so": h8["so"], "h8_arm": h8["arm"],
           "n_pri_funcs": len(pri["functions"]), "n_h8_funcs": len(h8["functions"]),
           "n_alt_instantiations_skipped": {"pri": pri_alt, "h8": h8_alt},
           "matched": rows, "only_pri": [list(x) for x in only_pri],
           "only_h8": [list(x) for x in only_h8],
           "summary": {}}
    by_fam = collections.defaultdict(lambda: {"n": 0, "identical": 0})
    for r in rows:
        by_fam[r["family"]]["n"] += 1
        by_fam[r["family"]]["identical"] += int(r["identical"])
    out["summary"] = {k: v for k, v in sorted(by_fam.items())}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"SASSHASH compare: {len(rows)} matched instantiations, {len(only_pri)} only-pri, "
          f"{len(only_h8)} only-h8 -> {args.out}")
    for k, v in out["summary"].items():
        print(f"  {k}: identical {v['identical']}/{v['n']}")
    for r in rows:
        if not r["identical"]:
            print(f"  DIFF {r['family']} n_inst {r['n_instructions']} regs {r['regs']} "
                  f"first_diff@{r.get('first_diff_index')} ndiff={r.get('n_differing_instructions')}")
            for d in (r.get("diff_context") or [])[:4]:
                print(f"      [{d['i']}] pri: {d['pri']}")
                print(f"      [{d['i']}] h8 : {d['h8']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash")
    h.add_argument("--so", required=True)
    h.add_argument("--arm", required=True)
    h.add_argument("--keep-instrs", action="store_true",
                   help="store the normalised instruction text (for the compare diff)")
    h.add_argument("--out", required=True)
    h.set_defaults(func=cmd_hash)
    c = sub.add_parser("compare")
    c.add_argument("--pri", required=True)
    c.add_argument("--h8", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
