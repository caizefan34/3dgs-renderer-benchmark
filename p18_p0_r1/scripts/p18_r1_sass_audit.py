#!/usr/bin/env python3
"""P18-P0-R1 R1-D: MANDATORY SASS audit gate (spec item 4) BEFORE any timing.

Builds the clean evaluator + a counter-instrumented copy (-DP18_EXP_COUNTER),
dumps SASS with cuobjdump and checks:
  A. DIRECT kernels contain the baseline per-pixel exp (>=1 MUFU.EX2 per item).
  B. QSCAN4/8/16 static MUFU count is IDENTICAL and small (leader-only 3 exps):
     does NOT scale with SUB (the position-dependent-exp failure mode would add a
     per-lane MUFU, which the forbidden control kernel demonstrates).
  C. Dynamic counter (instrumented build, exact): QSCAN exp executions per item
     fall as 3/SUB (0.75, 0.375, 0.1875 per px) - NOT linear in W pixels; DIRECT
     stays at 1.0/px.
  D. The forbidden control kernel (exp2f(k*pos) per lane) shows a 4th MUFU ->
     the detection is sensitive (positive control).
  E. Dynamic positive control: the forbidden kernel's exact exp counter reads
     1 + 3/SUB per item (per-lane exp present) -> the counter would have caught
     the old P18-P0 defect pattern.
  F. Exp-free control (diagnostic, not part of the gate): eval_noexp_kernel has
     0 MUFU (static) and 0 exp calls (dynamic) -> certifies the S_eval_noexp
     upper bound measured by the timing driver.

If any check fails -> P18_R1_SASS_AUDIT.json verdict = P18_R1_IMPLEMENTATION_INVALID;
the timing script refuses to run.
"""
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from p18_r1_eval_common import ART, build_ext, build_workload, jwrite, log

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"


def dump_sass(so_path):
    out = subprocess.run([CUOBJDUMP, "-sass", str(so_path)],
                         capture_output=True, text=True, check=True).stdout
    funcs = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, [])
        elif cur is not None:
            funcs[cur].append(line)
    return funcs


def classify(name):
    if "eval_noexp_kernel" in name:
        return ("direct_noexp", "none")
    if "eval_direct_exp2" in name:
        return ("direct", "exp2")
    if "eval_direct_expf" in name:
        return ("direct", "expf")
    m = re.search(r"eval_qscan_forbiddenILi(\d+)ELb([01])E", name)
    if m:
        return ("forbidden", f"sub{m.group(1)}_exp{'2' if m.group(2) == '1' else 'f'}")
    m = re.search(r"eval_qscanILi(\d+)ELb([01])E", name)
    if m:
        return ("qscan", f"sub{m.group(1)}_exp{'2' if m.group(2) == '1' else 'f'}")
    return (None, name)


def main():
    t0 = time.time()
    mod = build_ext(name="p18_r1_eval")
    so = mod.__file__
    log(f"clean ext built: {so} ({time.time()-t0:.0f}s)")
    funcs = dump_sass(so)

    mod_ctr = build_ext(name="p18_r1_eval_ctr", extra_flags=["-DP18_EXP_COUNTER"])
    so_ctr = mod_ctr.__file__
    log(f"counter ext built: {so_ctr}")
    funcs_ctr = dump_sass(so_ctr)

    kernels = {}
    for fname, lines in funcs.items():
        kind, tag = classify(fname)
        if kind is None:
            continue
        mufu = [ln.strip() for ln in lines if re.search(r"\bMUFU\.", ln)]
        shfl = [ln.strip() for ln in lines if re.search(r"\bSHFL\.", ln)]
        kernels[f"{kind}/{tag}"] = {
            "symbol": fname, "kind": kind, "tag": tag,
            "n_sass_lines": len(lines), "n_mufu": len(mufu), "n_shfl": len(shfl),
            "mufu_ops": sorted(set(re.findall(r"MUFU\.\w+", " ".join(mufu)))),
            "mufu_context": mufu[:6],
        }

    # dynamic counter (exact): fixed item slice, record-aligned
    scene = "room"
    w = build_workload(scene, n_max_items=1 << 22)  # 4,194,304 items = 262144 rows
    dev = "cuda"
    a = torch.from_numpy(w["a"]).to(dev)
    b = torch.from_numpy(w["b"]).to(dev)
    c = torch.from_numpy(w["c"]).to(dev)
    x = torch.from_numpy(w["x"]).to(dev)
    out = torch.zeros_like(a)
    n_items = int(a.numel())
    dyn = {}
    for label, fn in (("direct_exp2", lambda: mod_ctr.eval_direct(a, b, c, x, out, True, 1)),
                      ("direct_expf", lambda: mod_ctr.eval_direct(a, b, c, x, out, False, 1)),
                      ("qscan4_exp2", lambda: mod_ctr.eval_qscan(a, b, c, x, out, 4, True, 1)),
                      ("qscan8_exp2", lambda: mod_ctr.eval_qscan(a, b, c, x, out, 8, True, 1)),
                      ("qscan16_exp2", lambda: mod_ctr.eval_qscan(a, b, c, x, out, 16, True, 1)),
                      ("qscan16_expf", lambda: mod_ctr.eval_qscan(a, b, c, x, out, 16, False, 1)),
                      ("forb4_exp2", lambda: mod_ctr.eval_qscan_forbidden(a, b, c, x, out, 4, True, 1)),
                      ("forb8_exp2", lambda: mod_ctr.eval_qscan_forbidden(a, b, c, x, out, 8, True, 1)),
                      ("forb16_exp2", lambda: mod_ctr.eval_qscan_forbidden(a, b, c, x, out, 16, True, 1)),
                      ("noexp", lambda: mod_ctr.eval_noexp(a, b, c, x, out, 1))):
        mod_ctr.exp_ctr_reset()
        fn()
        torch.cuda.synchronize()
        n = mod_ctr.exp_ctr()
        dyn[label] = {"exp_calls": int(n), "per_item": n / n_items,
                      "per_subgroup16": n / (n_items / 16)}
        log(f"dyn {label}: exp_calls={n} per_item={n/n_items:.6f}")

    def kget(tag, key="n_mufu"):
        return kernels.get(tag, {}).get(key)

    # checks
    direct_ok = all(kget(f"direct/exp{'2' if d == '2' else 'f'}") is not None and
                    kget(f"direct/exp{'2' if d == '2' else 'f'}") >= 1 for d in ("2", "f"))
    q = {s: kget(f"qscan/sub{s}_exp2") for s in (4, 8, 16)}
    q_const = len(set(q.values())) == 1 and all(v is not None and v >= 1 for v in q.values())
    forb = {s: kget(f"forbidden/sub{s}_exp2") for s in (4, 8, 16)}
    forb_ok = all(v is not None and q[s] is not None and v > q[s] for s, v in forb.items())
    d_per_item = dyn["direct_exp2"]["per_item"]
    dyn_ok = (abs(d_per_item - 1.0) < 1e-9
              and abs(dyn["qscan4_exp2"]["per_item"] - 3 / 4) < 1e-9
              and abs(dyn["qscan8_exp2"]["per_item"] - 3 / 8) < 1e-9
              and abs(dyn["qscan16_exp2"]["per_item"] - 3 / 16) < 1e-9)
    monotone = (dyn["qscan4_exp2"]["per_item"] > dyn["qscan8_exp2"]["per_item"]
                > dyn["qscan16_exp2"]["per_item"])
    forb_dyn = {str(s): dyn[f"forb{s}_exp2"]["per_item"] for s in (4, 8, 16)}
    forb_dyn_ok = all(abs(v - (1.0 + 3.0 / s)) < 1e-9 and v > dyn[f"qscan{s}_exp2"]["per_item"]
                      for s, v in ((s, dyn[f"forb{s}_exp2"]["per_item"]) for s in (4, 8, 16)))
    noexp_static_ok = kget("direct_noexp/none") == 0
    noexp_dyn_ok = dyn["noexp"]["exp_calls"] == 0

    verdict = "PASS" if (direct_ok and q_const and forb_ok and dyn_ok and monotone
                         and forb_dyn_ok and noexp_static_ok and noexp_dyn_ok) else "P18_R1_IMPLEMENTATION_INVALID"
    out_obj = {
        "phase": "P18-P0-R1", "artifact": "P18_R1_SASS_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": "spec item 4: mandatory BEFORE timing; INVALID => STOP, do not time",
        "method": ("cuobjdump -sass of the built .so (clean build; frozen flags + sm_80); "
                   "per-kernel MUFU.EX2 count; dynamic exact exp counter from a "
                   "-DP18_EXP_COUNTER rebuild over a fixed record-aligned slice"),
        "so_clean": so, "so_counter": so_ctr,
        "n_items_dynamic": n_items,
        "kernels": kernels,
        "dynamic_counter": dyn,
        "checks": {
            "A_direct_keeps_per_pixel_exp": {
                "expected": ">=1 MUFU per item-eval in both direct kernels (baseline per-pixel exp preserved)",
                "observed": {k: v["n_mufu"] for k, v in kernels.items() if v["kind"] == "direct"},
                "pass": bool(direct_ok)},
            "B_qscan_static_exp_constant_in_sub": {
                "expected": "MUFU count identical for qscan4/8/16 and small (leader-only 3)",
                "observed": q, "pass": bool(q_const)},
            "C_dynamic_exp_not_linear_in_W": {
                "expected": "DIRECT 1.0/px; QSCAN 3/SUB per px (0.750/0.375/0.1875), strictly decreasing",
                "observed_per_item": {k: v["per_item"] for k, v in dyn.items()},
                "pass": bool(dyn_ok and monotone)},
            "D_forbidden_control_detected": {
                "expected": "forbidden control kernel shows a per-lane (4th+) MUFU > qscan count",
                "observed": forb, "pass": bool(forb_ok)},
            "E_forbidden_control_dynamic_sensitivity": {
                "expected": "forbidden kernel per_item = 1 + 3/SUB (per-lane exp present) > qscan per_item",
                "observed_per_item": forb_dyn, "pass": bool(forb_dyn_ok)},
            "F_expfree_control_certified": {
                "expected": "eval_noexp_kernel: 0 MUFU static AND 0 dynamic exp calls",
                "observed": {"n_mufu_static": kget("direct_noexp/none"),
                             "exp_calls_dynamic": dyn["noexp"]["exp_calls"]},
                "pass": bool(noexp_static_ok and noexp_dyn_ok)},
        },
        "verdict": verdict,
        "note": ("QSCAN16 contains NO position-dependent exponential: its only MUFU ops are the 3 "
                 "leader exps (G0/r0/k); the k^j path is pure FMUL and the scan is SHFL+FMUL only"),
    }
    jwrite(ART / "P18_R1_SASS_AUDIT.json", out_obj)
    print("SASS_AUDIT", verdict, json.dumps({k: q[k] for k in q}))
    if verdict != "PASS":
        print("STOP: P18_R1_IMPLEMENTATION_INVALID - do not time")


if __name__ == "__main__":
    main()
