"""Compare two trainer checkpoints (C0 baseline vs VOS arm) attr-by-attr.

Reports bitwise equality and max|diff| for every parameter and every optimizer
m/v tensor, plus step counters, so the Phase-8 functional gate is explicit.
"""
import argparse
import json

import torch


def flat(t):
    return t.detach().float().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference ckpt (C0)")
    ap.add_argument("--b", required=True, help="candidate ckpt (VOS)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A = torch.load(args.a, map_location="cpu", weights_only=False)
    B = torch.load(args.b, map_location="cpu", weights_only=False)

    report = {"a": args.a, "b": args.b, "attrs": {}, "all_bitwise": True}
    names = ["xyz", "shs", "opacity", "scaling", "rotation"]
    for n in names:
        ta = A["model_params"][n] if "model_params" in A else A[n]
        tb = B["model_params"][n] if "model_params" in B else B[n]
        row = {"shape": list(ta.shape)}
        if ta.shape != tb.shape:
            row["shape_mismatch"] = [list(ta.shape), list(tb.shape)]
            report["all_bitwise"] = False
            report["attrs"][n] = row
            continue
        fa, fb = flat(ta), flat(tb)
        bw = bool(torch.equal(fa, fb))
        row["theta_bitwise"] = bw
        row["theta_max_abs"] = float((fa - fb).abs().max())
        report["all_bitwise"] &= bw

        sa = A["optimizer_state_dict"]["state"][A["optimizer_state_dict"]
                                                ["param_groups"].index(
            next(g for g in A["optimizer_state_dict"]["param_groups"]
                 if g["name"] == n)["params"][0])] if False else None
        report["attrs"][n] = row

    # optimizer state comparison: iterate param_groups by name
    for tag, D in (("a", A), ("b", B)):
        od = D.get("optimizer_state_dict")
        if od is None:
            continue
        groups = od["param_groups"]
        # map param id -> group name
        pid2name = {}
        for g in groups:
            for pid in g["params"]:
                pid2name[pid] = g.get("name", "?")
        for pid, st in od["state"].items():
            name = pid2name.get(pid, f"p{pid}")
            if name not in report["attrs"]:
                continue
            row = report["attrs"][name]
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st and f"{key}_{tag}_bitwise" not in row:
                    row[f"{key}_{tag}"] = list(st[key].shape)
    # direct m/v bitwise between the two ckpts
    def state_by_name(D):
        od = D["optimizer_state_dict"]
        pid2name = {}
        for g in od["param_groups"]:
            for pid in g["params"]:
                pid2name[pid] = g.get("name", "?")
        return {pid2name[pid]: st for pid, st in od["state"].items()}

    sa, sb = state_by_name(A), state_by_name(B)
    for n in names:
        if n not in sa or n not in sb:
            continue
        row = report["attrs"][n]
        if "step" in sa[n] and "step" in sb[n]:
            row["step_a"] = float(sa[n]["step"])
            row["step_b"] = float(sb[n]["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            if key in sa[n] and key in sb[n]:
                fa, fb = flat(sa[n][key]), flat(sb[n][key])
                if fa.shape != fb.shape:
                    row[f"{key}_shape_mismatch"] = True
                    report["all_bitwise"] = False
                    continue
                bw = bool(torch.equal(fa, fb))
                row[f"{key}_bitwise"] = bw
                row[f"{key}_max_abs"] = float((fa - fb).abs().max())
                report["all_bitwise"] &= bw

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print("ALL_BITWISE" if report["all_bitwise"] else "NOT_ALL_BITWISE")


if __name__ == "__main__":
    main()
