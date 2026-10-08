#!/usr/bin/env python3
"""Read-only dump of final artifacts needed to fill P16_REPORT placeholders."""
import json
import os

D = "/root/epic05_p16rb"
A = os.path.join(D, "artifacts")
METHODS = ["FasterGS", "FasterGSFused", "FastGS", "SADGS"]
SCENES = ["room", "bicycle", "garden"]


def J(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


print("=== VERDICT ===")
v = J(os.path.join(A, "P16_VERDICT.json"))
print("verdict:", v["verdict"])
print("strong_methods:", v["strong_methods"])
print("methods:", json.dumps(v["methods"]))
print("scene_pareto:", json.dumps(v["scene_pareto_vs_C0"]))
print("census_classes:", json.dumps(v["census_classes"]))
print("ttq:", json.dumps(v["ttq"]))

print()
print("=== CENSUS ROWS ===")
c = J(os.path.join(A, "P16_BOTTLENECK_CENSUS.json"))
if isinstance(c, dict):
    print("top keys:", list(c.keys()))
    rows = c.get("rows") or c.get("census") or []
else:
    rows = c
for r in rows:
    print("---")
    if isinstance(r, dict):
        for k, val in r.items():
            s = json.dumps(val)
            print("  %s: %s%s" % (k, s[:400], "..." if len(s) > 400 else ""))

print()
print("=== CURVE SUMMARIES vs BOX ===")
bt = J(os.path.join(A, "P16_BASELINE_TABLE.json"))
print("baseline top keys:", [k for k in bt.keys() if k != "table"])
print("method_summary:", json.dumps(bt.get("methods")))
print("admissible_cells:", json.dumps(
    [{"m": r["method"], "s": r["scene"], "adm": r.get("quality_admissible"),
      "sp": r.get("speedup_vs_C0")} for r in bt["table"] if r["method"] != "P12_C0"]))
c0 = {}
for r in bt["table"]:
    if r["method"] == "P12_C0":
        c0[r["scene"]] = r
for m in METHODS:
    for s in SCENES:
        p = os.path.join(D, "curves", m, s + ".json")
        if not os.path.exists(p):
            print("%s/%s: MISSING" % (m, s))
            continue
        cu = J(p)
        pts = cu.get("points") or cu.get("curve") or []
        ref = c0[s]
        pe = ref["psnr"] - 0.10
        se = ref["ssim"] - 0.003
        le = ref["lpips"] + 0.005
        first_in = None
        best = None
        fpts = []
        for pt in pts:
            mm = pt.get("metrics") or pt
            ps, ss, ls = mm.get("psnr"), mm.get("ssim"), mm.get("lpips")
            x = pt.get("x") or pt.get("x_s") or pt.get("wall_s")
            if ps is None:
                continue
            fpts.append((x, ps, ss, ls))
            if best is None or ps > best[1]:
                best = (x, ps, ss, ls)
            if first_in is None and ps >= pe and (ss or 0) >= se and (ls or 9) <= le:
                first_in = (x, ps, ss, ls)
        fin = fpts[-1] if fpts else None
        print("%s/%s n=%d box(%.4f/%.4f/%.4f)" % (m, s, len(fpts), pe, se, le))
        print("   final: %s" % (json.dumps(fin),))
        print("   best_psnr: %s" % (json.dumps(best),))
        print("   first_in_box: %s" % (json.dumps(first_in),))
