p = "cvt0_report.py"
s = open(p, encoding="utf-8").read()

old_block = '''# gate table (thresholds from cvt0_analyze.py strong/promote)
gates = [
    ("T_oracle_median > 0", "strong:>0 promote:>0"),
    ("G_random_median >= 0.60 / 0.35", "strong>=0.60 promote>=0.35"),
    ("G_hard_median >= 0.35 / 0.20", "strong>=0.35 promote>=0.20"),
    ("G_true_median >= 0.40 / 0.20", "strong>=0.40 promote>=0.20"),
    ("H_oracle_positive_count ==3 / >=2", "strong==3 promote>=2"),
    ("F_pos_median >= 0.75 / 0.65", "strong>=0.75 promote>=0.65"),
    ("rho_TH_median >= 0.40 / 0.20", "strong>=0.40 promote>=0.20"),
]
gate_rows = []
for name, thr in gates:
    vals = []
    for c in cells:
        s = summ["cells"][c]
        vals.append("P" if s["complete"] and s["nan_inf_count"] == 0 else "-")
    gate_rows.append(f"| {name} | {thr} | " + " | ".join(vals) + " |")'''

new_block = '''# gate table: evaluate ACTUAL pass/fail per cell vs frozen thresholds.
# Returns (strong_ok, promote_ok) per cell. nan/None -> fail.
def gpass(s):
    def ge(v, t):
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and v >= t
    def gt0(v):
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0
    complete = s["complete"] and s["nan_inf_count"] == 0
    strong = complete and gt0(s["T_oracle_median"]) and \\
        ge(s["G_random_median"], 0.60) and ge(s["G_hard_median"], 0.35) and \\
        ge(s["G_true_median"], 0.40) and s["H_oracle_positive_count"] == 3 and \\
        ge(s["F_pos_median"], 0.75) and ge(s["rho_TH_median"], 0.40)
    promote = complete and gt0(s["T_oracle_median"]) and \\
        ge(s["G_random_median"], 0.35) and ge(s["G_hard_median"], 0.20) and \\
        ge(s["G_true_median"], 0.20) and s["H_oracle_positive_count"] >= 2 and \\
        ge(s["F_pos_median"], 0.65) and ge(s["rho_TH_median"], 0.20)
    return strong, promote

gates = [
    ("T_oracle_median > 0", ">0", ">0"),
    ("G_random_median", ">=0.60", ">=0.35"),
    ("G_hard_median (MANDATORY)", ">=0.35", ">=0.20"),
    ("G_true_median", ">=0.40", ">=0.20"),
    ("H_oracle_positive_count", "==3", ">=2"),
    ("F_pos_median", ">=0.75", ">=0.65"),
    ("rho_TH_median", ">=0.40", ">=0.20"),
]
gate_rows = []
for name, sthr, pthr in gates:
    scells = []; pcells = []
    for c in cells:
        s = summ["cells"][c]
        strong, promote = gpass(s)
        scells.append("Y" if strong else ".")
        pcells.append("Y" if promote else ".")
    gate_rows.append(f"| {name} | {sthr} | {pthr} | " +
                     " | ".join(scells) + " | " + " | ".join(pcells) + " |")'''

assert old_block in s, "gate block not found"
s = s.replace(old_block, new_block)

# ensure math imported
if "\nimport math\n" not in s and "import math" not in s:
    s = s.replace("import json, os, hashlib, sys", "import json, os, hashlib, sys, math")

# update the gate table header to include strong/promote columns
old_hdr = '''| gate | threshold (strong / promote) | room10 | room15 | room20 | bike10 | bike15 | bike20 |
|------|------|--------|--------|--------|--------|--------|--------|'''
new_hdr = '''| gate | strong thr | promote thr | r10 | r15 | r20 | b10 | b15 | b20 | (strong) |
|------|------|------|------|------|------|------|------|------|'''
assert old_hdr in s, "gate header not found"
s = s.replace(old_hdr, new_hdr)

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched gate table to real pass/fail")
