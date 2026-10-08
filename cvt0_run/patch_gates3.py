p = "cvt0_report.py"
s = open(p, encoding="utf-8").read()

old = '''gates = [
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
    gate_rows.append(f"| {name} ({sthr}/{pthr}) | " +
                     " | ".join(scells) + " |")'''

new = '''# each gate: (label, key, strong_thr, promote_thr, mode)  mode: ge/eq/gt0
gates = [
    ("T_oracle_median>0", "T_oracle_median", 0.0, 0.0, "gt0"),
    ("G_random_median", "G_random_median", 0.60, 0.35, "ge"),
    ("G_hard_median (MANDATORY)", "G_hard_median", 0.35, 0.20, "ge"),
    ("G_true_median", "G_true_median", 0.40, 0.20, "ge"),
    ("H_oracle_positive_count", "H_oracle_positive_count", 3, 2, "ge"),
    ("F_pos_median", "F_pos_median", 0.75, 0.65, "ge"),
    ("rho_TH_median", "rho_TH_median", 0.40, 0.20, "ge"),
]
def ok(v, thr, mode):
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    if mode == "gt0":
        return v > 0.0
    if mode == "eq":
        return v == thr
    return v >= thr
gate_rows = []
for name, key, sthr, pthr, mode in gates:
    scells = []
    for c in cells:
        s = summ["cells"][c]
        complete = s["complete"] and s["nan_inf_count"] == 0
        scells.append("Y" if (complete and ok(s[key], sthr, mode)) else ".")
    gate_rows.append(f"| {name} (strong>={sthr}) | " + " | ".join(scells) + " |")'''

assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched per-gate evaluation")
