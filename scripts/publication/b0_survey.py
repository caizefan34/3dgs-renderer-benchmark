import subprocess

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (r.stdout + r.stderr).strip()

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
print("== locate dataset.py (GTDataset) ==")
print(run(f"grep -rln 'class GTDataset' {REPO}/baseline {REPO}/scripts {REPO}/src 2>/dev/null | head -5"))
print(run(f"ls {REPO}/scripts/epic05/phase7/ 2>/dev/null | head -15"))

print("\n== gaussian_model.py: class + key methods ==")
print(run(f"grep -n 'def |class ' {REPO}/baseline/reference_v1/gaussian_model.py"))
print(run(f"grep -n 'def \\|class ' {REPO}/baseline/reference_v1/gaussian_model.py"))

print("\n== add_densification_stats body ==")
import re
txt = open(f"{REPO}/baseline/reference_v1/gaussian_model.py").read()
for name in ("add_densification_stats", "densify_and_prune", "densify_and_clone", "densify_and_split", "prune_points", "reset_opacity", "training_setup", "create_from_pcd", "update_learning_rate"):
    m = re.search(rf"def {name}\(.*?\n(?=    def |class |\Z)", txt, re.S)
    if m:
        body = m.group(0)
        print(f"---- {name} ({len(body)} chars) ----")
        print(body[:1500])
        print()
