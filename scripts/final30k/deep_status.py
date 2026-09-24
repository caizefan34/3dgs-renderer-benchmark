import re, subprocess, os

dirs = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
log = open(f"{dirs}/scheduler.log").read()
print("== all LAUNCH/CONTAMINATION/finished/skip since 20:00 ==")
for line in log.splitlines():
    if re.match(r"\[2026-09-23 2[01]:", line) and re.search(r"LAUNCH|CONTAM|finished|skip|HOLD", line):
        print("  ", line)

print("\n== live trainer processes ==")
out = subprocess.run(["ps", "-eo", "pid,etime,stat,args"], capture_output=True, text=True).stdout
for line in out.splitlines():
    if "final30k_trainer.py" in line and "grep" not in line:
        parts = line.split(None, 3)
        print("  pid=%s etime=%s stat=%s" % (parts[0], parts[1], parts[2]))
        print("     cmd: %s" % parts[3][:130])

print("\n== trainer log tails ==")
for name in ("log_b1a_bonsai.txt", "log_b1a_counter.txt", "log_b1a_flowers.txt",
             "log_c0_flowers.txt", "log_c0_drjohnson.txt"):
    p = f"{dirs}/{name}"
    if os.path.exists(p):
        lines = open(p, errors="replace").read().strip().splitlines()
        print(f"  {name}: {len(lines)} lines; last: {lines[-1][:110] if lines else '(empty)'}")
    else:
        print(f"  {name}: (missing)")
