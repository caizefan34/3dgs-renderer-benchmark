import subprocess
import json

def ssh(cmd):
    r = subprocess.run(['ssh', 'mx', cmd], capture_output=True, text=True)
    return r.stdout

print("=== R17 P0: R16 Arithmetic Audit ===\n")
csv = ssh('cat /mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX/R16_REPLAY_COUNTS.csv')
print(csv)
d = [l.split(',') for l in csv.strip().split('\n')[1:]]
raw, sig, thr = int(d[0][1]), int(d[1][1]), int(d[2][1])
print(f"\nRAW: {raw:,}\nSIG: {sig:,}\nTHR: {thr:,}")
print(f"\nActual diff: {raw-sig:,}")
print(f"Claimed: 360,251")
print(f"ERROR: {360251-(raw-sig):,}")
print(f"\nThreshold branch exercised: {sig != thr}")
if sig == thr:
    print("→ THRESHOLD_BRANCH_NOT_EXERCISED")
