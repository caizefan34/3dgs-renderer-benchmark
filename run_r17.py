import subprocess
import json

print("=== R17 P0: R16 Arithmetic Audit ===\n")

# Read R16 CSV
result = subprocess.run(['ssh', 'mx', 'cat /mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX/R16_REPLAY_COUNTS.csv'], 
                       capture_output=True, text=True)
print(result.stdout)

lines = result.stdout.strip().split('\n')
data = [l.split(',') for l in lines[1:]]
raw = int(data[0][1])
sig = int(data[1][1])
thr = int(data[2][1])

print(f"\nRAW: {raw:,}")
print(f"SIG: {sig:,}")
print(f"THR: {thr:,}")
print(f"\nActual diff: {raw-sig:,}")
print(f"Claimed: 360,251")
print(f"ERROR: {360251-(raw-sig):,}")
print(f"\nThreshold exercised: {sig != thr}")

if sig == thr:
    print("→ THRESHOLD_BRANCH_NOT_EXERCISED")
