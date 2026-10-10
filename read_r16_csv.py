import subprocess
import json

def ssh_cmd(cmd):
    result = subprocess.run(['ssh', 'mx', cmd], capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

print("=== R17 P0: R16 Arithmetic Audit ===\n")

# Read R16_REPLAY_COUNTS.csv
stdout, stderr, rc = ssh_cmd('cat /mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX/R16_REPLAY_COUNTS.csv')
print(stdout)

# Parse
lines = stdout.strip().split('\n')
data = [line.split(',') for line in lines[1:]]

raw_gpu = int(data[0][1])
sigmoid_gpu = int(data[1][1])
threshold_gpu = int(data[2][1])

print(f"\nRAW GPU count sum: {raw_gpu:,}")
print(f"Sigmoid GPU count sum: {sigmoid_gpu:,}")
print(f"Threshold GPU count sum: {threshold_gpu:,}")
print(f"\nActual difference (RAW - Sigmoid): {raw_gpu - sigmoid_gpu:,}")

claimed = 360251
actual = raw_gpu - sigmoid_gpu
print(f"\nClaimed in report: {claimed:,}")
print(f"Actual difference: {actual:,}")
print(f"Discrepancy: {claimed - actual:,}")

if actual != claimed:
    print("\n⚠️ ARITHMETIC ERROR DETECTED in R16 report!")
else:
    print("\n✓ Arithmetic matches")

# Read oracle summaries to check threshold branch
print("\n=== Checking threshold branch ===")
for mode in ['raw', 'sigmoid', 'sigmoid_threshold_skip']:
    stdout, _, _ = ssh_cmd(f'cat /mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX/oracle_{mode}/oracle_summary.json')
    if stdout:
        summary = json.loads(stdout)
        print(f"\n{mode}:")
        print(f"  GPU count sum: {summary.get('gpu_sum_valid_and_skipped', 'N/A')}")
        print(f"  Oracle count sum: {summary.get('reference_sum_valid', 'N/A')}")
        print(f"  Mismatch pairs: {summary.get('mismatch_pairs', 'N/A')}")
        print(f"  Verdict: {summary.get('verdict', 'N/A')}")