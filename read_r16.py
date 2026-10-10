import subprocess

# Read R16_REPLAY_COUNTS.csv
result = subprocess.run([
    'ssh', 'mx',
    'cat /mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX/R16_REPLAY_COUNTS.csv'
], capture_output=True, text=True)

print("=== R16_REPLAY_COUNTS.csv ===")
print(result.stdout)
