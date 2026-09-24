import glob

cands = glob.glob("/mnt/storage_pool/3dgs-renderer-benchmark/repo/benchmark/run_*train_benchmark.py")
P = cands[0]
lines = open(P).read().splitlines()

for i, l in enumerate(lines):
    if "create_higs_renderer(" in l:
        for j in range(i, min(i + 12, len(lines))):
            print(f"{j+1}: {lines[j]}")
        print("---")

# Also show the import block at 637-640
print("=== import block ===")
for j in range(636, 650):
    print(f"{j+1}: {lines[j]}")