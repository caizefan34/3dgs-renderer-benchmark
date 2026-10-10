#!/usr/bin/env python3
"""R11 P0: Audit R10 numerical consistency"""

print("=== R11 P0: R10 Allocation Audit ===\n")

# R10 claimed values
oom_bytes = 119488359424  # From R10 snapshot
claimed_n_isects = 9957363285  # R10 claimed

print(f"1. R10 OOM allocation: {oom_bytes:,} bytes")
print(f"   = {oom_bytes / 1e9:.2f} GB")
print(f"   = {oom_bytes / 1024**3:.2f} GiB")
print()

print(f"2. R10 claimed n_isects: {claimed_n_isects:,}")
print()

# Calculate expected allocation sizes
print(f"3. Allocation size analysis:")
print()

# If single int64 array
int64_bytes = claimed_n_isects * 8
print(f"   Single int64 array:")
print(f"     {claimed_n_isects:,} × 8 = {int64_bytes:,} bytes")
print(f"     = {int64_bytes / 1e9:.2f} GB")
print(f"     = {int64_bytes / 1024**3:.2f} GiB")
print()

# If single int32 array
int32_bytes = claimed_n_isects * 4
print(f"   Single int32 array:")
print(f"     {claimed_n_isects:,} × 4 = {int32_bytes:,} bytes")
print(f"     = {int32_bytes / 1e9:.2f} GB")
print(f"     = {int32_bytes / 1024**3:.2f} GiB")
print()

# If both arrays (12 bytes total)
both_bytes = claimed_n_isects * 12
print(f"   Both arrays (int64 + int32):")
print(f"     {claimed_n_isects:,} × 12 = {both_bytes:,} bytes")
print(f"     = {both_bytes / 1e9:.2f} GB")
print(f"     = {both_bytes / 1024**3:.2f} GiB")
print()

# Compare with actual OOM
print(f"4. Comparison:")
print(f"   Actual OOM:      {oom_bytes:,} bytes ({oom_bytes / 1024**3:.2f} GiB)")
print(f"   Both arrays:     {both_bytes:,} bytes ({both_bytes / 1024**3:.2f} GiB)")
print(f"   Difference:      {abs(oom_bytes - both_bytes):,} bytes")
print()

if abs(oom_bytes - both_bytes) < 1024**2:  # Within 1 MB
    print(f"   ✓ MATCH: OOM size matches both arrays allocation")
    print(f"   Note: OOM likely on first array (isect_ids)")
else:
    print(f"   ⚠ MISMATCH: OOM size does not match expected allocation")
    print(f"   This suggests n_isects calculation may be incorrect")

print()

# Back-calculate n_isects from OOM
print(f"5. Back-calculate n_isects from OOM:")
n_from_oom_div8 = oom_bytes // 8
n_from_oom_div12 = oom_bytes // 12

print(f"   If int64 only: {oom_bytes:,} / 8 = {n_from_oom_div8:,}")
print(f"   If both arrays: {oom_bytes:,} / 12 = {n_from_oom_div12:,}")
print()

print(f"6. Conclusion:")
print(f"   R10 claimed n_isects = {claimed_n_isects:,}")
print(f"   OOM / 12 = {n_from_oom_div12:,}")
print(f"   Match: {claimed_n_isects == n_from_oom_div12}")
