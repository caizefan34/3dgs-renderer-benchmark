#!/usr/bin/env python3
"""Patch Intersect.cpp to use C17-0 tile_segmented_sort when segmented=true."""
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

# Simple approach: replace the function name and adjust parameters
# Original: segmented_radix_sort_double_buffer(n_isects, I, image_n_bits, tile_n_bits, offsets, ...)
# C17-0:    tile_segmented_sort_double_buffer(n_isects, I, n_tiles, image_n_bits, tile_n_bits, ...)

old_name = "segmented_radix_sort_double_buffer("
new_name = "tile_segmented_sort_double_buffer("

if old_name not in content:
    print("ERROR: segmented_radix_sort_double_buffer not found in Intersect.cpp")
    sys.exit(1)

# Replace function name
content = content.replace(old_name, new_name)

# Now need to add n_tiles parameter and remove offsets parameter
# The call looks like:
#             tile_segmented_sort_double_buffer(
#                 n_isects,
#                 I,
#                 image_n_bits,
#                 tile_n_bits,
#                 offsets,
#                 isect_ids,
# ...
# We need to change to:
#             tile_segmented_sort_double_buffer(
#                 n_isects,
#                 I,
#                 n_tiles,
#                 image_n_bits,
#                 tile_n_bits,
#                 isect_ids,
# ...

# Replace "I,\n                image_n_bits," with "I,\n                n_tiles,\n                image_n_bits,"
old_params = """                I,
                image_n_bits,"""
new_params = """                I,
                n_tiles,
                image_n_bits,"""
content = content.replace(old_params, new_params, 1)

# Remove the "offsets," line that comes after tile_n_bits
old_offset = """                tile_n_bits,
                offsets,
                isect_ids,"""
new_no_offset = """                tile_n_bits,
                isect_ids,"""
content = content.replace(old_offset, new_no_offset, 1)

with open(path, 'w') as f:
    f.write(content)

# Verify
with open(path) as f:
    c = f.read()
has_c17 = "tile_segmented_sort_double_buffer" in c
has_n_tiles = "n_tiles,\n                image_n_bits" in c
no_offsets_param = "offsets,\n                isect_ids" not in c
print(f"Function name replaced: {has_c17}")
print(f"n_tiles added: {has_n_tiles}")
print(f"offsets removed: {no_offsets_param}")
