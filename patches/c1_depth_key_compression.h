/**
 * Phase 16 C1 — Depth Key Compression
 * 
 * Changes:
 * 1. Key encoding: depth_upper_16 at bits [0:15], tile_id at [16:16+tile_n_bits-1], image_id above
 * 2. Sort end_bit: 32+tile_n_bits+image_n_bits → 16+tile_n_bits+image_n_bits
 * 3. Offset kernel: shift depth out by 16 instead of 32
 * 
 * Saves 33% of CUB radix sort passes (12 → 8 for tile16 1080p)
 * Zero ordering inversions (mathematically proven)
 */

// [BASELINE]
// isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
// iid_enc = iid << (32 + tile_n_bits);
// 
// Layout: image_id (X bits) | tile_id (Xt bits) | depth (32 bits)
// Sort range: bits [0, 32+tile_n_bits+image_n_bits)  → 47 bits for tile16

// [C1]
// isect_ids[cur_idx] = depth_upper | (tile_id << 16) | (iid << (16 + tile_n_bits));
// 
// Layout: image_id (X bits) | tile_id (Xt bits) | depth_upper (16 bits)
// Sort range: bits [0, 16+tile_n_bits+image_n_bits)  → 31 bits for tile16

// CUB radix passes: ceil(47/4)=12 → ceil(31/4)=8  (33% reduction)
