# BFC0 Source Audit

## gsplat Version: 1.4.0

## Semantics

- **threads_per_block**: tile_size, tile_size, 1
- **traversal_direction**: reverse_front_to_back
- **last_ids_computed_in_forward**: True
- **last_ids_forward_evidence**: [{'line': 37, 'code': 'int32_t *__restrict__ last_ids // [C, image_height, image_width]'}, {'line': 52, 'code': 'last_ids += camera_id * image_height * image_width;'}, {'line': 184, 'code': 'last_ids[pix_id] = static_cast<int32_t>(cur_idx);'}, {'line': 240, 'code': 'torch::Tensor last_ids = torch::empty('}, {'line': 285, 'code': 'last_ids.data_ptr<int32_t>()'}, {'line': 288, 'code': 'return std::make_tuple(renders, alphas, last_ids);'}]
- **backward_references_last_ids**: True
- **has_early_termination**: True
- **warp_skip_granularity**: warp_all_lanes_inactive

## Source Evidence

### pixel_to_lane_mapping

- Line 51: `uint32_t camera_id = block.group_index().x;`
- Line 53: `block.group_index().y * tile_width + block.group_index().z;`
- Line 54: `uint32_t i = block.group_index().y * tile_size + block.thread_index().y;`
- Line 55: `uint32_t j = block.group_index().z * tile_size + block.thread_index().x;`
- Line 75: `const S px = (S)j + 0.5f;`
- Line 76: `const S py = (S)i + 0.5f;`

### warp_boundaries

- Line 183: `if (!warp.any(valid)) {`
- Line 243: `warpSum<COLOR_DIM, S>(v_rgb_local, warp);`
- Line 244: `warpSum<decltype(warp), S>(v_conic_local, warp);`
- Line 245: `warpSum<decltype(warp), S>(v_xy_local, warp);`
- Line 247: `warpSum<decltype(warp), S>(v_xy_abs_local, warp);`
- Line 249: `warpSum<decltype(warp), S>(v_opacity_local, warp);`
- Line 250: `if (warp.thread_rank() == 0) {`
- Line 340: `dim3 threads = {tile_size, tile_size, 1};`

### last_ids_usage

- Line 38: `const int32_t *__restrict__ last_ids, // [C, image_height, image_width]`
- Line 59: `last_ids += camera_id * image_height * image_width;`
- Line 111: `const int32_t bin_final = inside ? last_ids[pix_id] : 0;`
- Line 303: `const torch::Tensor &last_ids,      // [C, image_height, image_width]`
- Line 319: `GSPLAT_CHECK_INPUT(last_ids);`
- Line 391: `last_ids.data_ptr<int32_t>(),`
- Line 433: `const torch::Tensor &last_ids,      // [C, image_height, image_width]`
- Line 459: `last_ids,                                                          \`

### traversal_index_mapping

- Line 34: `const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]`
- Line 35: `const int32_t *__restrict__ flatten_ids,  // [n_isects]`
- Line 57: `tile_offsets += camera_id * tile_height * tile_width;`
- Line 84: `// have all threads in tile process the same gaussians in batches`
- Line 85: `// first collect gaussians between range.x and range.y in batches`
- Line 87: `int32_t range_start = tile_offsets[tile_id];`
- Line 88: `int32_t range_end =`
- Line 91: `: tile_offsets[tile_id + 1];`
- Line 93: `const uint32_t num_batches =`
- Line 94: `(range_end - range_start + block_size - 1) / block_size;`
- Line 97: `int32_t *id_batch = (int32_t *)s; // [block_size]`
- Line 98: `vec3<S> *xy_opacity_batch =`
- Line 99: `reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]`
- Line 100: `vec3<S> *conic_batch =`
- Line 101: `reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]`
- Line 103: `S *rgbs_batch = (S *)&conic_batch[block_size]; // [block_size * COLOR_DIM]`
- Line 111: `const int32_t bin_final = inside ? last_ids[pix_id] : 0;`
- Line 121: `// collect and process batches of gaussians`
- Line 125: `const int32_t warp_bin_final =`
- Line 126: `cg::reduce(warp, bin_final, cg::greater<int>());`
- Line 127: `for (uint32_t b = 0; b < num_batches; ++b) {`
- Line 128: `// resync all threads before writing next batch of shared mem`
- Line 132: `// 0 index will be furthest back in batch`
- Line 134: `// batch end is the index of the last gaussian in the batch`
- Line 136: `const int32_t batch_end = range_end - 1 - block_size * b;`
- Line 137: `const int32_t batch_size = min(block_size, batch_end + 1 - range_start);`
- Line 138: `const int32_t idx = batch_end - tr;`
- Line 139: `if (idx >= range_start) {`
- Line 140: `int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]`
- Line 141: `id_batch[tr] = g;`
- Line 144: `xy_opacity_batch[tr] = {xy.x, xy.y, opac};`
- Line 145: `conic_batch[tr] = conics[g];`
- Line 148: `rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];`
- Line 151: `// wait for other threads to collect the gaussians in batch`
- Line 153: `// process gaussians in the current batch for this pixel`
- Line 154: `// 0 index is the furthest back gaussian in the batch`
- Line 155: `for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;`
- Line 158: `if (batch_end - t > bin_final) {`
- Line 168: `conic = conic_batch[t];`
- Line 169: `vec3<S> xy_opac = xy_opacity_batch[t];`
- Line 206: `(rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *`
- Line 240: `buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;`
- Line 251: `int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]`
- Line 299: `const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]`
- Line 300: `const torch::Tensor &flatten_ids,  // [n_isects]`
- Line 316: `GSPLAT_CHECK_INPUT(tile_offsets);`
- Line 317: `GSPLAT_CHECK_INPUT(flatten_ids);`
- Line 331: `uint32_t C = tile_offsets.size(0);         // number of cameras`
- Line 333: `uint32_t n_isects = flatten_ids.size(0);`
- Line 335: `uint32_t tile_height = tile_offsets.size(1);`
- Line 336: `uint32_t tile_width = tile_offsets.size(2);`
- Line 388: `tile_offsets.data_ptr<int32_t>(),`
- Line 389: `flatten_ids.data_ptr<int32_t>(),`
- Line 429: `const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]`
- Line 430: `const torch::Tensor &flatten_ids,  // [n_isects]`
- Line 456: `tile_offsets,                                                      \`
- Line 457: `flatten_ids,                                                       \`

### validity_skip

- Line 82: `bool inside = (i < image_height && j < image_width);`
- Line 111: `const int32_t bin_final = inside ? last_ids[pix_id] : 0;`
- Line 157: `bool valid = inside;`
- Line 159: `valid = 0;`
- Line 167: `if (valid) {`
- Line 178: `valid = false;`
- Line 183: `if (!warp.any(valid)) {`
- Line 191: `// initialize everything to 0, only set if the lane is valid`
- Line 192: `if (valid) {`

### alpha_power_validity

- Line 24: `const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]`
- Line 37: `const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]`
- Line 42: `const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]`
- Line 46: `vec3<S> *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]`
- Line 58: `render_alphas += camera_id * image_height * image_width;`
- Line 61: `v_render_alphas += camera_id * image_height * image_width;`
- Line 98: `vec3<S> *xy_opacity_batch =`
- Line 100: `vec3<S> *conic_batch =`
- Line 101: `reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]`
- Line 103: `S *rgbs_batch = (S *)&conic_batch[block_size]; // [block_size * COLOR_DIM]`
- Line 106: `S T_final = 1.0f - render_alphas[pix_id];`
- Line 119: `const S v_render_a = v_render_alphas[pix_id];`
- Line 144: `xy_opacity_batch[tr] = {xy.x, xy.y, opac};`
- Line 145: `conic_batch[tr] = conics[g];`
- Line 161: `S alpha;`
- Line 164: `vec3<S> conic;`
- Line 165: `S vis;`
- Line 168: `conic = conic_batch[t];`
- Line 169: `vec3<S> xy_opac = xy_opacity_batch[t];`
- Line 172: `S sigma = 0.5f * (conic.x * delta.x * delta.x +`
- Line 173: `conic.z * delta.y * delta.y) +`
- Line 174: `conic.y * delta.x * delta.y;`
- Line 175: `vis = __expf(-sigma);`
- Line 176: `alpha = min(0.999f, opac * vis);`
- Line 177: `if (sigma < 0.f || alpha < 1.f / 255.f) {`
- Line 187: `vec3<S> v_conic_local = {0.f, 0.f, 0.f};`
- Line 190: `S v_opacity_local = 0.f;`
- Line 194: `S ra = 1.0f / (1.0f - alpha);`
- Line 197: `const S fac = alpha * T;`
- Line 203: `S v_alpha = 0.f;`
- Line 205: `v_alpha +=`
- Line 210: `v_alpha += T_final * ra * v_render_a;`
- Line 218: `v_alpha += -T_final * ra * accum;`
- Line 221: `if (opac * vis <= 0.999f) {`
- Line 222: `const S v_sigma = -opac * vis * v_alpha;`
- Line 223: `v_conic_local = {`
- Line 224: `0.5f * v_sigma * delta.x * delta.x,`
- Line 225: `v_sigma * delta.x * delta.y,`
- Line 226: `0.5f * v_sigma * delta.y * delta.y`
- Line 229: `v_sigma * (conic.x * delta.x + conic.y * delta.y),`
- Line 230: `v_sigma * (conic.y * delta.x + conic.z * delta.y)`
- Line 235: `v_opacity_local = vis * v_alpha;`
- Line 244: `warpSum<decltype(warp), S>(v_conic_local, warp);`
- Line 249: `warpSum<decltype(warp), S>(v_opacity_local, warp);`
- Line 258: `S *v_conic_ptr = (S *)(v_conics) + 3 * g;`
- Line 259: `gpuAtomicAdd(v_conic_ptr, v_conic_local.x);`
- Line 260: `gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);`
- Line 261: `gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);`
- Line 273: `gpuAtomicAdd(v_opacities + g, v_opacity_local);`
- Line 289: `const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]`
- Line 302: `const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]`
- Line 306: `const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]`
- Line 313: `GSPLAT_CHECK_INPUT(conics);`
- Line 318: `GSPLAT_CHECK_INPUT(render_alphas);`
- Line 321: `GSPLAT_CHECK_INPUT(v_render_alphas);`
- Line 344: `torch::Tensor v_conics = torch::zeros_like(conics);`
- Line 377: `reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),`
- Line 390: `render_alphas.data_ptr<float>(),`
- Line 393: `v_render_alphas.data_ptr<float>(),`
- Line 399: `reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),`
- Line 406: `v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities`
- Line 419: `const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]`
- Line 432: `const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]`
- Line 436: `const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]`
- Line 448: `conics,                                                            \`
- Line 458: `render_alphas,                                                     \`
- Line 461: `v_render_alphas,                                                   \`

### reductions_atomics

- Line 126: `cg::reduce(warp, bin_final, cg::greater<int>());`
- Line 243: `warpSum<COLOR_DIM, S>(v_rgb_local, warp);`
- Line 244: `warpSum<decltype(warp), S>(v_conic_local, warp);`
- Line 245: `warpSum<decltype(warp), S>(v_xy_local, warp);`
- Line 247: `warpSum<decltype(warp), S>(v_xy_abs_local, warp);`
- Line 249: `warpSum<decltype(warp), S>(v_opacity_local, warp);`
- Line 255: `gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);`
- Line 259: `gpuAtomicAdd(v_conic_ptr, v_conic_local.x);`
- Line 260: `gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);`
- Line 261: `gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);`
- Line 264: `gpuAtomicAdd(v_xy_ptr, v_xy_local.x);`
- Line 265: `gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);`
- Line 269: `gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);`
- Line 270: `gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);`
- Line 273: `gpuAtomicAdd(v_opacities + g, v_opacity_local);`

