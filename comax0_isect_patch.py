#!/usr/bin/env python3
# Apply isect-order composite fix to production envelope kernel + bindings.
import sys, os, shutil
SRC = "/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_CUDA_PROD/src"
def rd(p): return open(p, encoding="utf-8").read()
def wr(p, s): open(p, "w", encoding="utf-8").write(s)
def must(c, m):
    if not c:
        print("PATCH_FAIL: " + m); sys.exit(1)
ok = []

p = os.path.join(SRC, "comax_envelope_kernel.cu")
s = rd(p); shutil.copy(p, p + ".bak_pre_isect_20261006")
old = "struct Rec { float depth; float alpha; float prev_max; float r, g, b; };"
new = "struct Rec { float alpha; float prev_max; float r, g, b; int gid; int isect_pos; };"
must(old in s, "Rec struct"); s = s.replace(old, new, 1); ok.append("Rec")
old = """    float* __restrict__ out_rgb, float* __restrict__ out_alpha,
    unsigned long long* __restrict__ counters) // [8]"""
new = """    const int* __restrict__ tile_ids, const int* __restrict__ tile_offsets,
    float* __restrict__ out_rgb, float* __restrict__ out_alpha,
    unsigned long long* __restrict__ counters) // [8]"""
must(old in s, "kernel sig"); s = s.replace(old, new, 1); ok.append("kernel_sig")
old = """                        recs[n_rec].depth = depth[gid];
                        recs[n_rec].alpha = al;
                        recs[n_rec].prev_max = m_g;
                        recs[n_rec].r = color[3*gid];
                        recs[n_rec].g = color[3*gid+1];
                        recs[n_rec].b = color[3*gid+2];
                        ++n_rec;"""
new = """                        recs[n_rec].alpha = al;
                        recs[n_rec].prev_max = m_g;
                        recs[n_rec].gid = gid;
                        recs[n_rec].r = color[3*gid];
                        recs[n_rec].g = color[3*gid+1];
                        recs[n_rec].b = color[3*gid+2];
                        ++n_rec;"""
must(old in s, "emit"); s = s.replace(old, new, 1); ok.append("emit")
old = """    // depth-sort records (insertion sort; n_rec small)
    for (int i = 1; i < n_rec; ++i) {
        Rec key = recs[i];
        int j = i - 1;
        while (j >= 0 && recs[j].depth > key.depth) { recs[j+1] = recs[j]; --j; }
        recs[j+1] = key;
    }"""
new = """    // Compute each record isect position within this tile, then sort by it so the
    // composite order matches the dense scan / trusted oracle (gsplat isect order),
    // not pure depth order. isect order = depth+radial key; depth-sort mis-weights T.
    {
        int s0 = tile_offsets[tile], e0 = tile_offsets[tile+1];
        for (int i = 0; i < n_rec; ++i) {
            int g = recs[i].gid; int pos = e0 - s0;
            for (int k = s0; k < e0; ++k) { if (tile_ids[k] == g) { pos = k - s0; break; } }
            recs[i].isect_pos = pos;
        }
        for (int i = 1; i < n_rec; ++i) {
            Rec key = recs[i];
            int j = i - 1;
            while (j >= 0 && recs[j].isect_pos > key.isect_pos) { recs[j+1] = recs[j]; --j; }
            recs[j+1] = key;
        }
    }"""
must(old in s, "sort"); s = s.replace(old, new, 1); ok.append("sort")
old = """    int n_tiles, int tw, int th, int width, int height,
    const float* background, float* out_rgb, float* out_alpha, unsigned long long* counters)
{
    dim3 block(PX);
    dim3 grid(n_tiles);
    comax_envelope_kernel<<<grid, block>>>(
        means2d, conic, opacity, color, depth, tile_groups, tile_group_count, TG,
        n_x0, n_y0, n_x1, n_y1, n_o, n_lam, n_leaf, node_base, node_pow2,
        n_tiles, tw, th, width, height, background, out_rgb, out_alpha, counters);"""
new = """    int n_tiles, int tw, int th, int width, int height,
    const int* tile_ids, const int* tile_offsets,
    const float* background, float* out_rgb, float* out_alpha, unsigned long long* counters)
{
    dim3 block(PX);
    dim3 grid(n_tiles);
    comax_envelope_kernel<<<grid, block>>>(
        means2d, conic, opacity, color, depth, tile_groups, tile_group_count, TG,
        n_x0, n_y0, n_x1, n_y1, n_o, n_lam, n_leaf, node_base, node_pow2,
        tile_ids, tile_offsets,
        n_tiles, tw, th, width, height, background, out_rgb, out_alpha, counters);"""
must(old in s, "launch"); s = s.replace(old, new, 1); ok.append("launch")
wr(p, s)

p = os.path.join(SRC, "comax_prod.cpp")
s = rd(p); shutil.copy(p, p + ".bak_pre_isect_20261006")
old = """    const int* node_base, const int* node_pow2,
    int n_tiles, int tw, int th, int width, int height,
    const float* background, float* out_rgb, float* out_alpha, unsigned long long* counters);"""
new = """    const int* node_base, const int* node_pow2,
    int n_tiles, int tw, int th, int width, int height,
    const int* tile_ids, const int* tile_offsets,
    const float* background, float* out_rgb, float* out_alpha, unsigned long long* counters);"""
must(old in s, "cpp decl"); s = s.replace(old, new, 1); ok.append("cpp_decl")
old = """    torch::Tensor node_base, torch::Tensor node_pow2,
    int64_t n_tiles, int64_t tw, int64_t th, int64_t width, int64_t height,
    torch::Tensor background) {"""
new = """    torch::Tensor node_base, torch::Tensor node_pow2,
    int64_t n_tiles, int64_t tw, int64_t th, int64_t width, int64_t height,
    torch::Tensor tile_ids, torch::Tensor tile_offsets,
    torch::Tensor background) {"""
must(old in s, "cpp def"); s = s.replace(old, new, 1); ok.append("cpp_def")
old = """      node_base.data_ptr<int>(), node_pow2.data_ptr<int>(),
      (int)n_tiles, (int)tw, (int)th, (int)width, (int)height,
      background.data_ptr<float>(), out_rgb.data_ptr<float>(), out_alpha.data_ptr<float>(),"""
new = """      node_base.data_ptr<int>(), node_pow2.data_ptr<int>(),
      (int)n_tiles, (int)tw, (int)th, (int)width, (int)height,
      tile_ids.data_ptr<int>(), tile_offsets.data_ptr<int>(),
      background.data_ptr<float>(), out_rgb.data_ptr<float>(), out_alpha.data_ptr<float>(),"""
must(old in s, "cpp call"); s = s.replace(old, new, 1); ok.append("cpp_call")
wr(p, s)

p = os.path.join(SRC, "comax_envelope_ext.py")
s = rd(p); shutil.copy(p, p + ".bak_pre_isect_20261006")
old = """        "node_base": node_base.contiguous(), "node_pow2": node_pow2.contiguous(),
        "n_tiles": n_tiles, "tw": tw, "th": th, "width": width, "height": height,"""
new = """        "node_base": node_base.contiguous(), "node_pow2": node_pow2.contiguous(),
        "tile_ids": tile_ids.contiguous(), "tile_offsets": tile_offsets.contiguous(),
        "n_tiles": n_tiles, "tw": tw, "th": th, "width": width, "height": height,"""
must(old in s, "bundle dict"); s = s.replace(old, new, 1); ok.append("bundle")
old = """        bundle["node_base"], bundle["node_pow2"],
        bundle["n_tiles"], bundle["tw"], bundle["th"], bundle["width"], bundle["height"],"""
new = """        bundle["node_base"], bundle["node_pow2"],
        bundle["tile_ids"], bundle["tile_offsets"],
        bundle["n_tiles"], bundle["tw"], bundle["th"], bundle["width"], bundle["height"],"""
must(old in s, "render call"); s = s.replace(old, new, 1); ok.append("render_call")
wr(p, s)

print("PATCH_OK", ",".join(ok))
