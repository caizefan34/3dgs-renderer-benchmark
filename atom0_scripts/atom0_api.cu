#include "atom0_api.h"
#include "atom0_blend_backward.cuh"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include "helper_math.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

namespace faster_gs::rasterization {

double backward_atom0_launcher(
    const float* grad_image, const float* image, const float3* bg_color,
    char* primitive_buffers_blob, char* tile_buffers_blob,
    char* instance_buffers_blob, char* bucket_buffers_blob,
    int n_primitives, int width, int height, int n_instances, int n_buckets,
    int instance_primitive_indices_selector, int variant, int chunk_first_bucket, int chunk_n_buckets,
    void* grad_mean2d, void* grad_conic, void* grad_opacity, void* grad_colors,
    void* private_grad, float* partial_out, int capture_partials)
{
    setvbuf(stderr, NULL, _IONBF, 0);
    if (chunk_n_buckets <= 0) return 0.0;
    const int n_tiles = (width / config::tile_width + (width % config::tile_width != 0)) *
                        (height / config::tile_height + (height % config::tile_height != 0));
    const int end_bit = extract_end_bit(n_tiles - 1);
    fprintf(stderr, "ATOM0DBG A n_tiles=%d end_bit=%d npr=%d\n", n_tiles, end_bit, n_primitives);

    PrimitiveBuffers primitive_buffers = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    fprintf(stderr, "ATOM0DBG B primitive done\n");
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles);
    fprintf(stderr, "ATOM0DBG C tile done\n");
    InstanceBuffers instance_buffers = InstanceBuffers::from_blob(instance_buffers_blob, n_instances, end_bit);
    fprintf(stderr, "ATOM0DBG D instance done\n");
    BucketBuffers bucket_buffers = BucketBuffers::from_blob(bucket_buffers_blob, n_buckets);
    fprintf(stderr, "ATOM0DBG E bucket done\n");
    instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;
    // chunk_base is computed on-device inside the kernel (host cannot read device pointers).
    const uint grid_width = static_cast<uint>((width + config::tile_width - 1) / config::tile_width);
    const dim3 grid(static_cast<unsigned int>(chunk_n_buckets), 1, 1);
    fprintf(stderr, "ATOM0DBG G gridw=%u\n", grid_width);

    float2* g_mean2d = reinterpret_cast<float2*>(grad_mean2d);
    float* g_conic = reinterpret_cast<float*>(grad_conic);
    float* g_opacity = reinterpret_cast<float*>(grad_opacity);
    float3* g_colors = reinterpret_cast<float3*>(grad_colors);
    float* priv = reinterpret_cast<float*>(private_grad);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernels::backward::blend_backward_atom0_cu<<<grid, 32>>>(
        tile_buffers.instance_ranges, tile_buffers.buckets_offset,
        instance_buffers.primitive_indices.Current(),
        primitive_buffers.mean2d, primitive_buffers.conic_opacity, primitive_buffers.color,
        bg_color, grad_image, image,
        tile_buffers.final_transmittances, tile_buffers.max_n_processed, tile_buffers.n_processed,
        bucket_buffers.tile_index, bucket_buffers.color_transmittance,
        g_mean2d, g_conic, g_opacity, g_colors,
        static_cast<uint>(n_primitives), static_cast<uint>(width), static_cast<uint>(height),
        grid_width, static_cast<uint>(variant), static_cast<uint>(chunk_first_bucket),
        0, priv, partial_out);
    cudaEventRecord(stop);
    fprintf(stderr, "ATOM0DBG H kernel launched\n");
    cudaEventSynchronize(stop);
    fprintf(stderr, "ATOM0DBG I synced\n");
    float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return static_cast<double>(ms);
}

}

torch::Tensor get_instance_primitive_indices(torch::Tensor instance_buffers, int n_instances, int width, int height, int selector)
{
    using namespace faster_gs::rasterization;
    const int n_tiles = (width / config::tile_width + (width % config::tile_width != 0)) *
                        (height / config::tile_height + (height % config::tile_height != 0));
    const int end_bit = extract_end_bit(n_tiles - 1);
    char* ptr = reinterpret_cast<char*>(instance_buffers.data_ptr());
    InstanceBuffers ib = InstanceBuffers::from_blob(ptr, n_instances, end_bit);
    ib.primitive_indices.selector = selector;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({n_instances}, opts);
    cudaMemcpyAsync(out.data_ptr<int>(), ib.primitive_indices.Current(),
                    static_cast<size_t>(n_instances) * sizeof(int),
                    cudaMemcpyDeviceToDevice, c10::cuda::getCurrentCUDAStream());
    return out;
}
