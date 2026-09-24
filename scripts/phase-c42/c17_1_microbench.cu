// c17_1_microbench.cu
// Standalone benchmark: CUB DeviceRadixSort vs CUB BlockRadixSort
// Tests per-tile sorting performance for 3DGS tile-local sort feasibility
//
// Build:
//   nvcc -std=c++17 -arch=sm_80 -O2 -o c17_1_microbench c17_1_microbench.cu
//
// Run:
//   ./c17_1_microbench
//
// Measures:
//   Method 1: DeviceRadixSort on N_total 46-bit keys (6 passes) — baseline
//   Method 2: BlockRadixSort per tile, N_tile 32-bit keys (4 passes) — C17-1
//   Method 3: DeviceRadixSort on N_total 14-bit keys (2 passes) — reduced-bit global sort
//   Method 3+2: Combined: 14-bit global sort + per-tile BlockRadixSort

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define N_TILES_16 8160   // 120x68 tiles for 1920x1080 at tile_size=16
#define N_WARMUP 10
#define N_ITERS 100

// ============================================================
// Method 1: CUB DeviceRadixSort (baseline — 46-bit, 6 passes)
// ============================================================
double bench_device_radix_sort_46bit(int64_t n_total, int n_iters) {
    // Keys: int64 (46-bit used), Values: int32
    int64_t *d_keys_in, *d_keys_out;
    int32_t *d_vals_in, *d_vals_out;
    CHECK_CUDA(cudaMalloc(&d_keys_in, n_total * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_keys_out, n_total * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_vals_in, n_total * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_vals_out, n_total * sizeof(int32_t)));

    // Fill with random data
    srand(42);
    std::vector<int64_t> h_keys(n_total);
    std::vector<int32_t> h_vals(n_total);
    for (int64_t i = 0; i < n_total; i++) {
        // Simulate: image_id (1 bit) | tile_id (13 bits) | depth (32 bits)
        int64_t tile_id = rand() % N_TILES_16;
        int64_t depth = (int64_t)(rand() << 16) & 0xFFFFFFFFLL;
        h_keys[i] = (tile_id << 32) | depth;
        h_vals[i] = (int32_t)(rand() % n_total);
    }
    CHECK_CUDA(cudaMemcpy(d_keys_in, h_keys.data(), n_total * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_in, h_vals.data(), n_total * sizeof(int32_t), cudaMemcpyHostToDevice));

    // CUB temp storage
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 0, 46, 0);
    void *d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 0, 46, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iters; i++) {
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 0, 46, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaFree(d_keys_in));
    CHECK_CUDA(cudaFree(d_keys_out));
    CHECK_CUDA(cudaFree(d_vals_in));
    CHECK_CUDA(cudaFree(d_vals_out));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / n_iters;
}

// ============================================================
// Method 3: CUB DeviceRadixSort (14-bit, 2 passes — reduced-bit)
// ============================================================
double bench_device_radix_sort_14bit(int64_t n_total, int n_iters) {
    int64_t *d_keys_in, *d_keys_out;
    int32_t *d_vals_in, *d_vals_out;
    CHECK_CUDA(cudaMalloc(&d_keys_in, n_total * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_keys_out, n_total * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_vals_in, n_total * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_vals_out, n_total * sizeof(int32_t)));

    srand(42);
    std::vector<int64_t> h_keys(n_total);
    std::vector<int32_t> h_vals(n_total);
    for (int64_t i = 0; i < n_total; i++) {
        int64_t tile_id = rand() % N_TILES_16;
        int64_t depth = (int64_t)(rand() << 16) & 0xFFFFFFFFLL;
        h_keys[i] = (tile_id << 32) | depth;
        h_vals[i] = (int32_t)(rand() % n_total);
    }
    CHECK_CUDA(cudaMemcpy(d_keys_in, h_keys.data(), n_total * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_in, h_vals.data(), n_total * sizeof(int32_t), cudaMemcpyHostToDevice));

    size_t temp_bytes = 0;
    // Only sort by tile_id bits [32, 32+14) = [32, 46) — but we need [0,46) to include depth
    // For reduced-bit: sort only [32, 46) = 14 bits = 2 passes
    // This groups by tile but doesn't sort by depth within tile
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 32, 46, 0);
    void *d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));

    for (int i = 0; i < N_WARMUP; i++) {
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 32, 46, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iters; i++) {
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 32, 46, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaFree(d_keys_in));
    CHECK_CUDA(cudaFree(d_keys_out));
    CHECK_CUDA(cudaFree(d_vals_in));
    CHECK_CUDA(cudaFree(d_vals_out));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / n_iters;
}

// ============================================================
// Method 2: CUB BlockRadixSort (per-tile, 32-bit depth, 4 passes)
// ============================================================

// Kernel template: each CTA sorts one tile's intersections
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void block_sort_kernel(
    const int32_t * __restrict__ keys_in,   // [n_tiles * tile_capacity]
    int32_t * __restrict__ keys_out,         // [n_tiles * tile_capacity]
    const int32_t * __restrict__ vals_in,    // [n_tiles * tile_capacity]
    int32_t * __restrict__ vals_out,          // [n_tiles * tile_capacity]
    int tile_capacity,                        // actual elements per tile (may be < BLOCK_THREADS*ITEMS_PER_THREAD)
    int n_tiles
) {
    using BlockSort = cub::BlockRadixSort<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD, int32_t>;
    __shared__ typename BlockSort::TempStorage temp_storage;

    int tile_idx = blockIdx.x;
    if (tile_idx >= n_tiles) return;

    int offset = tile_idx * BLOCK_THREADS * ITEMS_PER_THREAD;

    // Load keys and values
    int32_t thread_keys[ITEMS_PER_THREAD];
    int32_t thread_vals[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = offset + threadIdx.x * ITEMS_PER_THREAD + i;
        if (threadIdx.x * ITEMS_PER_THREAD + i < tile_capacity) {
            thread_keys[i] = keys_in[idx];
            thread_vals[i] = vals_in[idx];
        } else {
            thread_keys[i] = INT32_MAX;  // pad with max value
            thread_vals[i] = 0;
        }
    }

    // Sort by key (32-bit depth, ascending)
    BlockSort(temp_storage).Sort(thread_keys, thread_vals);

    // Write back
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = offset + threadIdx.x * ITEMS_PER_THREAD + i;
        if (threadIdx.x * ITEMS_PER_THREAD + i < tile_capacity) {
            keys_out[idx] = thread_keys[i];
            vals_out[idx] = thread_vals[i];
        }
    }
}

// Benchmark wrapper for a specific tile size
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
double bench_block_sort(int n_tiles, int tile_capacity, int n_iters) {
    int total_elements = n_tiles * BLOCK_THREADS * ITEMS_PER_THREAD;

    int32_t *d_keys_in, *d_keys_out, *d_vals_in, *d_vals_out;
    CHECK_CUDA(cudaMalloc(&d_keys_in, total_elements * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_keys_out, total_elements * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_vals_in, total_elements * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_vals_out, total_elements * sizeof(int32_t)));

    // Fill with random depth values
    srand(42);
    std::vector<int32_t> h_keys(total_elements), h_vals(total_elements);
    for (int i = 0; i < total_elements; i++) {
        h_keys[i] = (i < n_tiles * tile_capacity) ? (rand() & 0x7FFFFFFF) : 0;
        h_vals[i] = rand();
    }
    CHECK_CUDA(cudaMemcpy(d_keys_in, h_keys.data(), total_elements * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_in, h_vals.data(), total_elements * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Get shared memory and occupancy info
    cudaFuncAttributes attr;
    CHECK_CUDA(cudaFuncGetAttributes(&attr, block_sort_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>));
    int smem = attr.sharedSizeBytes;
    int regs = attr.numRegs;
    int max_blocks_per_sm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, block_sort_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>, BLOCK_THREADS, smem));

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        block_sort_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<n_tiles, BLOCK_THREADS>>>(
                d_keys_in, d_keys_out, d_vals_in, d_vals_out, tile_capacity, n_tiles);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iters; i++) {
        block_sort_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<n_tiles, BLOCK_THREADS>>>(
                d_keys_in, d_keys_out, d_vals_in, d_vals_out, tile_capacity, n_tiles);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Print occupancy info
    int n_sms;
    cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, 0);
    int waves = (max_blocks_per_sm > 0) ?
        (n_tiles + max_blocks_per_sm * n_sms - 1) / (max_blocks_per_sm * n_sms) : -1;
    printf("    [BlockSort] threads=%d items=%d smem=%d regs=%d max_blocks/SM=%d waves=%d",
           BLOCK_THREADS, ITEMS_PER_THREAD, smem, regs, max_blocks_per_sm, waves);
    if (max_blocks_per_sm == 0) {
        printf(" [OVERFLOW: smem > 48KB limit]");
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d_keys_in));
    CHECK_CUDA(cudaFree(d_keys_out));
    CHECK_CUDA(cudaFree(d_vals_in));
    CHECK_CUDA(cudaFree(d_vals_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / n_iters;
}

// ============================================================
// Method 3+2: Combined: 14-bit global sort + per-tile block sort
// ============================================================
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
double bench_combined_sort(int64_t n_total, int n_tiles, int tile_capacity, int n_iters) {
    // Step 1: 14-bit global sort
    int64_t *d_keys_in, *d_keys_out;
    int32_t *d_vals_in, *d_vals_out;
    CHECK_CUDA(cudaMalloc(&d_keys_in, n_total * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_keys_out, n_total * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_vals_in, n_total * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_vals_out, n_total * sizeof(int32_t)));

    srand(42);
    std::vector<int64_t> h_keys(n_total);
    std::vector<int32_t> h_vals(n_total);
    for (int64_t i = 0; i < n_total; i++) {
        int64_t tile_id = rand() % n_tiles;
        int64_t depth = (int64_t)(rand() << 16) & 0xFFFFFFFFLL;
        h_keys[i] = (tile_id << 32) | depth;
        h_vals[i] = (int32_t)(rand() % n_total);
    }
    CHECK_CUDA(cudaMemcpy(d_keys_in, h_keys.data(), n_total * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals_in, h_vals.data(), n_total * sizeof(int32_t), cudaMemcpyHostToDevice));

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 32, 46, 0);
    void *d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));

    // Step 2: per-tile block sort on the depth (lower 32 bits)
    int32_t *d_depth_in, *d_depth_out, *d_gauss_in, *d_gauss_out;
    int padded_total = n_tiles * BLOCK_THREADS * ITEMS_PER_THREAD;
    CHECK_CUDA(cudaMalloc(&d_depth_in, padded_total * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_depth_out, padded_total * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_gauss_in, padded_total * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_gauss_out, padded_total * sizeof(int32_t)));

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        // Step 1: global sort by tile_id (14-bit)
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 32, 46, 0);
        // Step 2: per-tile block sort by depth (would need extract+sort, simplified here)
        block_sort_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<n_tiles, BLOCK_THREADS>>>(
                d_depth_in, d_depth_out, d_gauss_in, d_gauss_out, tile_capacity, n_tiles);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iters; i++) {
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out, n_total, 32, 46, 0);
        block_sort_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<n_tiles, BLOCK_THREADS>>>(
                d_depth_in, d_depth_out, d_gauss_in, d_gauss_out, tile_capacity, n_tiles);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaFree(d_keys_in));
    CHECK_CUDA(cudaFree(d_keys_out));
    CHECK_CUDA(cudaFree(d_vals_in));
    CHECK_CUDA(cudaFree(d_vals_out));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_depth_in));
    CHECK_CUDA(cudaFree(d_depth_out));
    CHECK_CUDA(cudaFree(d_gauss_in));
    CHECK_CUDA(cudaFree(d_gauss_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / n_iters;
}

// ============================================================
// Main
// ============================================================
int main() {
    // Get GPU info
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Shared mem per block (default): %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared mem per block (max): %zu KB\n", prop.sharedMemPerBlockOptin / 1024);
    printf("Shared mem per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n\n", prop.warpSize);

    int n_tiles = N_TILES_16;
    int n_iters = N_ITERS;

    // Test sizes: (tile_capacity, BLOCK_THREADS, ITEMS_PER_THREAD)
    struct TestCase {
        int n_tile;
        int block_threads;
        int items_per_thread;
    };
    TestCase tests[] = {
        {256,   128, 2},
        {512,   128, 4},
        {1024,  128, 8},
        {2048,  256, 8},
        {4096,  256, 16},
        {8192,  256, 32},
        {12000, 256, 48},   // 256*48=12288, pad 288 elements
    };

    printf("===================================================================\n");
    printf("C17-1 Microbenchmark: DeviceRadixSort vs BlockRadixSort\n");
    printf("n_tiles=%d, n_iters=%d\n", n_tiles, n_iters);
    printf("===================================================================\n\n");

    // Print header
    printf("%-8s | %-12s | %-12s | %-12s | %-12s | %-12s | %s\n",
           "N_tile", "Global46(ms)", "Global14(ms)", "BlockSort(ms)", "Combined(ms)", "Block/Global", "Decision");
    printf("%-8s-+-%-12s-+-%-12s-+-%-12s-+-%-12s-+-%-12s-+-%s\n",
           "--------", "------------", "------------", "------------", "------------", "------------", "----------");

    for (auto &tc : tests) {
        int64_t n_total = (int64_t)tc.n_tile * n_tiles;

        // Method 1: Global sort 46-bit (6 passes)
        double t_global46 = bench_device_radix_sort_46bit(n_total, n_iters);

        // Method 3: Global sort 14-bit (2 passes)
        double t_global14 = bench_device_radix_sort_14bit(n_total, n_iters);

        // Method 2: Block sort per tile (32-bit, 4 passes)
        double t_block = 0;
        bool block_overflow = false;
        switch (tc.items_per_thread) {
            case 2:  t_block = bench_block_sort<128, 2>(n_tiles, tc.n_tile, n_iters); break;
            case 4:  t_block = bench_block_sort<128, 4>(n_tiles, tc.n_tile, n_iters); break;
            case 8:  t_block = (tc.block_threads == 128) ?
                               bench_block_sort<128, 8>(n_tiles, tc.n_tile, n_iters) :
                               bench_block_sort<256, 8>(n_tiles, tc.n_tile, n_iters); break;
            case 16: t_block = bench_block_sort<256, 16>(n_tiles, tc.n_tile, n_iters); break;
            case 32:
            case 48:
                // 256x32 = 8192 items, 8 bytes each = 64KB > 48KB default static smem
                // 256x48 = 12288 items, 8 bytes each = 96KB > 48KB
                // CUB BlockRadixSort uses static shared memory, cannot exceed 48KB
                block_overflow = true;
                t_block = -1;
                printf("    [BlockSort] N=%d exceeds 48KB static shared memory limit (OVERFLOW)\n", tc.n_tile);
                break;
        }

        // Method 3+2: Combined (14-bit global + block sort)
        double t_combined = 0;
        if (!block_overflow) {
            switch (tc.items_per_thread) {
                case 2:  t_combined = bench_combined_sort<128, 2>(n_total, n_tiles, tc.n_tile, n_iters); break;
                case 4:  t_combined = bench_combined_sort<128, 4>(n_total, n_tiles, tc.n_tile, n_iters); break;
                case 8:  t_combined = (tc.block_threads == 128) ?
                                      bench_combined_sort<128, 8>(n_total, n_tiles, tc.n_tile, n_iters) :
                                      bench_combined_sort<256, 8>(n_total, n_tiles, tc.n_tile, n_iters); break;
                case 16: t_combined = bench_combined_sort<256, 16>(n_total, n_tiles, tc.n_tile, n_iters); break;
            }
        } else {
            t_combined = -1;
        }

        double ratio = block_overflow ? -1 : t_block / t_global46;
        const char *decision = block_overflow ? "OVERFLOW (>48KB)" :
                               (ratio < 1.0/1.5) ? "PASS (>=1.5x)" : (ratio < 1.0) ? "MARGINAL" : "FAIL (<1.5x)";

        if (block_overflow) {
            printf("%-8d | %-12.4f | %-12.4f | %-12s | %-12s | %-12s | %s\n",
                   tc.n_tile, t_global46, t_global14, "OVERFLOW", "OVERFLOW", "-", decision);
        } else {
            printf("%-8d | %-12.4f | %-12.4f | %-12.4f | %-12.4f | %-12.2f | %s\n",
                   tc.n_tile, t_global46, t_global14, t_block, t_combined, ratio, decision);
        }
    }

    printf("\n===================================================================\n");
    printf("Decision gate: Block sort must be >=1.5x faster than global sort\n");
    printf("for realistic tile sizes (1024-4096 intersections)\n");
    printf("===================================================================\n");

    return 0;
}
