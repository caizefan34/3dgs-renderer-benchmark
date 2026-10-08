// BWD-ARITH1 R1: host-side global definitions (shared across forward/backward in same .so)
namespace faster_gs::rasterization::kernels {
    float* bwd1_gauss_cache_host = nullptr;
    unsigned int bwd1_cache_enable_host = 0u;
}
