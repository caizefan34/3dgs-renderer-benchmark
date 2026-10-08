
// f0_kernels.cu — CGT-F0 fused CUDA kernels (teacher / student / mixed-fused).
//
// Design (CGT-F0 kill-shot):
//   * One CUDA block per tile; 256 threads = one per 16x16 pixel.
//   * Centered pixel coords px[256,2] are IDENTICAL for every tile (every tile is a
//     16x16 grid centered on its own mean), so a single shared px array is used.
//   * Teacher path: front-to-back alpha composite over the tile's M Gaussians.
//     Gaussians are cached in shared memory in contiguous depth-ordered chunks
//     (CHUNK at a time) so global HBM traffic is ~M floats/tile, not ~M*256.
//   * Student path: rank-R CGT mixture eval (R<=16). Coeffs cached in shared mem.
//   * Fused (mixed) path: per-block UNIFORM branch on safe_label[t]. Because the
//     safe/fallback decision is constant across the block, there is NO warp
//     divergence — the whole block takes the student path or the teacher path.
//   * The alpha composite IS the compositing; student/teacher both emit (E_rgb, T).
//
// Math is bit-faithful to g1_compressibility.tile_teacher / predict_and_vjp.
// Templated on scalar_t so an fp64 build cross-checks the fp32 fast path against
// the float64 numpy reference (correctness gate max_abs_RGB <= 2e-5).

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#define TILE   16
#define NPX    (TILE * TILE)   // 256
#define POWER_CLIP (-4.5f)     // ~3 sigma, matches g1 POWER_CLIP
#define CHUNK  512             // gaussians cached in shared per chunk (512*9*4B=18KB)

template <typename T>
__device__ __forceinline__ T sigmoid(T x) { return T(1) / (T(1) + exp(-x)); }

// -------------------------------- teacher (canonical) --------------------------------
template <typename T>
__global__ void f0_teacher_kernel(
    const T* __restrict__ means2d,   // [Mt, 2]
    const T* __restrict__ conics,    // [Mt, 3]
    const T* __restrict__ opac,      // [Mt]
    const T* __restrict__ colors,    // [Mt, 3]
    const int* __restrict__ tile_off,// [n_tiles + 1]
    const T* __restrict__ px,        // [NPX, 2]  (centered, shared by all tiles)
    T* __restrict__ out,             // [n_tiles, NPX, 4]
    int n_tiles)
{
    const int t   = blockIdx.x;
    if (t >= n_tiles) return;
    const int pix = threadIdx.x;
    if (pix >= NPX) return;

    extern __shared__ unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);   // [CHUNK * 9]

    const int goff = tile_off[t];
    const int gcnt = tile_off[t + 1] - goff;

    const T px_x = px[pix * 2 + 0];
    const T px_y = px[pix * 2 + 1];

    T E0 = 0, E1 = 0, E2 = 0, Tv = 1;

    for (int base = 0; base < gcnt; base += CHUNK) {
        const int cm = min(CHUNK, gcnt - base);
        for (int idx = pix; idx < cm * 9; idx += blockDim.x) {
            const int j = idx / 9;
            const int f = idx % 9;
            const int g = goff + base + j;
            T v;
            switch (f) {
                case 0: v = means2d[g * 2 + 0]; break;
                case 1: v = means2d[g * 2 + 1]; break;
                case 2: v = conics[g * 3 + 0]; break;
                case 3: v = conics[g * 3 + 1]; break;
                case 4: v = conics[g * 3 + 2]; break;
                case 5: v = opac[g];          break;
                case 6: v = colors[g * 3 + 0]; break;
                case 7: v = colors[g * 3 + 1]; break;
                default: v = colors[g * 3 + 2]; break;
            }
            smem[idx] = v;
        }
        __syncthreads();
        for (int j = 0; j < cm; ++j) {
            const T mx = smem[j * 9 + 0];
            const T my = smem[j * 9 + 1];
            const T a  = smem[j * 9 + 2];
            const T b  = smem[j * 9 + 3];
            const T c  = smem[j * 9 + 4];
            const T op = smem[j * 9 + 5];
            const T cr = smem[j * 9 + 6];
            const T cg = smem[j * 9 + 7];
            const T cb = smem[j * 9 + 8];
            const T dx = px_x - mx;
            const T dy = px_y - my;
            const T pw = -T(0.5) * (a * dx * dx + T(2) * b * dx * dy + c * dy * dy);
            T alpha = 0;
            if (pw > POWER_CLIP) {
                alpha = op * exp(pw < T(0) ? pw : T(0));
                if (alpha > T(0.999)) alpha = T(0.999);
                if (alpha < T(0))     alpha = T(0);
            }
            E0 += Tv * alpha * cr;
            E1 += Tv * alpha * cg;
            E2 += Tv * alpha * cb;
            Tv *= (T(1) - alpha);
        }
        __syncthreads();
    }

    const int o = (t * NPX + pix) * 4;
    out[o + 0] = E0; out[o + 1] = E1; out[o + 2] = E2; out[o + 3] = Tv;
}

// -------------------------------- student (CGT fast path) --------------------------------
template <typename T>
__global__ void f0_student_kernel(
    const T* __restrict__ coeffs,   // [n_tiles, R, 10]
    const int R,
    const T* __restrict__ px,       // [NPX, 2]
    T* __restrict__ out,            // [n_tiles, NPX, 4]
    int n_tiles)
{
    const int t   = blockIdx.x;
    if (t >= n_tiles) return;
    const int pix = threadIdx.x;
    if (pix >= NPX) return;

    extern __shared__ unsigned char sraw[];
    T* sp = reinterpret_cast<T*>(sraw);   // [R * 10]
    for (int idx = pix; idx < R * 10; idx += blockDim.x) sp[idx] = coeffs[t * R * 10 + idx];
    __syncthreads();

    const T px_x = px[pix * 2 + 0];
    const T px_y = px[pix * 2 + 1];

    T logit[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r >= R) { logit[r] = 0; continue; }
        const T dx = px_x - sp[r * 10 + 0];
        const T dy = px_y - sp[r * 10 + 1];
        const T l00 = exp(sp[r * 10 + 2]);
        const T l10 = sp[r * 10 + 3];
        const T l11 = exp(sp[r * 10 + 4]);
        const T z0 = dx * l00 + dy * l10;
        const T z1 = dy * l11;
        logit[r] = sp[r * 10 + 5] - T(0.5) * (z0 * z0 + z1 * z1);
    }
    // softmax over R (no max-subtraction: matches numpy softmax; logits bounded by p5 in [-12,12])
    T mx = logit[0];
    for (int r = 1; r < R; ++r) mx = (logit[r] > mx) ? logit[r] : mx;
    T s = 0;
    for (int r = 0; r < R; ++r) { logit[r] = exp(logit[r] - mx); s += logit[r]; }
    T E0 = 0, E1 = 0, E2 = 0, E3 = 0;
    for (int r = 0; r < R; ++r) {
        const T w = logit[r] / s;
        const T tt = sigmoid(sp[r * 10 + 6]);
        const T cr = sigmoid(sp[r * 10 + 7]);
        const T cg = sigmoid(sp[r * 10 + 8]);
        const T cb = sigmoid(sp[r * 10 + 9]);
        const T er0 = (T(1) - tt) * cr;
        const T er1 = (T(1) - tt) * cg;
        const T er2 = (T(1) - tt) * cb;
        const T er3 = tt;
        E0 += w * er0; E1 += w * er1; E2 += w * er2; E3 += w * er3;
    }
    const int o = (t * NPX + pix) * 4;
    out[o + 0] = E0; out[o + 1] = E1; out[o + 2] = E2; out[o + 3] = E3;
}

// -------------------------------- fused mixed (safe->student, else->teacher) --------------------------------
template <typename T>
__global__ void f0_fused_kernel(
    const T* __restrict__ means2d, const T* __restrict__ conics,
    const T* __restrict__ opac, const T* __restrict__ colors,
    const int* __restrict__ tile_off,
    const T* __restrict__ coeffs, const int R,
    const int* __restrict__ safe_label,
    const T* __restrict__ px,
    T* __restrict__ out,
    int n_tiles)
{
    const int t   = blockIdx.x;
    if (t >= n_tiles) return;
    const int pix = threadIdx.x;
    if (pix >= NPX) return;

    if (safe_label[t]) {
        // ---- student path (safe tile) ----
        extern __shared__ unsigned char sraw[];
        T* sp = reinterpret_cast<T*>(sraw);
        for (int idx = pix; idx < R * 10; idx += blockDim.x) sp[idx] = coeffs[t * R * 10 + idx];
        __syncthreads();
        const T px_x = px[pix * 2 + 0];
        const T px_y = px[pix * 2 + 1];
        T logit[16];
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r >= R) { logit[r] = 0; continue; }
            const T dx = px_x - sp[r * 10 + 0];
            const T dy = px_y - sp[r * 10 + 1];
            const T l00 = exp(sp[r * 10 + 2]);
            const T l10 = sp[r * 10 + 3];
            const T l11 = exp(sp[r * 10 + 4]);
            const T z0 = dx * l00 + dy * l10;
            const T z1 = dy * l11;
            logit[r] = sp[r * 10 + 5] - T(0.5) * (z0 * z0 + z1 * z1);
        }
        T mx = logit[0];
        for (int r = 1; r < R; ++r) mx = (logit[r] > mx) ? logit[r] : mx;
        T s = 0;
        for (int r = 0; r < R; ++r) { logit[r] = exp(logit[r] - mx); s += logit[r]; }
        T E0 = 0, E1 = 0, E2 = 0, E3 = 0;
        for (int r = 0; r < R; ++r) {
            const T w = logit[r] / s;
            const T tt = sigmoid(sp[r * 10 + 6]);
            const T cr = sigmoid(sp[r * 10 + 7]);
            const T cg = sigmoid(sp[r * 10 + 8]);
            const T cb = sigmoid(sp[r * 10 + 9]);
            E0 += w * (T(1) - tt) * cr; E1 += w * (T(1) - tt) * cg;
            E2 += w * (T(1) - tt) * cb; E3 += w * tt;
        }
        const int o = (t * NPX + pix) * 4;
        out[o + 0] = E0; out[o + 1] = E1; out[o + 2] = E2; out[o + 3] = E3;
    } else {
        // ---- teacher path (fallback tile) ----
        extern __shared__ unsigned char smem_raw[];
        T* smem = reinterpret_cast<T*>(smem_raw);
        const int goff = tile_off[t];
        const int gcnt = tile_off[t + 1] - goff;
        const T px_x = px[pix * 2 + 0];
        const T px_y = px[pix * 2 + 1];
        T E0 = 0, E1 = 0, E2 = 0, Tv = 1;
        for (int base = 0; base < gcnt; base += CHUNK) {
            const int cm = min(CHUNK, gcnt - base);
            for (int idx = pix; idx < cm * 9; idx += blockDim.x) {
                const int j = idx / 9;
                const int f = idx % 9;
                const int g = goff + base + j;
                T v;
                switch (f) {
                    case 0: v = means2d[g * 2 + 0]; break;
                    case 1: v = means2d[g * 2 + 1]; break;
                    case 2: v = conics[g * 3 + 0]; break;
                    case 3: v = conics[g * 3 + 1]; break;
                    case 4: v = conics[g * 3 + 2]; break;
                    case 5: v = opac[g];          break;
                    case 6: v = colors[g * 3 + 0]; break;
                    case 7: v = colors[g * 3 + 1]; break;
                    default: v = colors[g * 3 + 2]; break;
                }
                smem[idx] = v;
            }
            __syncthreads();
            for (int j = 0; j < cm; ++j) {
                const T mx = smem[j * 9 + 0];
                const T my = smem[j * 9 + 1];
                const T a  = smem[j * 9 + 2];
                const T b  = smem[j * 9 + 3];
                const T c  = smem[j * 9 + 4];
                const T op = smem[j * 9 + 5];
                const T cr = smem[j * 9 + 6];
                const T cg = smem[j * 9 + 7];
                const T cb = smem[j * 9 + 8];
                const T dx = px_x - mx;
                const T dy = px_y - my;
                const T pw = -T(0.5) * (a * dx * dx + T(2) * b * dx * dy + c * dy * dy);
                T alpha = 0;
                if (pw > POWER_CLIP) {
                    alpha = op * exp(pw < T(0) ? pw : T(0));
                    if (alpha > T(0.999)) alpha = T(0.999);
                    if (alpha < T(0))     alpha = T(0);
                }
                E0 += Tv * alpha * cr;
                E1 += Tv * alpha * cg;
                E2 += Tv * alpha * cb;
                Tv *= (T(1) - alpha);
            }
            __syncthreads();
        }
        const int o = (t * NPX + pix) * 4;
        out[o + 0] = E0; out[o + 1] = E1; out[o + 2] = E2; out[o + 3] = Tv;
    }
}

// =============================== C ABI ===============================
extern "C" {

void* f0_malloc(size_t n) { void* p = nullptr; cudaMalloc(&p, n); return p; }
void  f0_free(void* p)    { if (p) cudaFree(p); }
void  f0_h2d(void* d, const void* h, size_t n) { cudaMemcpy(d, h, n, cudaMemcpyHostToDevice); }
void  f0_d2h(void* h, const void* d, size_t n) { cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost); }
void  f0_init(int dev) { cudaSetDevice(dev); cudaFree(0); }
void  f0_sync()        { cudaDeviceSynchronize(); }

int f0_smem_teacher(int dev) {
    int support = 0;
    cudaDeviceGetAttribute(&support, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return support;
}

// ---- float32 (fast / timed path) ----
void f0_teacher_f32(const float* m2d, const float* co, const float* op, const float* cl,
                    const int* off, const float* px, float* out, int nt) {
    size_t sh = (size_t)CHUNK * 9 * sizeof(float);
    f0_teacher_kernel<float><<<nt, NPX, sh>>>(m2d, co, op, cl, off, px, out, nt);
}
void f0_student_f32(const float* coeffs, int R, const float* px, float* out, int nt) {
    size_t sh = (size_t)R * 10 * sizeof(float);
    f0_student_kernel<float><<<nt, NPX, sh>>>(coeffs, R, px, out, nt);
}
void f0_fused_f32(const float* m2d, const float* co, const float* op, const float* cl,
                  const int* off, const float* coeffs, int R, const int* safe,
                  const float* px, float* out, int nt) {
    // teacher branch needs CHUNK*9 floats; student branch needs R*10 floats; take max.
    size_t sh = (size_t)CHUNK * 9 * sizeof(float);
    if ((size_t)R * 10 * sizeof(float) > sh) sh = (size_t)R * 10 * sizeof(float);
    f0_fused_kernel<float><<<nt, NPX, sh>>>(m2d, co, op, cl, off, coeffs, R, safe, px, out, nt);
}

// ---- float64 (correctness cross-check) ----
void f0_teacher_f64(const double* m2d, const double* co, const double* op, const double* cl,
                    const int* off, const double* px, double* out, int nt) {
    size_t sh = (size_t)CHUNK * 9 * sizeof(double);
    f0_teacher_kernel<double><<<nt, NPX, sh>>>(m2d, co, op, cl, off, px, out, nt);
}
void f0_student_f64(const double* coeffs, int R, const double* px, double* out, int nt) {
    size_t sh = (size_t)R * 10 * sizeof(double);
    f0_student_kernel<double><<<nt, NPX, sh>>>(coeffs, R, px, out, nt);
}
void f0_fused_f64(const double* m2d, const double* co, const double* op, const double* cl,
                  const int* off, const double* coeffs, int R, const int* safe,
                  const double* px, double* out, int nt) {
    size_t sh = (size_t)CHUNK * 9 * sizeof(double);
    if ((size_t)R * 10 * sizeof(double) > sh) sh = (size_t)R * 10 * sizeof(double);
    f0_fused_kernel<double><<<nt, NPX, sh>>>(m2d, co, op, cl, off, coeffs, R, safe, px, out, nt);
}


// ---- CUDA-event timing helpers (default stream, self-contained) ----
void* f0_ev_create() { cudaEvent_t e; cudaEventCreate(&e); return (void*)e; }
void  f0_ev_record(void* e) { cudaEventRecord((cudaEvent_t)e, cudaStreamDefault); }
float f0_ev_elapsed(void* a, void* b) {
    cudaEventSynchronize((cudaEvent_t)b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, (cudaEvent_t)a, (cudaEvent_t)b);
    return ms;
}
void  f0_ev_destroy(void* e) { if (e) cudaEventDestroy((cudaEvent_t)e); }
void  f0_stream_sync() { cudaStreamSynchronize(cudaStreamDefault); }
} // extern "C"
