// H5-1 validation microkernel (self-contained, ctypes-loadable).
// Each kernel is preceded by a plain-C host launcher that does the launch,
// so we never call a __global__ directly through the ctypes ABI.
// Compile: nvcc -arch=sm_80 -Xptxas=-v -O3 -shared -Xcompiler -fPIC -o h5_micro.so h5_micro.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define MIN_T    (1e-4f)   // early-termination residual-T threshold
#define A_EPS    (1e-6f)   // minimum alpha treated as a contributing reachable Gaussian
#define MAXCHAIN 2048

__device__ __forceinline__ float clampf_(float x, float a, float b){return fminf(fmaxf(x,a),b);}

// ---------------------------------------------------------------------------
// Kernel 1: per-pixel reachable count + early termination (one thread/pixel).
// Outputs L[p], last_id[p], sT[p].
// ---------------------------------------------------------------------------
extern "C" __global__ void h5_count(
    const long long* __restrict__ tile_ptr,
    const int*        __restrict__ tile_ids,
    int n_tiles,
    const float*      __restrict__ m2d,      // [N*2] fx,fy
    const float*      __restrict__ conics,   // [N*3] A,B,C
    const float*      __restrict__ opacity,  // [N]
    const float*      __restrict__ color,    // [N*3]
    int W,int H,int tw,int th,int TS,
    int* __restrict__ L, int* __restrict__ last_id, float* __restrict__ sT)
{
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= W*H) return;
    int x = p % W, y = p / W;
    int ty = y / TS, tx = x / TS;
    int t = ty*tw + tx;
    long long lo = tile_ptr[t], hi = tile_ptr[t+1];
    float Tprev = 1.0f;
    int cnt = 0, last = -1;
    for (long long i=lo; i<hi; ++i){
        int g = tile_ids[i];
        float fx = m2d[g*2], fy = m2d[g*2+1];
        float A=conics[g*3], B=conics[g*3+1], C=conics[g*3+2];
        float op0=opacity[g];
        float dx = (float)(x+0.5f) - fx;
        float dy = (float)(y+0.5f) - fy;
        float q  = A*dx*dx + 2.0f*B*dx*dy + C*dy*dy;
        float a  = clampf_(op0 * __expf(-0.5f*q), 0.0f, 1.0f);
        if (a > A_EPS && Tprev > A_EPS){ last = (int)i; ++cnt; }
        Tprev *= (1.0f - a);
        if (Tprev < MIN_T) break;
    }
    L[p] = cnt; last_id[p] = last; sT[p] = Tprev;
}

extern "C" void launch_h5_count(
    const long long* tile_ptr, const int* tile_ids, int n_tiles,
    const float* m2d, const float* conics, const float* opacity, const float* color,
    int W,int H,int tw,int th,int TS,
    int* L, int* last_id, float* sT)
{
    int npx = W*H; int block=256; int grid=(npx+block-1)/block;
    h5_count<<<grid,block>>>(tile_ptr,tile_ids,n_tiles,m2d,conics,opacity,color,
                             W,H,tw,th,TS,L,last_id,sT);
}

// ---------------------------------------------------------------------------
// Kernel 2: extract full chains for sampled pixels (one thread/sample).
// p = samp_idx[gt]. Writes front-to-back chain arrays.
// ---------------------------------------------------------------------------
extern "C" __global__ void h5_sample_chains(
    const long long* __restrict__ tile_ptr,
    const int*        __restrict__ tile_ids,
    int n_tiles,
    const float*      __restrict__ m2d,
    const float*      __restrict__ conics,
    const float*      __restrict__ opacity,
    const float*      __restrict__ color,
    int W,int H,int tw,int th,int TS,
    int nsamp, const int* __restrict__ samp_idx,
    float* __restrict__ chain_alpha, float* __restrict__ chain_rgb,
    float* __restrict__ chain_vis, int* __restrict__ chain_len)
{
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= nsamp) return;
    int p = samp_idx[r];
    if (p < 0 || p >= W*H){ chain_len[r]=0; return; }
    int x = p % W, y = p / W;
    int ty = y / TS, tx = x / TS;
    int t = ty*tw + tx;
    long long lo = tile_ptr[t], hi = tile_ptr[t+1];
    float Tprev = 1.0f; int cnt = 0;
    float* sa = chain_alpha + r*MAXCHAIN;
    float* sr = chain_rgb + (long long)r*MAXCHAIN*3;
    float* sv = chain_vis + r*MAXCHAIN;
    for (long long i=lo; i<hi; ++i){
        int g = tile_ids[i];
        float fx = m2d[g*2], fy = m2d[g*2+1];
        float A=conics[g*3], B=conics[g*3+1], C=conics[g*3+2];
        float op0=opacity[g];
        float dx = (float)(x+0.5f) - fx;
        float dy = (float)(y+0.5f) - fy;
        float q  = A*dx*dx + 2.0f*B*dx*dy + C*dy*dy;
        float a  = clampf_(op0 * __expf(-0.5f*q), 0.0f, 1.0f);
        if (a > A_EPS && Tprev > A_EPS && cnt < MAXCHAIN){
            sa[cnt]=a; sr[cnt*3]=color[g*3]; sr[cnt*3+1]=color[g*3+1]; sr[cnt*3+2]=color[g*3+2];
            sv[cnt]=Tprev; ++cnt;
        }
        Tprev *= (1.0f - a);
        if (Tprev < MIN_T) break;
    }
    chain_len[r] = cnt;
}

extern "C" void launch_h5_sample_chains(
    const long long* tile_ptr, const int* tile_ids, int n_tiles,
    const float* m2d, const float* conics, const float* opacity, const float* color,
    int W,int H,int tw,int th,int TS,
    int nsamp, const int* samp_idx,
    float* chain_alpha, float* chain_rgb, float* chain_vis, int* chain_len)
{
    int block=256; int grid=(nsamp+block-1)/block;
    h5_sample_chains<<<grid,block>>>(tile_ptr,tile_ids,n_tiles,m2d,conics,opacity,color,
                                     W,H,tw,th,TS,nsamp,samp_idx,
                                     chain_alpha,chain_rgb,chain_vis,chain_len);
}

// ---------------------------------------------------------------------------
// Shared per-chain scalar-adjoint backward recurrence.
// See patches/higs-scalar-adjoint.patch for the derivation.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void bwd_seq_rec(
    const float* __restrict__ a, const float* __restrict__ rgb,
    const float* __restrict__ Vc, float tail_const, int L,
    float* va, float* vrgb)
{
    float Tf = 1.0f; for(int i=0;i<L;++i) Tf *= (1.0f - a[i]);
    float T = Tf; float buffer_dot = 0.0f;
    for (int ii=0; ii<L; ++ii){
        int i = L-1-ii;
        float ra = 1.0f/(1.0f-a[i]);
        T *= ra; float fac = a[i]*T;
        float rgb_dot = rgb[i*3]*Vc[0]+rgb[i*3+1]*Vc[1]+rgb[i*3+2]*Vc[2];
        vrgb[i*3+0]=fac*Vc[0]; vrgb[i*3+1]=fac*Vc[1]; vrgb[i*3+2]=fac*Vc[2];
        va[i]=T*rgb_dot + ra*(tail_const - buffer_dot);
        buffer_dot += rgb_dot*fac;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: sequential (baseline) backward over packed chains. One thread/chain.
// ---------------------------------------------------------------------------
extern "C" __global__ void h5_bwd_sequential(
    const float* __restrict__ chain_alpha, const float* __restrict__ chain_rgb,
    const int* __restrict__ chain_len, int SAMP, int MAXC,
    const float* __restrict__ Vc_all, const float* __restrict__ tail_all,
    float* __restrict__ va, float* __restrict__ vrgb)
{
    int r = blockIdx.x*blockDim.x+threadIdx.x;
    if (r>=SAMP) return;
    int L = chain_len[r];
    if (L<=0) return;
    bwd_seq_rec(chain_alpha + r*MAXC, chain_rgb + (long long)r*MAXC*3,
                Vc_all + r*3, tail_all[r], L,
                va + r*MAXC, vrgb + (long long)r*MAXC*3);
}

extern "C" void launch_h5_bwd_sequential(
    const float* chain_alpha, const float* chain_rgb,
    const int* chain_len, int SAMP, int MAXC,
    const float* Vc_all, const float* tail_all,
    float* va, float* vrgb, int threads)
{
    int grid=(SAMP+threads-1)/threads;
    h5_bwd_sequential<<<grid,threads>>>(chain_alpha,chain_rgb,chain_len,SAMP,MAXC,
                                        Vc_all,tail_all,va,vrgb);
}

// ---------------------------------------------------------------------------
// Kernel 4: factored segment backward over packed chains. One thread/chain.
// Per-thread local arrays (MAXSEG=256) = the "temporary state" cost.
// ---------------------------------------------------------------------------
#define MAXSEG 256
extern "C" __global__ void h5_bwd_factored(
    const float* __restrict__ chain_alpha, const float* __restrict__ chain_rgb,
    const int* __restrict__ chain_len, int SAMP, int MAXC, int SEGSIZE,
    const float* __restrict__ Vc_all, const float* __restrict__ tail_all,
    float* __restrict__ va, float* __restrict__ vrgb)
{
    int r = blockIdx.x*blockDim.x+threadIdx.x;
    if (r>=SAMP) return;
    int L = chain_len[r];
    if (L<=0) return;
    int nseg = (L + SEGSIZE - 1) / SEGSIZE;
    if (nseg > MAXSEG){ nseg = MAXSEG; } // guard; real scenes have small L/S
    const float* a  = chain_alpha + r*MAXC;
    const float* rgb= chain_rgb + (long long)r*MAXC*3;
    float Vc0=Vc_all[r*3], Vc1=Vc_all[r*3+1], Vc2=Vc_all[r*3+2];
    float tail_const = tail_all[r];
    float tau[MAXSEG], segc[MAXSEG], Va[MAXSEG], Buf[MAXSEG];
    for (int s=0;s<nseg;++s){
        int b=s*SEGSIZE, e=min(L,(s+1)*SEGSIZE);
        float tt=1.0f, sc=0.0f, tw=1.0f;
        for (int i=b;i<e;++i){
            float rgb_dot=rgb[i*3]*Vc0+rgb[i*3+1]*Vc1+rgb[i*3+2]*Vc2;
            sc += rgb_dot*a[i]*tw;
            tw *= (1.0f-a[i]); tt *= (1.0f-a[i]);
        }
        tau[s]=tt; segc[s]=sc;
    }
    { float pref=1.0f;
      for (int s=0;s<nseg;++s){ Va[s]=pref; pref*=tau[s]; }
      float Gbuf=0.0f;
      for (int s=nseg-1;s>=0;--s){ Buf[s]=Gbuf; Gbuf += Va[s]*segc[s]; } }
    for (int s=0;s<nseg;++s){
        int b=s*SEGSIZE, e=min(L,(s+1)*SEGSIZE);
        int m=e-b; float Vsegs[MAXSEG], facq[MAXSEG];
        float Vbase=Va[s]; float tow=1.0f;
        for (int i=b;i<e;++i){ Vsegs[i-b]=Vbase*tow; facq[i-b]=a[i]*Vsegs[i-b]; tow*=(1.0f-a[i]); }
        float buf=Buf[s];
        for (int j=e-1;j>=b;--j){
            int i=j-b;
            float rgb_dot=rgb[j*3]*Vc0+rgb[j*3+1]*Vc1+rgb[j*3+2]*Vc2;
            va[r*MAXC+j]=Vsegs[i]*rgb_dot + (1.0f/(1.0f-a[j]))*(tail_const - buf);
            vrgb[(long long)r*MAXC*3+j*3+0]=facq[i]*Vc0;
            vrgb[(long long)r*MAXC*3+j*3+1]=facq[i]*Vc1;
            vrgb[(long long)r*MAXC*3+j*3+2]=facq[i]*Vc2;
            buf += rgb_dot*facq[i];
        }
    }
}

extern "C" void launch_h5_bwd_factored(
    const float* chain_alpha, const float* chain_rgb,
    const int* chain_len, int SAMP, int MAXC, int SEGSIZE,
    const float* Vc_all, const float* tail_all,
    float* va, float* vrgb, int threads)
{
    int grid=(SAMP+threads-1)/threads;
    h5_bwd_factored<<<grid,threads>>>(chain_alpha,chain_rgb,chain_len,SAMP,MAXC,SEGSIZE,
                                      Vc_all,tail_all,va,vrgb);
}

extern "C" void h5_noop(){}