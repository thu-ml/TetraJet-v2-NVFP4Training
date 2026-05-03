#include <cuda_runtime.h>
#include "utils.hpp"
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdint>

// utils
DEVICE float warpReduceScale(float outer_max, float& inner_max) {
    outer_max = fmaxf(__shfl_xor_sync(0xffffffff, outer_max, 2), outer_max);
    outer_max = fmaxf(__shfl_xor_sync(0xffffffff, outer_max, 1), outer_max);
    inner_max = outer_max;
    #pragma unroll
    for (int offset = 16; offset >= 4; offset /= 2) {
        outer_max = fmaxf(__shfl_down_sync(0xffffffff, outer_max, offset), outer_max);
    }
    return outer_max;
}

DEVICE float fast_rand(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    uint32_t mantissa = state >> 9; 
    uint32_t val = 0x3F800000u | mantissa;
    return __uint_as_float(val) - 1.5f;
}

DEVICE float get_fp4_gap(float x) {
    float abs_x = fabsf(x);
    return (abs_x < 2.0f ? 0.5f : (abs_x < 4.0f ? 1.0f : 2.0f));
}

DEVICE float get_value_added_noise(float x, float r) {
    float x_abs = fabsf(x);
    float res;

    if (x_abs <= 2.0f) {
        res = fminf(fmaxf(x_abs + r * 0.5f, 0.0f), 2.0f);
    } else if (x_abs <= 4.0f) {
        res = fminf(fmaxf(x_abs + r       , 2.0f), 4.0f);
    } else {
        res = fminf(fmaxf(x_abs + r * 2.0f, 4.0f), 6.0f);
    }

    return copysignf(res, x);
}

DEVICE float fp4_to_float(uint8_t x) {
    uint32_t u = static_cast<uint32_t>(x);

    uint32_t sign = (u & 0x8) << 28;

    uint32_t abs_u = u & 0x7;

    uint32_t exp = (abs_u >> 1) + 126;
    uint32_t mant_bit = (abs_u & 1) & (abs_u > 1);

    uint32_t result_bits = (exp << 23) | (mant_bit << 22);

    result_bits = (abs_u > 0 ? (result_bits | sign) : 0);

    return __uint_as_float(result_bits);
}

DEVICE void random_hadamard_16(float *x, int lane_idx, uint32_t seed) {
    int sub_lane = lane_idx % 4;

    unsigned int h = (seed ^ (sub_lane * 7919)); 
    h *= 2654435761u;
    h ^= h >> 16;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if ((h >> i) & 1) x[i] = -x[i];
    }

    float a;

    a = x[0]; x[0] += x[1]; x[1] = a - x[1];
    a = x[2]; x[2] += x[3]; x[3] = a - x[3];

    a = x[0]; x[0] += x[2]; x[2] = a - x[2];
    a = x[1]; x[1] += x[3]; x[3] = a - x[3];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float neighbor = __shfl_xor_sync(0xffffffff, x[i], 1);
        if ((sub_lane & 1) == 0) {
            x[i] += neighbor;
        } else {
            x[i] = neighbor - x[i];
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float neighbor = __shfl_xor_sync(0xffffffff, x[i], 2);
        
        if ((sub_lane & 2) == 0) {
            x[i] += neighbor;
        } else {
            x[i] = neighbor - x[i];
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[i] *= 0.25f;
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

// quant
// for test
template <
    int ThreadNum
>
__global__ void nvfp4_double_quantized_scale_only(
    const float* __restrict__ g_x,
    float* __restrict__ x_q,
    int n_elements
) {
    int tid = threadIdx.x;
    int blk_id = blockIdx.x;
    // load 4 elems per thread
    int offset = blk_id * ThreadNum * 4 + tid * 4;
    // each warp handles a 1x128 blcok
    float outer_max = -INFINITY;
    float x[4];
    if (offset < n_elements) {
        *(float4*)x = *(float4*)(g_x + offset);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            outer_max = fmaxf(fabsf(x[i]), outer_max);
        }
    }
    else {
        outer_max = -INFINITY;
    }
    float inner_max = -INFINITY;
    outer_max = warpReduceScale(outer_max, inner_max);

    float outer_scale = outer_max / 448.0f / 6.0f;
    if (outer_scale == 0.0f) outer_scale = 1.0f;
    outer_scale = __shfl_sync(0xffffffff, outer_scale, 0);

    float inner_scale = inner_max / outer_scale / 6.0f;
    if (inner_scale == 0.0f) inner_scale = 0.001953125f;

    __nv_fp8_storage_t inner_scale_fp8 = __nv_cvt_float_to_fp8(inner_scale, __NV_SATFINITE, __NV_E4M3);

    inner_scale = (float)(*(__nv_fp8_e4m3*)&inner_scale_fp8);
    float scale_product = outer_scale * inner_scale;
    
    float scales[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        scales[i] = scale_product;
    }
    if (offset < n_elements) {
        *(float4*)(x_q + offset) = *(float4*)scales;
    }
}

template <
    int ThreadNum,
    bool Stochastic
>
__global__ void nvfp4_double_quant(
    const float* __restrict__ g_x,
    uint8_t* __restrict__ x_q,
    float* __restrict__ g_outer_scale,
    uint8_t* __restrict__ g_inner_scale,
    int n_elements,
    uint32_t seed
) {
    int tid = threadIdx.x;
    int blk_id = blockIdx.x;
    // load 4 elems per thread
    // each warp handles a 1x128 blcok
    int offset = blk_id * ThreadNum * 4 + tid * 4;
    float outer_max = -INFINITY;
    float x[4];
    if (offset < n_elements) {
        *(float4*)x = *(float4*)(g_x + offset);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            outer_max = fmaxf(fabsf(x[i]), outer_max);
        }
    }
    else {
        outer_max = -INFINITY;
    }
    float inner_max = -INFINITY;
    outer_max = warpReduceScale(outer_max, inner_max);

    float outer_scale = outer_max / 448.0f / 6.0f;
    if (outer_scale == 0.0f) outer_scale = 1.0f;
    outer_scale = __shfl_sync(0xffffffff, outer_scale, 0);
    
    // store outer scale
    if (tid % 32 == 0) {
        int outer_scale_offset = blk_id * ThreadNum / 32 + tid / 32;
        g_outer_scale[outer_scale_offset] = outer_scale;
    }

    float inner_scale = inner_max / outer_scale / 6.0f;
    if (inner_scale == 0.0f) inner_scale = 0.001953125f;

    __nv_fp8_storage_t inner_scale_fp8 = __nv_cvt_float_to_fp8(inner_scale, __NV_SATFINITE, __NV_E4M3);
    // store inner scale
    if (tid % 4 == 0) {
        int inner_scale_offset = blk_id * ThreadNum / 4 + tid / 4;
        g_inner_scale[inner_scale_offset] = inner_scale_fp8;
    }

    inner_scale = (float)(*(__nv_fp8_e4m3*)&inner_scale_fp8);
    float scale_product = outer_scale * inner_scale;
    float scale_inv = (scale_product != 0.0f) ? (1.0f / scale_product) : 0.0f;

    if constexpr (Stochastic) {
        uint32_t rng_state = (uint32_t)seed 
            + ((uint32_t)offset * 0x9E3779B1u);
        rng_state = rng_state * 747796405u + 2891336453u;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            x[i] *= scale_inv;
            float noise = fast_rand(rng_state);
            x[i] = get_value_added_noise(x[i], noise);
            // ### another way (not safe enough):
            // float gap = get_fp4_gap(x[i]);
            // x[i] += noise * gap;
            // x[i] = rintf(x[i] / gap) * gap;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            x[i] *= scale_inv;
        }
    }

    float2 lo = {x[0], x[1]};
    float2 hi = {x[2], x[3]};
    uint8_t packed[2];
    packed[0] = __nv_cvt_float2_to_fp4x2(lo, __NV_E2M1, cudaRoundNearest);
    packed[1] = __nv_cvt_float2_to_fp4x2(hi, __NV_E2M1, cudaRoundNearest);
    
    // each thread store 2 bytes
    int out_offset = blk_id * ThreadNum * 2 + tid * 2;
    if (out_offset * 2 < n_elements) {
        *(uint16_t*)(x_q + out_offset) = *(uint16_t*)packed;
    }
}

template<bool Stochastic>
DEVICE float quant_fp4_from_reg(
    float x[], uint8_t packed[],
    uint8_t* g_x_q, int x_q_off,
    float* g_os, int os_off, uint8_t* g_is, int is_off,
    int tid, int offset, uint32_t seed
) {
    float outer_max = -INFINITY;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        outer_max = fmaxf(fabsf(x[i]), outer_max);
    }

    float inner_max = -INFINITY;
    outer_max = warpReduceScale(outer_max, inner_max);

    float outer_scale = outer_max / 448.0f / 6.0f;
    if (outer_scale == 0.0f) outer_scale = 1.0f;
    outer_scale = __shfl_sync(0xffffffff, outer_scale, 0);

    // store outer scale
    if (tid % 32 == 0) {
        g_os[os_off] = outer_scale;
    }

    float inner_scale = inner_max / outer_scale / 6.0f;
    if (inner_scale == 0.0f) inner_scale = 0.001953125f;

    __nv_fp8_storage_t inner_scale_fp8 = __nv_cvt_float_to_fp8(inner_scale, __NV_SATFINITE, __NV_E4M3);
    // store inner scale
    if (tid % 4 == 0) {
        g_is[is_off] = inner_scale_fp8;
    }

    inner_scale = (float)(*(__nv_fp8_e4m3*)&inner_scale_fp8);
    float scale_product = outer_scale * inner_scale;
    float scale_inv = (scale_product != 0.0f) ? (1.0f / scale_product) : 0.0f;
    
    if constexpr (Stochastic) {
        uint32_t rng_state = (uint32_t)seed 
            + ((uint32_t)offset * 0x9E3779B1u);
        rng_state = rng_state * 747796405u + 2891336453u;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            x[i] *= scale_inv;
            // stochastic rounding
            float noise = fast_rand(rng_state);
            x[i] = get_value_added_noise(x[i], noise);
            // ### another way (not safe enough):
            // float gap = get_fp4_gap(x[i]);
            // x[i] += noise * gap;
            // x[i] = rintf(x[i] / gap) * gap;
        }
    }
    else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            x[i] *= scale_inv;
        }
    }
    
    // pack
    float2 lo = {x[0], x[1]};
    float2 hi = {x[2], x[3]};
    packed[0] = __nv_cvt_float2_to_fp4x2(lo, __NV_E2M1, cudaRoundNearest);
    packed[1] = __nv_cvt_float2_to_fp4x2(hi, __NV_E2M1, cudaRoundNearest);
    
    // each thread store 2 bytes
    *(uint16_t*)(g_x_q + x_q_off / 2) = *((uint16_t*)packed);

    return scale_product;
}

// CtaM & CtaN should be the same as GEMM
template <
    int ThreadNum,
    int CtaM,
    int CtaN,
    bool Stochastic
>
__global__ void rh_nvfp4_double_quant_tile_major_fused(
    const float* __restrict__ g_x,
    uint8_t* __restrict__ g_x_q,
    float* __restrict__ g_os,
    uint8_t* __restrict__ g_is,
    int m, int n,
    uint32_t seed
) {
    // thread ~ 4 elements * Iter
    // warp ~ 128 elements * Iter
    constexpr int WarpNum = ThreadNum / 32;
    constexpr int WarpPerRow = CtaN / 128;
    constexpr int RowPerIter = WarpNum / WarpPerRow;
    constexpr int Iter = CtaM / RowPerIter;

    int tid = threadIdx.x;
    int lane_idx = tid % 32;
    int blk_row = blockIdx.y * CtaM;
    int blk_col = blockIdx.x * CtaN;
    int warp_idx = tid / 32;
    int warp_row = warp_idx / WarpPerRow;
    int warp_col = warp_idx % WarpPerRow * 128;
    const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;
    float x[Iter][4];
    uint8_t packed[Iter][4];
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row = blk_row + warp_row + it * RowPerIter;
        int offset = row * n + blk_col + warp_col + lane_idx * 4;
        float outer_max = -INFINITY;
        *((float4*)x + it) = *(float4*)(g_x + offset);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            outer_max = fmaxf(fabsf(x[it][i]), outer_max);
        }

        random_hadamard_16(x[it], lane_idx, seed);
        
        int os_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        quant_fp4_from_reg<Stochastic>(
            x[it], packed[it],
            g_x_q, offset,
            g_os, os_off, g_is, is_off,
            tid, offset, seed
        );
    }
}

// for dy
template <
    int ThreadNum,
    int CtaM,
    int CtaN
>
__global__ void rh_quant_fp4_trans_rh_requant_fp4_fp8_fused(
    const float* __restrict__ g_x,
    uint8_t* __restrict__ g_x_q,
    float* __restrict__ g_outer_scale,
    uint8_t* __restrict__ g_inner_scale,
    uint8_t* __restrict__ g_x_t_fp4_q,
    float* __restrict__ g_t_fp4_os,
    uint8_t* __restrict__ g_t_fp4_is,
    uint8_t* __restrict__ g_x_t_fp8_q,
    uint8_t* __restrict__ g_t_fp8_scale,
    int m, int n, uint32_t seed_dx, uint32_t seed_dw
) {
    extern __shared__ float x_t_smem[];

    // thread ~ 4 elements * Iter
    // warp ~ 128 elements * Iter
    constexpr int WarpNum = ThreadNum / 32;
    constexpr int WarpPerRow = CtaN / 128;
    constexpr int RowPerIter = WarpNum / WarpPerRow;
    constexpr int Iter = CtaM / RowPerIter;

    int tid = threadIdx.x;
    int lane_idx = tid % 32;
    int blk_row = blockIdx.y * CtaM;
    int blk_col = blockIdx.x * CtaN;
    int warp_idx = tid / 32;
    int warp_row = warp_idx / WarpPerRow;
    int warp_col = warp_idx % WarpPerRow * 128;
    const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;

    // quant
    float x[Iter][4];
    uint8_t packed[Iter][2];
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row_in_blk = warp_row + it * RowPerIter;
        int col_in_blk = warp_col + lane_idx * 4;
        int row = blk_row + row_in_blk;
        int col = blk_col + col_in_blk;
        int offset = row * n + col;

        *((float4*)x + it) = *(float4*)(g_x + offset);
        
        // store transposed matrix
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = get_swizzled_idx<CtaM>(col_in_blk + i, row_in_blk);
            x_t_smem[idx] = x[it][i];
        }

        random_hadamard_16(x[it], lane_idx, seed_dx);

        int os_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        quant_fp4_from_reg<true>(
            x[it], packed[it],
            g_x_q, offset,
            g_outer_scale, os_off, g_inner_scale, is_off,
            tid, offset, seed_dx
        );
    }

    __syncthreads();
    
    // requant
    float x_t[Iter][4], x_t_fp8[Iter][4];
    uint8_t packed_t[Iter][2];
    uint8_t x_t_fp8_q[Iter][4];
    const int blk_idx_t = blockIdx.x * gridDim.y + blockIdx.y;
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row_in_blk = warp_row + it * RowPerIter;
        int col_in_blk = warp_col + lane_idx * 4;
        int row = blk_col + row_in_blk;
        int col = blk_row + col_in_blk;
        int offset = row * m + col;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = get_swizzled_idx<CtaM>(row_in_blk, col_in_blk + i);
            x_t[it][i] = x_t_smem[idx];
        }
        *((float4*)x_t_fp8 + it) = *((float4*)x_t + it);

        // fp4
        random_hadamard_16(x_t[it], lane_idx, seed_dw);

        int os_off = (blk_idx_t * CtaN + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx_t * CtaN + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        quant_fp4_from_reg<true>(
            x_t[it], packed_t[it],
            g_x_t_fp4_q, offset,
            g_t_fp4_os, os_off, g_t_fp4_is, is_off,
            tid, offset, seed_dw
        );

        // fp8
        float group_max = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) group_max = fmax(group_max, fabs(x_t_fp8[it][i]));

        int shfl_size = 8;
        #pragma unroll
        for (int i = 1; i < shfl_size; i <<= 1) {
            float peer = __shfl_xor_sync(0xffffffff, group_max, i);
            group_max = fmax(group_max, fabs(peer));
        }

        float scale_fp = group_max / 448.0;
        uint8_t scale_e8m0 = _fp32_to_ue8m0(scale_fp);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            x_t_fp8[it][i] *= __uint_as_float(static_cast<uint32_t>(254u - scale_e8m0) << 23);
        _fp32x2_to_e4m3x2(&x_t_fp8[it][0], (uint16_t*)&x_t_fp8_q[it][0]);
        _fp32x2_to_e4m3x2(&x_t_fp8[it][2], (uint16_t*)&x_t_fp8_q[it][2]);

        *(uint32_t*)(g_x_t_fp8_q + offset) = *((uint32_t*)x_t_fp8_q + it);
        if (threadIdx.x % 8 == 0) {
            int scale_offset = (blk_idx_t * CtaN + it * RowPerIter) * WarpPerRow * 4 + tid / 8;
            *(g_t_fp8_scale + scale_offset) = scale_e8m0;
        }
    }
}

template <
    int ThreadNum,
    int CtaM,
    int CtaN
>
__global__ void rh_quant_fp4_trans_rh_requant_fused(
    const float* __restrict__ g_x,
    uint8_t* __restrict__ g_x_q,
    float* __restrict__ g_os,
    uint8_t* __restrict__ g_is,
    uint8_t* __restrict__ g_x_t_q,
    float* __restrict__ g_t_os,
    uint8_t* __restrict__ g_t_is,
    int m, int n, uint32_t seed_dx, uint32_t seed_dw
) {
    extern __shared__ float x_t_smem[];

    // thread ~ 4 elements * Iter
    // warp ~ 128 elements * Iter
    constexpr int WarpNum = ThreadNum / 32;
    constexpr int WarpPerRow = CtaN / 128;
    constexpr int RowPerIter = WarpNum / WarpPerRow;
    constexpr int Iter = CtaM / RowPerIter;

    int tid = threadIdx.x;
    int lane_idx = tid % 32;
    int blk_row = blockIdx.y * CtaM;
    int blk_col = blockIdx.x * CtaN;
    int warp_idx = tid / 32;
    int warp_row = warp_idx / WarpPerRow;
    int warp_col = warp_idx % WarpPerRow * 128;
    const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;

    float x[Iter][4];
    uint8_t packed[Iter][2];
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row_in_blk = warp_row + it * RowPerIter;
        int col_in_blk = warp_col + lane_idx * 4;
        int row = blk_row + row_in_blk;
        int col = blk_col + col_in_blk;
        int offset = row * n + col;

        *((float4*)x + it) = *(float4*)(g_x + offset);
        
        // store transposed matrix
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = get_swizzled_idx<CtaM>(col_in_blk + i, row_in_blk);
            x_t_smem[idx] = x[it][i];
        }

        random_hadamard_16(x[it], lane_idx, seed_dx);

        int os_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        quant_fp4_from_reg<true>(
            x[it], packed[it],
            g_x_q, offset,
            g_os, os_off, g_is, is_off,
            tid, offset, seed_dx
        );
    }

    __syncthreads();
    
    // requant
    float x_t[Iter][4];
    uint8_t packed_t[Iter][2];
    const int blk_idx_t = blockIdx.x * gridDim.y + blockIdx.y;
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row_in_blk = warp_row + it * RowPerIter;
        int col_in_blk = warp_col + lane_idx * 4;
        int row = blk_col + row_in_blk;
        int col = blk_row + col_in_blk;
        int offset = row * m + col;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = get_swizzled_idx<CtaM>(row_in_blk, col_in_blk + i);
            x_t[it][i] = x_t_smem[idx];
        }

        random_hadamard_16(x_t[it], lane_idx, seed_dw);
        
        int os_off = (blk_idx_t * CtaN + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx_t * CtaN + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        quant_fp4_from_reg<true>(
            x_t[it], packed_t[it],
            g_x_t_q, offset,
            g_t_os, os_off, g_t_is, is_off,
            tid, offset, seed_dw
        );
    }
}

// assume that CtaM = CtaN = 128
template <
    int ThreadNum,
    int CtaM,
    int CtaN
>
__global__ void nvfp4_double_quant_tile_major_dequant_trans_rh_requant_fused(
    const float* __restrict__ g_x,
    uint8_t* __restrict__ g_x_q,
    float* __restrict__ g_os,
    uint8_t* __restrict__ g_is,
    uint8_t* __restrict__ g_x_t_q,
    float* __restrict__ g_t_os,
    uint8_t* __restrict__ g_t_is,
    int m, int n, uint32_t seed
) {
    extern __shared__ float x_t_smem[];

    // thread ~ 4 elements * Iter
    // warp ~ 128 elements * Iter
    constexpr int WarpNum = ThreadNum / 32;
    constexpr int WarpPerRow = CtaN / 128;
    constexpr int RowPerIter = WarpNum / WarpPerRow;
    constexpr int Iter = CtaM / RowPerIter;

    int tid = threadIdx.x;
    int lane_idx = tid % 32;
    int blk_row = blockIdx.y * CtaM;
    int blk_col = blockIdx.x * CtaN;
    int warp_idx = tid / 32;
    int warp_row = warp_idx / WarpPerRow;
    int warp_col = warp_idx % WarpPerRow * 128;
    const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;
    float x[Iter][4];
    float x_dq[Iter][4];
    uint8_t packed[Iter][2];
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row = blk_row + warp_row + it * RowPerIter;
        int col = blk_col + warp_col + lane_idx * 4;
        int offset = row * n + col;
        *((float4*)x + it) = *(float4*)(g_x + offset);

        int os_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        float scale_product = quant_fp4_from_reg<false>(
            x[it], packed[it],
            g_x_q, offset,
            g_os, os_off, g_is, is_off,
            tid, offset, seed
        );

        // dequant
        x_dq[it][0] = fp4_to_float(packed[it][0] & 0x0f) * scale_product;
        x_dq[it][1] = fp4_to_float((packed[it][0] >> 4) & 0x0f) * scale_product;
        x_dq[it][2] = fp4_to_float(packed[it][1] & 0x0f) * scale_product;
        x_dq[it][3] = fp4_to_float((packed[it][1] >> 4) & 0x0f) * scale_product;

        int row_in_blk = warp_row + it * RowPerIter;
        int col_in_blk = warp_col + lane_idx * 4;

        // store dequanted transposed matrix
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = get_swizzled_idx<CtaM>(col_in_blk + i, row_in_blk);
            x_t_smem[idx] = x_dq[it][i];
        }
    }

    __syncthreads();
    
    // requant
    float x_t[Iter][4];
    uint8_t packed_t[Iter][2];
    const int blk_idx_t = blockIdx.x * gridDim.y + blockIdx.y;
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row_in_blk = warp_row + it * RowPerIter;
        int col_in_blk = warp_col + lane_idx * 4;
        int row = blk_col + row_in_blk;
        int col = blk_row + col_in_blk;
        int offset = row * m + col;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = get_swizzled_idx<CtaM>(row_in_blk, col_in_blk + i);
            x_t[it][i] = x_t_smem[idx];
        }

        random_hadamard_16(x_t[it], lane_idx, seed);
        
        int os_off = (blk_idx_t * CtaN + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx_t * CtaN + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        quant_fp4_from_reg<true>(
            x_t[it], packed_t[it],
            g_x_t_q, offset,
            g_t_os, os_off, g_t_is, is_off,
            tid, offset, seed
        );
    }
}

template <
    int ThreadNum,
    int CtaM,
    int CtaN
>
__global__ void nvfp4_double_quant_tile_major_dequant_trans_fused(
    const float* __restrict__ g_x,
    uint8_t* __restrict__ g_x_q,
    float* __restrict__ g_os,
    uint8_t* __restrict__ g_is,
    float* __restrict__ g_x_dq,
    int m, int n
) {
    extern __shared__ float x_t_smem[];

    // thread ~ 4 elements * Iter
    // warp ~ 128 elements * Iter
    constexpr int WarpNum = ThreadNum / 32;
    constexpr int WarpPerRow = CtaN / 128;
    constexpr int RowPerIter = WarpNum / WarpPerRow;
    constexpr int Iter = CtaM / RowPerIter;

    int tid = threadIdx.x;
    int lane_idx = tid % 32;
    int blk_row = blockIdx.y * CtaM;
    int blk_col = blockIdx.x * CtaN;
    int warp_idx = tid / 32;
    int warp_row = warp_idx / WarpPerRow;
    int warp_col = warp_idx % WarpPerRow * 128;
    const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;
    float x[Iter][4];
    float x_dq[Iter][4];
    uint8_t packed[Iter][2];
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row = blk_row + warp_row + it * RowPerIter;
        int col = blk_col + warp_col + lane_idx * 4;
        int offset = row * n + col;
        *((float4*)x + it) = *(float4*)(g_x + offset);
        
        int os_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        float scale_product = quant_fp4_from_reg<false>(
            x[it], packed[it],
            g_x_q, offset,
            g_os, os_off, g_is, is_off,
            tid, offset, 0
        );

        // dequant
        x_dq[it][0] = fp4_to_float(packed[it][0] & 0x0f) * scale_product;
        x_dq[it][1] = fp4_to_float((packed[it][0] >> 4) & 0x0f) * scale_product;
        x_dq[it][2] = fp4_to_float(packed[it][1] & 0x0f) * scale_product;
        x_dq[it][3] = fp4_to_float((packed[it][1] >> 4) & 0x0f) * scale_product;

        int row_in_blk = warp_row + it * RowPerIter;
        int col_in_blk = warp_col + lane_idx * 4;

        // store dequanted transposed matrix
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = get_swizzled_idx<CtaM>(col_in_blk + i, row_in_blk);
            x_t_smem[idx] = x_dq[it][i];
        }
    }

    __syncthreads();
    float x_trans[Iter][4];
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row_in_blk = warp_row + it * RowPerIter;
        int col_in_blk = warp_col + lane_idx * 4;
        int trans_row = blk_col + row_in_blk;
        int trans_col = blk_row + col_in_blk;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = get_swizzled_idx<CtaM>(row_in_blk, col_in_blk + i);
            x_trans[it][i] = x_t_smem[idx];
        }
        int offset = trans_row * m + trans_col;
        *(float4*)(g_x_dq + offset) = *((float4*)x_trans + it);
    }
}

// CtaM & CtaN should be the same as GEMM
template <
    int ThreadNum,
    int CtaM,
    int CtaN,
    bool Stochastic
>
__global__ void nvfp4_double_quant_tile_major(
    const float* __restrict__ g_x,
    uint8_t* __restrict__ g_x_q,
    float* __restrict__ g_os,
    uint8_t* __restrict__ g_is,
    int m, int n,
    uint32_t seed
) {
    // thread ~ 4 elements * Iter
    // warp ~ 128 elements * Iter
    constexpr int WarpNum = ThreadNum / 32;
    constexpr int WarpPerRow = CtaN / 128;
    constexpr int RowPerIter = WarpNum / WarpPerRow;
    constexpr int Iter = CtaM / RowPerIter;

    int tid = threadIdx.x;
    int lane_idx = tid % 32;
    int blk_row = blockIdx.y * CtaM;
    int blk_col = blockIdx.x * CtaN;
    int warp_idx = tid / 32;
    int warp_row = warp_idx / WarpPerRow;
    int warp_col = warp_idx % WarpPerRow * 128;
    const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;
    float x[Iter][4];
    uint8_t packed[Iter][2];
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row = blk_row + warp_row + it * RowPerIter;
        int offset = row * n + blk_col + warp_col + lane_idx * 4;
        *((float4*)x + it) = *(float4*)(g_x + offset);

        int os_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        quant_fp4_from_reg<Stochastic>(
            x[it], packed[it],
            g_x_q, offset,
            g_os, os_off, g_is, is_off,
            tid, offset, seed
        );
    }
}

// dequant
template <
    int ThreadNum,
    int CtaM,
    int CtaN
>
__global__ void dequant_fp4_rh_requant_tile_major_fused(
    uint8_t* __restrict__ g_x,
    float* __restrict__ g_os,
    uint8_t* __restrict__ g_is,
    uint8_t* __restrict__ s_x_q,
    float* __restrict__ s_os,
    uint8_t* __restrict__ s_is,
    int m, int n,
    uint32_t seed
) {
    // thread ~ 4 elements * Iter
    // warp ~ 128 elements * Iter
    constexpr int WarpNum = ThreadNum / 32;
    constexpr int WarpPerRow = CtaN / 128;
    constexpr int RowPerIter = WarpNum / WarpPerRow;
    constexpr int Iter = CtaM / RowPerIter;
    const int n_elements = m * n;

    int tid = threadIdx.x;
    int lane_idx = tid % 32;
    int blk_row = blockIdx.y * CtaM;
    int blk_col = blockIdx.x * CtaN;
    int warp_idx = tid / 32;
    int warp_row = warp_idx / WarpPerRow;
    int warp_col = warp_idx % WarpPerRow * 128;
    const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;
    float x[Iter][4];
    uint8_t x_q[Iter][2];
    uint8_t packed[Iter][2];
    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row = blk_row + warp_row + it * RowPerIter;
        int col = blk_col + warp_col + lane_idx * 4;
        int offset = row * n + col;

        if (offset < n_elements) {
            *((uint16_t*)x_q + it) = *(uint16_t*)(g_x + offset / 2);
            x[it][0] = fp4_to_float(x_q[it][0] & 0x0f);
            x[it][1] = fp4_to_float((x_q[it][0] >> 4) & 0x0f);
            x[it][2] = fp4_to_float(x_q[it][1] & 0x0f);
            x[it][3] = fp4_to_float((x_q[it][1] >> 4) & 0x0f);
        }

        float outer_scale = 0.0f;
        if (tid % 32 == 0) {
            int outer_scale_offset = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
            outer_scale = g_os[outer_scale_offset];
        }
        outer_scale = __shfl_sync(0xffffffff, outer_scale, 0);

        float inner_scale = 0.0f;
        if (tid % 4 == 0) {
            int inner_scale_offset = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
            inner_scale = (float)(*(__nv_fp8_e4m3*)(g_is + inner_scale_offset));
        }
        inner_scale = __shfl_sync(0xffffffff, inner_scale, lane_idx / 4 * 4);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            x[it][i] *= outer_scale * inner_scale;
        }

        random_hadamard_16(x[it], lane_idx, seed);

        int os_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
        int is_off = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
        quant_fp4_from_reg<true>(
            x[it], packed[it],
            s_x_q, offset,
            s_os, os_off, s_is, is_off,
            tid, offset, seed
        );
    }
}

template <int ThreadNum, int CtaM, int CtaN>
__global__ void nvfp4_dequant_tile_major(
    const uint8_t* __restrict__ g_x,
    float* __restrict__ x_dq,
    const float* __restrict__ g_outer_scale,
    const uint8_t* __restrict__ g_inner_scale,
    int m, int n
) {
    // thread ~ 4 elements
    // warp ~ 128 elements
    constexpr int WarpNum = ThreadNum / 32;
    constexpr int WarpPerRow = CtaN / 128;
    constexpr int RowPerIter = WarpNum / WarpPerRow;
    constexpr int Iter = CtaM / RowPerIter;

    const int n_elements = m * n;
    int tid = threadIdx.x;
    int lane_idx = tid % 32;
    int blk_row = blockIdx.y * CtaM;
    int blk_col = blockIdx.x * CtaN;
    int warp_idx = tid / 32;
    int warp_row = warp_idx / WarpPerRow;
    int warp_col = warp_idx % WarpPerRow * 128;
    const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint8_t x[Iter][2];
    float output[Iter][4];

    #pragma unroll
    for (int it = 0; it < Iter; it++) {
        int row = blk_row + warp_row + it * RowPerIter;
        int offset = row * n + blk_col + warp_col + lane_idx * 4;

        if (offset < n_elements) {
            *((uint16_t*)x + it) = *(uint16_t*)(g_x + offset / 2);
            output[it][0] = fp4_to_float(x[it][0] & 0x0f);
            output[it][1] = fp4_to_float((x[it][0] >> 4) & 0x0f);
            output[it][2] = fp4_to_float(x[it][1] & 0x0f);
            output[it][3] = fp4_to_float((x[it][1] >> 4) & 0x0f);
        }

        float outer_scale = 0.0f;
        if (tid % 32 == 0) {
            int outer_scale_offset = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow + warp_idx;
            outer_scale = g_outer_scale[outer_scale_offset];
        }
        outer_scale = __shfl_sync(0xffffffff, outer_scale, 0);

        float inner_scale = 0.0f;
        if (tid % 4 == 0) {
            int inner_scale_offset = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 8 + tid / 4;
            inner_scale = (float)(*(__nv_fp8_e4m3*)(g_inner_scale + inner_scale_offset));
        }
        inner_scale = __shfl_sync(0xffffffff, inner_scale, lane_idx / 4 * 4);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            output[it][i] *= outer_scale * inner_scale;
        }

        if (offset < n_elements) {
            *(float4*)(x_dq + offset) = *((float4*)output + it);
        }
    }
}

template <int ThreadNum>
__global__ void nvfp4_dequant(
    const uint8_t* __restrict__ g_x,
    float* __restrict__ x_dq,
    const float* __restrict__ g_outer_scale,
    const uint8_t* __restrict__ g_inner_scale,
    int n_elements
) {
    int tid = threadIdx.x;
    int blk_id = blockIdx.x;
    int lane_idx = tid % 32;
    // each thread handles 4 fp4 elements, which is 2 bytes
    // every warp share a outer scale
    // each 4 threads share inner_scale
    int offset = blk_id * ThreadNum * 2 + tid * 2;
    uint8_t x[2];
    float output[4];
    if (offset * 2 < n_elements) {
        *(uint16_t*)x = *(uint16_t*)(g_x + offset);
        output[0] = fp4_to_float(x[0] & 0x0f);
        output[1] = fp4_to_float((x[0] >> 4) & 0x0f);
        output[2] = fp4_to_float(x[1] & 0x0f);
        output[3] = fp4_to_float((x[1] >> 4) & 0x0f);
    }

    float outer_scale = 0.0f;
    if (tid % 32 == 0) {
        int outer_scale_offset = blk_id * ThreadNum / 32 + tid / 32;
        outer_scale = g_outer_scale[outer_scale_offset];
    }
    outer_scale = __shfl_sync(0xffffffff, outer_scale, 0);

    float inner_scale = 0.0f;
    if (tid % 4 == 0) {
        int inner_scale_offset = blk_id * ThreadNum / 4 + tid / 4;
        inner_scale = (float)(*((__nv_fp8_e4m3*)g_inner_scale + inner_scale_offset));
    }
    inner_scale = __shfl_sync(0xffffffff, inner_scale, lane_idx / 4 * 4);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        output[i] *= outer_scale * inner_scale;
    }

    int out_offset = blk_id * ThreadNum * 4 + tid * 4;
    if (out_offset < n_elements) {
        *(float4*)(x_dq + out_offset) = *(float4*)output;
    }
}

// for test
__global__ void unpack_fp4_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        uint8_t packed = input[idx];
        uint8_t lo = packed & 0x0f;
        uint8_t hi = (packed >> 4) & 0x0f;
        output[idx * 2] = fp4_to_float(lo);
        output[idx * 2 + 1] = fp4_to_float(hi);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////


// quant launch
void nvfp4_double_quant_scale_only_launch(
    const float* g_x, float* x_q, int n_elements, cudaStream_t stream
) {
    constexpr int ThreadNum = 128;
    dim3 grid_dim(cdiv(n_elements, ThreadNum * 4));
    nvfp4_double_quantized_scale_only<ThreadNum>
        <<<grid_dim, ThreadNum, 0, stream>>>(g_x, x_q, n_elements);
}

template<bool Stochastic>
void rh_nvfp4_double_quant_tile_major_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
) {
    constexpr int ThreadNum = 256;
    constexpr int CtaM = 128;
    constexpr int CtaN = 128;
    dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
    rh_nvfp4_double_quant_tile_major_fused<ThreadNum, CtaM, CtaN, Stochastic>
        <<<grid_dim, ThreadNum, 0, stream>>>(g_x, x_q, outer_scale, inner_scale, m, n, seed);
}

void rh_quant_fp4_trans_rh_requant_fp4_fp8_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    uint8_t* x_t_fp4_q, float* t_fp4_outer_scale, uint8_t* t_fp4_inner_scale,
    uint8_t* x_t_fp8_q, uint8_t* t_fp8_scale,
    int m, int n, uint32_t seed_dx, uint32_t seed_dw, cudaStream_t stream
) {
    constexpr int ThreadNum = 256;
    constexpr int CtaM = 128;
    constexpr int CtaN = 128;
    dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
    size_t smem_size = sizeof(float) * CtaM * CtaN;
    cudaFuncSetAttribute(
        rh_quant_fp4_trans_rh_requant_fp4_fp8_fused<ThreadNum, CtaM, CtaN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    rh_quant_fp4_trans_rh_requant_fp4_fp8_fused<ThreadNum, CtaM, CtaN>
        <<<grid_dim, ThreadNum, smem_size, stream>>>(
            g_x, x_q, outer_scale, inner_scale,
            x_t_fp4_q, t_fp4_outer_scale, t_fp4_inner_scale,
            x_t_fp8_q, t_fp8_scale,
            m, n, seed_dx, seed_dw
        );
}

void rh_quant_fp4_trans_rh_requant_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    uint8_t* x_t_q, float* t_outer_scale, uint8_t* t_inner_scale,
    int m, int n, uint32_t seed_dx, uint32_t seed_dw, cudaStream_t stream
) {
    constexpr int ThreadNum = 256;
    constexpr int CtaM = 128;
    constexpr int CtaN = 128;
    dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
    size_t smem_size = sizeof(float) * CtaM * CtaN;
    cudaFuncSetAttribute(
        rh_quant_fp4_trans_rh_requant_fused<ThreadNum, CtaM, CtaN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    rh_quant_fp4_trans_rh_requant_fused<ThreadNum, CtaM, CtaN>
        <<<grid_dim, ThreadNum, smem_size, stream>>>(
            g_x, x_q, outer_scale, inner_scale, x_t_q, t_outer_scale, t_inner_scale, m, n, seed_dx, seed_dw
        );
}

void nvfp4_double_quant_tile_major_dequant_trans_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    float* g_x_dq,
    int m, int n, cudaStream_t stream
) {
    constexpr int ThreadNum = 256;
    constexpr int CtaM = 128;
    constexpr int CtaN = 128;
    dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
    size_t smem_size = sizeof(float) * CtaM * CtaN;
    cudaFuncSetAttribute(
        nvfp4_double_quant_tile_major_dequant_trans_fused<ThreadNum, CtaM, CtaN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    nvfp4_double_quant_tile_major_dequant_trans_fused<ThreadNum, CtaM, CtaN>
        <<<grid_dim, ThreadNum, smem_size, stream>>>(g_x, x_q, outer_scale, inner_scale, g_x_dq, m, n);
}

void nvfp4_double_quant_tile_major_dequant_trans_rh_requant_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    uint8_t* x_t_q, float* t_outer_scale, uint8_t* t_inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
) {
    constexpr int ThreadNum = 256;
    constexpr int CtaM = 128;
    constexpr int CtaN = 128;
    dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
    size_t smem_size = sizeof(float) * CtaM * CtaN;
    cudaFuncSetAttribute(
        nvfp4_double_quant_tile_major_dequant_trans_rh_requant_fused<ThreadNum, CtaM, CtaN>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    nvfp4_double_quant_tile_major_dequant_trans_rh_requant_fused<ThreadNum, CtaM, CtaN>
        <<<grid_dim, ThreadNum, smem_size, stream>>>(
            g_x, x_q, outer_scale, inner_scale, x_t_q, t_outer_scale, t_inner_scale, m, n, seed
        );
}

template<bool Stochastic>
void nvfp4_double_quant_tile_major_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
) {
    constexpr int ThreadNum = 256;
    constexpr int CtaM = 128;
    constexpr int CtaN = 128;
    dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
    nvfp4_double_quant_tile_major<ThreadNum, CtaM, CtaN, Stochastic>
        <<<grid_dim, ThreadNum, 0, stream>>>(g_x, x_q, outer_scale, inner_scale, m, n, seed);
}

template<bool Stochastic>
void nvfp4_double_quant_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int n_elements, uint32_t seed, cudaStream_t stream
) {
    constexpr int ThreadNum = 128;
    dim3 grid_dim(cdiv(n_elements, ThreadNum * 4));
    nvfp4_double_quant<ThreadNum, Stochastic>
        <<<grid_dim, ThreadNum, 0, stream>>>(g_x, x_q, outer_scale, inner_scale, n_elements, seed);
}

// dequant launch

void dequant_fp4_rh_requant_tile_major_fused_launch(
    uint8_t* __restrict__ g_x,
    float* __restrict__ g_outer_scale,
    uint8_t* __restrict__ g_inner_scale,
    uint8_t* __restrict__ s_x_q,
    float* __restrict__ s_outer_scale,
    uint8_t* __restrict__ s_inner_scale,
    int m, int n,
    uint32_t seed, cudaStream_t stream
) {
    constexpr int ThreadNum = 256;
    constexpr int CtaM = 128;
    constexpr int CtaN = 128;
    dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
    dequant_fp4_rh_requant_tile_major_fused<ThreadNum, CtaM, CtaN>
        <<<grid_dim, ThreadNum, 0, stream>>>(
            g_x, g_outer_scale, g_inner_scale, s_x_q, s_outer_scale, s_inner_scale, m, n, seed
        );
}

void nvfp4_dequant_launch(
    const uint8_t* g_x,
    float* x_dq, float* outer_scale, uint8_t* inner_scale,
    int n_elements, cudaStream_t stream
) {
    constexpr int ThreadNum = 128;
    dim3 grid_dim(cdiv(n_elements, ThreadNum * 4));
    nvfp4_dequant<ThreadNum>
        <<<grid_dim, ThreadNum, 0, stream>>>(g_x, x_dq, outer_scale, inner_scale, n_elements);
}

void nvfp4_dequant_tile_major_launch(
    const uint8_t* g_x,
    float* x_dq, float* outer_scale, uint8_t* inner_scale,
    int m, int n, cudaStream_t stream
) {
    constexpr int ThreadNum = 256;
    constexpr int CtaM = 128;
    constexpr int CtaN = 128;
    dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
    nvfp4_dequant_tile_major<ThreadNum, CtaM, CtaN>
        <<<grid_dim, ThreadNum, 0, stream>>>(g_x, x_dq, outer_scale, inner_scale, m, n);
}


void unpack_fp4_launch(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int n_elements, cudaStream_t stream
) {
    const int threads = 128;
    const int blocks = (n_elements + threads - 1) / threads;

    unpack_fp4_kernel<<<blocks, threads, 0, stream>>>(
        input, output, n_elements
    );
}


//////////////////////////////////////////////////////////////////////////////////////////


template void nvfp4_double_quant_tile_major_launch<true>(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
);

template void nvfp4_double_quant_tile_major_launch<false>(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
);
template void nvfp4_double_quant_launch<true>(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int n_elements, uint32_t seed, cudaStream_t stream
);

template void nvfp4_double_quant_launch<false>(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int n_elements, uint32_t seed, cudaStream_t stream
);

template void rh_nvfp4_double_quant_tile_major_fused_launch<true>(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
);

template void rh_nvfp4_double_quant_tile_major_fused_launch<false>(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
);