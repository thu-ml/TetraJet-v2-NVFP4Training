#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "utils.hpp"

template<int ThreadNum, int CtaM, int CtaN>
__global__ void quant_fp8_dequant_trans_requant_fused(
  const float *g_x, uint8_t *g_x_q, uint8_t *scale, 
  uint8_t* g_x_t_q, uint8_t* scale_t,
  int m, int n
) {
  extern __shared__ float x_trans_smem[];
  // thread ~ 4 elements * Iter
  // warp ~ 128 elements * Iter
  constexpr int WarpNum = ThreadNum / 32;
  constexpr int WarpPerRow = CtaN / 128;
  constexpr int RowPerIter = WarpNum / WarpPerRow;
  constexpr int Iter = CtaM / RowPerIter;
  const int n_elements = m * n;
  const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;

  int tid = threadIdx.x;
  int lane_idx = tid % 32;
  int blk_row = blockIdx.y * CtaM;
  int blk_col = blockIdx.x * CtaN;
  int warp_idx = tid / 32;
  int warp_row = warp_idx / WarpPerRow;
  int warp_col = warp_idx % WarpPerRow * 128;

  float x[Iter][4];
  uint8_t x_q[Iter][4];
  float x_dq[Iter][4];
  for (int it = 0; it < Iter; it++) {
    int row_in_blk = warp_row + it * RowPerIter;
    int col_in_blk = warp_col + lane_idx * 4;
    int row = blk_row + row_in_blk;
    int col = blk_col + col_in_blk;
    int offset = row * n + col;

    if (offset < n_elements) {
      *((float4*)x + it) = *(float4*)(g_x + offset);
    }

    float group_max = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) group_max = fmax(group_max, fabs(x[it][i]));

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
      x[it][i] *= __uint_as_float(static_cast<uint32_t>(254u - scale_e8m0) << 23);
    _fp32x2_to_e4m3x2(&x[it][0], (uint16_t*)&x_q[it][0]);
    _fp32x2_to_e4m3x2(&x[it][2], (uint16_t*)&x_q[it][2]);

    *(uint32_t*)(g_x_q + offset) = *((uint32_t*)x_q + it);
    if (threadIdx.x % 8 == 0) {
      int scale_offset = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 4 + tid / 8;
      *(scale + scale_offset) = scale_e8m0;
    }

    // dequant
    scale_fp = (float)(*(__nv_fp8_e8m0*)(&scale_e8m0));
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      x_dq[it][i] = (float)(*(__nv_fp8_e4m3*)(&x_q[it][i])) * scale_fp;
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int idx = get_swizzled_idx<CtaM>(col_in_blk + i, row_in_blk);
      x_trans_smem[idx] = x_dq[it][i];
    }
  }

  __syncthreads();
  // requant
  float x_t[Iter][4];
  uint8_t x_t_q[Iter][4];
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
      x_t[it][i] = x_trans_smem[idx];
    }
    
    float group_max = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) group_max = fmax(group_max, fabs(x_t[it][i]));

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
      x_t[it][i] *= __uint_as_float(static_cast<uint32_t>(254u - scale_e8m0) << 23);
    _fp32x2_to_e4m3x2(&x_t[it][0], (uint16_t*)&x_t_q[it][0]);
    _fp32x2_to_e4m3x2(&x_t[it][2], (uint16_t*)&x_t_q[it][2]);

    *(uint32_t*)(g_x_t_q + offset) = *((uint32_t*)x_t_q + it);
    if (threadIdx.x % 8 == 0) {
      int scale_offset = (blk_idx_t * CtaN + it * RowPerIter) * WarpPerRow * 4 + tid / 8;
      *(scale_t + scale_offset) = scale_e8m0;
    }
  }
}

template<int ThreadNum, int CtaM, int CtaN>
__global__ void quant_fp8_tile_major(
  const float *g_x, uint8_t *g_x_q, uint8_t *scale, 
  int m, int n
) {
  // thread ~ 4 elements * Iter
  // warp ~ 128 elements * Iter
  constexpr int WarpNum = ThreadNum / 32;
  constexpr int WarpPerRow = CtaN / 128;
  constexpr int RowPerIter = WarpNum / WarpPerRow;
  constexpr int Iter = CtaM / RowPerIter;
  const int n_elements = m * n;
  const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;

  int tid = threadIdx.x;
  int lane_idx = tid % 32;
  int blk_row = blockIdx.y * CtaM;
  int blk_col = blockIdx.x * CtaN;
  int warp_idx = tid / 32;
  int warp_row = warp_idx / WarpPerRow;
  int warp_col = warp_idx % WarpPerRow * 128;

  float x[Iter][4];
  uint8_t x_q[Iter][4];
  for (int it = 0; it < Iter; it++) {
    int row_in_blk = warp_row + it * RowPerIter;
    int col_in_blk = warp_col + lane_idx * 4;
    int row = blk_row + row_in_blk;
    int col = blk_col + col_in_blk;
    int offset = row * n + col;

    if (offset < n_elements) {
      *((float4*)x + it) = *(float4*)(g_x + offset);
    }

    float group_max = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) group_max = fmax(group_max, fabs(x[it][i]));

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
      x_q[it][i] = _fp32_to_e4m3(
        x[it][i] * __uint_as_float(static_cast<uint32_t>(254u - scale_e8m0) << 23)
      );

    *(uint32_t*)(g_x_q + offset) = *((uint32_t*)x_q + it);
    if (threadIdx.x % 8 == 0) {
      int scale_offset = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 4 + tid / 8;
      *(scale + scale_offset) = scale_e8m0;
    }
  }
}

template<int ThreadNum, int CtaM, int CtaN>
__global__ void quant_fp8(
  const float *g_x, uint8_t *g_x_q, uint8_t *scale, 
  int m, int n
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

  float x[Iter][4];
  uint8_t x_q[Iter][4];
  for (int it = 0; it < Iter; it++) {
    int row_in_blk = warp_row + it * RowPerIter;
    int col_in_blk = warp_col + lane_idx * 4;
    int row = blk_row + row_in_blk;
    int col = blk_col + col_in_blk;
    int offset = row * n + col;

    if (offset < n_elements) {
      *((float4*)x + it) = *(float4*)(g_x + offset);
    }

    float group_max = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) group_max = fmax(group_max, fabs(x[it][i]));

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
      x_q[it][i] = _fp32_to_e4m3(
        x[it][i] * __uint_as_float(static_cast<uint32_t>(254u - scale_e8m0) << 23)
      );

    *(uint32_t*)(g_x_q + offset) = *((uint32_t*)x_q + it);
    if (threadIdx.x % 8 == 0) {
      *(scale + offset / 32) = scale_e8m0;
    }
  }
}

template<int ThreadNum, int CtaM, int CtaN>
__global__ void dequant_fp8_tile_major(
  const uint8_t* g_x_q,
  const uint8_t* g_scale,
  float* g_x_dq,
  int m, int n
) {
  // thread ~ 4 elements * Iter
  // warp ~ 128 elements * Iter
  constexpr int WarpNum = ThreadNum / 32;
  constexpr int WarpPerRow = CtaN / 128;
  constexpr int RowPerIter = WarpNum / WarpPerRow;
  constexpr int Iter = CtaM / RowPerIter;
  const int n_elements = m * n;
  const int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;

  int tid = threadIdx.x;
  int lane_idx = tid % 32;
  int blk_row = blockIdx.y * CtaM;
  int blk_col = blockIdx.x * CtaN;
  int warp_idx = tid / 32;
  int warp_row = warp_idx / WarpPerRow;
  int warp_col = warp_idx % WarpPerRow * 128;

  uint8_t x_q[Iter][4];
  float x[Iter][4];
  for (int it = 0; it < Iter; it++) {
    int row_in_blk = warp_row + it * RowPerIter;
    int col_in_blk = warp_col + lane_idx * 4;
    int row = blk_row + row_in_blk;
    int col = blk_col + col_in_blk;
    int offset = row * n + col;

    if (offset < n_elements) {
      *((uint32_t*)x_q + it) = *(uint32_t*)(g_x_q + offset);
    }

    float scale = 0.0f;
    if (tid % 8 == 0) {
      int scale_offset = (blk_idx * CtaM + it * RowPerIter) * WarpPerRow * 4 + tid / 8;
      scale = (float)(*(__nv_fp8_e8m0*)(g_scale + scale_offset));
    }
    scale = __shfl_sync(0xffffffff, scale, tid / 8 * 8);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      x[it][i] = (float)(*(__nv_fp8_e4m3*)(&x_q[it][i])) * scale;
    }

    *(float4*)(g_x_dq + offset) = *((float4*)x + it);
  }
}

template<int ThreadNum, int CtaM, int CtaN>
__global__ void dequant_fp8(
  const uint8_t* g_x_q,
  const uint8_t* g_scale,
  float* g_x_dq,
  int m, int n
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

  uint8_t x_q[Iter][4];
  float x[Iter][4];
  for (int it = 0; it < Iter; it++) {
    int row_in_blk = warp_row + it * RowPerIter;
    int col_in_blk = warp_col + lane_idx * 4;
    int row = blk_row + row_in_blk;
    int col = blk_col + col_in_blk;
    int offset = row * n + col;

    if (offset < n_elements) {
      *((uint32_t*)x_q + it) = *(uint32_t*)(g_x_q + offset);
    }

    float scale = 0.0f;
    if (tid % 8 == 0) {
      scale = (float)(*(__nv_fp8_e8m0*)(g_scale + offset / 32));
    }
    scale = __shfl_sync(0xffffffff, scale, tid / 8 * 8);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      x[it][i] = (float)(*(__nv_fp8_e4m3*)(&x_q[it][i])) * scale;
    }

    *(float4*)(g_x_dq + offset) = *((float4*)x + it);
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////


void quant_fp8_dequant_trans_requant_fused_launch(
    const float* g_x, uint8_t* x_q, uint8_t* scale,
    uint8_t* x_t_q, uint8_t* scale_t,
    int m, int n, cudaStream_t stream
) {
  constexpr int ThreadNum = 256;
  constexpr int CtaM = 128;
  constexpr int CtaN = 128;
  dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
  size_t smem_size = sizeof(float) * CtaM * CtaN;
  cudaFuncSetAttribute(
    quant_fp8_dequant_trans_requant_fused<ThreadNum, CtaM, CtaN>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size
  );
  quant_fp8_dequant_trans_requant_fused<ThreadNum, CtaM, CtaN>
    <<<grid_dim, ThreadNum, smem_size, stream>>>(g_x, x_q, scale, x_t_q, scale_t, m, n);
}

void quant_fp8_launch(
    const float* g_x, uint8_t* x_q, uint8_t* scale,
    int m, int n, cudaStream_t stream
) {
  constexpr int ThreadNum = 256;
  constexpr int CtaM = 128;
  constexpr int CtaN = 128;
  dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
  quant_fp8<ThreadNum, CtaM, CtaN>
    <<<grid_dim, ThreadNum, 0, stream>>>(g_x, x_q, scale, m, n);
}

void quant_fp8_tile_major_launch(
    const float* g_x, uint8_t* x_q, uint8_t* scale,
    int m, int n, cudaStream_t stream
) {
  constexpr int ThreadNum = 256;
  constexpr int CtaM = 128;
  constexpr int CtaN = 128;
  dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
  quant_fp8_tile_major<ThreadNum, CtaM, CtaN>
    <<<grid_dim, ThreadNum, 0, stream>>>(g_x, x_q, scale, m, n);
}

void dequant_fp8_tile_major_launch(
  const uint8_t* x_q,
  const uint8_t* g_scale,
  float* x_dq,
  int m, int n, cudaStream_t stream
) {
  constexpr int ThreadNum = 256;
  constexpr int CtaM = 128;
  constexpr int CtaN = 128;
  dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
  dequant_fp8_tile_major<ThreadNum, CtaM, CtaN>
    <<<grid_dim, ThreadNum, 0, stream>>>(x_q, g_scale, x_dq, m, n);
}

void dequant_fp8_launch(
  const uint8_t* x_q,
  const uint8_t* g_scale,
  float* x_dq,
  int m, int n, cudaStream_t stream
) {
  constexpr int ThreadNum = 256;
  constexpr int CtaM = 128;
  constexpr int CtaN = 128;
  dim3 grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
  dequant_fp8<ThreadNum, CtaM, CtaN>
    <<<grid_dim, ThreadNum, 0, stream>>>(x_q, g_scale, x_dq, m, n);
}