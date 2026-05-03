#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda.h>

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__

#define cdiv(a, b) (((a) + (b) - 1) / (b))

template <class T>
HOST_DEVICE T max(const T& a, const T& b) {
    return (a > b ? a : b);
}

template <class T>
HOST_DEVICE T min(const T& a, const T& b) {
    return (a < b ? a : b);
}

// TensorMap
inline CUtensorMap make_tensormap_fp4(
  void *ptr, uint64_t m, uint64_t n,
  uint32_t cta_m, uint32_t cta_n
) {
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {n / 2, m};
  uint64_t stride[rank - 1] = {n / 2 * sizeof(uint8_t)};
  uint32_t box_size[rank] = {cta_n / 2, cta_m};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    rank,
    ptr,
    size,
    stride,
    box_size,
    elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  return tensor_map;
}

inline CUtensorMap make_tensormap_fp8(
  void *ptr, uint64_t m, uint64_t n,
  uint32_t cta_m, uint32_t cta_n
) {
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {n, m};
  uint64_t stride[rank - 1] = {n * sizeof(uint8_t)};
  uint32_t box_size[rank] = {cta_n, cta_m};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    rank,
    ptr,
    size,
    stride,
    box_size,
    elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  return tensor_map;
}

inline CUtensorMap make_tensormap_bf16(
  void *ptr, uint64_t m, uint64_t n,
  uint32_t cta_m, uint32_t cta_n
) {
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {n, m};
  uint64_t stride[rank - 1] = {n * sizeof(uint16_t)};
  uint32_t box_size[rank] = {cta_n, cta_m};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    ptr,
    size,
    stride,
    box_size,
    elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  return tensor_map;
}

DEVICE
uint8_t _fp32_to_e4m3(float const& x) {
  uint16_t out;
  asm volatile( \
    "{\n" \
    "cvt.rn.satfinite.e4m3x2.f32   %0, %2, %1;\n" \
    "}" \
    : "=h"(out) : "f"(x), "f"(x));

  return static_cast<uint8_t>(out);
}

DEVICE void _fp32x2_to_e4m3x2(float* in, uint16_t* out) {
  asm volatile(
    "cvt.rn.satfinite.e4m3x2.f32   %0, %2, %1;\n"
    : "=h"(*out)
    : "f"(in[0]), "f"(in[1])
  );
}

DEVICE
uint8_t _fp32_to_ue8m0(float const& x) {
  uint16_t out;
  asm volatile( \
    "{\n" \
    "cvt.rp.satfinite.ue8m0x2.f32   %0, %2, %1;\n" \
    "}" \
    : "=h"(out) : "f"(x), "f"(x));
  return static_cast<uint8_t>(out);
}

template<int CtaN>
HOST_DEVICE int get_swizzled_idx(int row, int col) {
    int swizzled_row_idx = row / 4;
    int swizzled_col_idx = col % 32;
    int swizzled_col = swizzled_col_idx ^ swizzled_row_idx + col / 32 * 32;
    return row * CtaN + swizzled_col;
}