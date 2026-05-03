#include <cuda.h>
#include <cuda_bf16.h>
#include <random>
#include <tuple>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "utils.hpp"

namespace fp8 {
constexpr int CtaM = 128;
constexpr int CtaN = 128;
constexpr int CtaK = 128;
constexpr int EpiM = 128;
constexpr int EpiN = 64;
constexpr int NumThreadsPerWarp = 32;
constexpr int NumThreadsPerWarpGroup = 128;
constexpr int Stages = 2;
// 2 * 4 warps
constexpr int WorkerRepM = 2;
constexpr int WorkerRepN = 4;
// m16n8k32 * 2 * 4
constexpr int AtomM = 32;
constexpr int AtomN = 32;
constexpr int AtomK = 32;
constexpr int AtomRegA = AtomM * AtomK / NumThreadsPerWarp / 4;
constexpr int AtomRegB = AtomN * AtomK / NumThreadsPerWarp / 4;
constexpr int AtomRegC = AtomM * AtomN / NumThreadsPerWarp;
constexpr int AtomRepM = CtaM / WorkerRepM / AtomM;
constexpr int AtomRepN = CtaN / WorkerRepN / AtomN;
constexpr int AtomRepK = CtaK / AtomK;
constexpr int WorkerM = CtaM / WorkerRepM;
constexpr int WorkerN = CtaN / WorkerRepN;

DEVICE
size_t worker_m_offset(uint32_t widx) {
  return (widx / WorkerRepN) * WorkerM;
}

DEVICE
size_t worker_n_offset(uint32_t widx) {
  return (widx % WorkerRepN) * WorkerN;
}

DEVICE
size_t lane_scale_a_offset_32(uint32_t lane_idx) {
  return (lane_idx / 4) + (lane_idx & 3) * 8;
}

DEVICE
size_t lane_scale_b_offset_32(uint32_t lane_idx) {
  return (lane_idx / 4) + (lane_idx & 3) * 8;
}

DEVICE
uint32_t generic_to_shared(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

DEVICE dim3 get_swizzled_blk_coord(dim3 origin) {
  constexpr int GM = 32;
  int blk_idx = origin.y * gridDim.x + origin.x;
  const int BlkPerGrpRow = GM * gridDim.x;
  int group_m = blk_idx / BlkPerGrpRow;
  int group_n = blk_idx % BlkPerGrpRow / GM;
  int row_in_group = blk_idx % GM;
  dim3 ret(group_n, group_m * GM + row_in_group, origin.z);
  return ret;
}

// get the address after swizzling
class SmemPtrS128B {
public:
  DEVICE 
  SmemPtrS128B(void *smem_ptr = nullptr): smem_ptr_(reinterpret_cast<uint8_t*>(smem_ptr)),smem_int_ptr_(generic_to_shared(smem_ptr)) {}

  // 32 banks corespond to 128 bytes
  DEVICE
  uint32_t operator() (uint32_t index) const {
    index += smem_int_ptr_;
    uint32_t row = index / 128;
    uint32_t col = index % 128;
    uint32_t swizzled_col = col ^ ((row & 0x7) * 16);
    return row * 128 + swizzled_col;
  }

  DEVICE
  void* operator + (size_t offset) const {
    offset += (size_t)smem_ptr_;
    size_t row = offset / 128;
    size_t col = offset % 128;
    size_t swizzled_col = col ^ ((row & 0x7) * 16);
    return reinterpret_cast<void*>( 
      row * 128 + swizzled_col
    );
  }

private:
  uint8_t *smem_ptr_;
  uint32_t smem_int_ptr_;
};


namespace ptx {


DEVICE
void
prefetch_tensormap(CUtensorMap const *tensormap_ptr)
{
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tensormap_ptr);
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
}

template <uint32_t RegCount>
DEVICE
void warpgroup_reg_alloc(){
  asm volatile ("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
DEVICE
void warpgroup_reg_dealloc(){
  asm volatile ("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

DEVICE
void fence_shared_async() {
  asm volatile ("fence.proxy.async.shared::cta;\n");
}

DEVICE
void bar_sync(uint32_t const &tag, uint32_t const &thread_count) {
  asm volatile ("bar.cta.sync %0, %1;\n" :: "r"(tag), "r"(thread_count));
}

DEVICE
void mbarrier_init(
   uint64_t *mbar_ptr,
   int thread_count
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile (
    "mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :
    : "r"(smem_int_ptr),
      "r"(thread_count)
  );
}

DEVICE
void mbarrier_arrive(
  uint64_t *mbar_ptr
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile(
    "{\n"
    ".reg .b64 state; \n"
    "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
    "}\n"
    :
    : "r"(smem_int_ptr)
  );
}

DEVICE
void mbarrier_arrive_expect_tx(
  uint64_t *mbar_ptr,
  uint32_t const &bytes
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile (
    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n"
    :
    : "r"(bytes), "r"(smem_int_ptr)
  );
} 

DEVICE
void mbarrier_arrive_cpasync(
  uint64_t *mbar_ptr
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile (
    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n"
    :: "r"(smem_int_ptr)
  );
}

DEVICE
uint32_t try_wait_barrier(
  uint64_t *mbar_ptr,
  uint32_t const &phase
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  uint32_t done;

  asm volatile (
    "{\n"
    ".reg .pred P1;\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
    "selp.b32 %0, 1, 0, P1;\n"
    "}\n"
    : "=r"(done)
    : "r"(smem_int_ptr),
      "r"(phase)
  );

  return done;
}

DEVICE
void wait_barrier(
  uint64_t *mbar_ptr,
  uint32_t const &phase,
  uint32_t const &done = 0
) {
  if (done) return;
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n"
    ".reg .pred                P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
    "@P1                       bra DONE;\n"
    "bra                   LAB_WAIT;\n"
    "DONE:\n"
    "}\n"
    :
    : "r"(smem_int_ptr),
      "r"(phase),
      "r"(ticks)
  );
}

DEVICE
void cpasync_32b(
  uint32_t *global,
  uint32_t *shared
) {
  uint32_t smem_int_ptr = generic_to_shared(shared);
  asm volatile (
    "cp.async.ca.shared.global [%0], [%1], 4; \n"
    :
    : "r"(smem_int_ptr), "l"(global)
  );
}

DEVICE
void tma_copy_tensor_2d(
  CUtensorMap const *tensormap_ptr,
  void const *smem_ptr,
  uint64_t *mbar_ptr,
  uint32_t const &crd0, 
  uint32_t const &crd1
) {
  uint32_t smem_int_ptr = generic_to_shared(smem_ptr);
  uint32_t mbar_int_ptr = generic_to_shared(mbar_ptr);
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tensormap_ptr);
  asm volatile (
    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%3, %4}], [%2], 256;\n"
    :
    : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(mbar_int_ptr),
      "r"(crd1), "r"(crd0)
    : "memory"
  );
}

DEVICE
void tma_store_2d( 
  CUtensorMap const *tensormap_ptr,
  void const *smem_ptr,
  int32_t const &crd0, 
  int32_t const &crd1
) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tensormap_ptr);
  uint32_t smem_int_ptr = generic_to_shared(smem_ptr);
  asm volatile (
    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n"
    :
    : "l"(gmem_int_desc), "r"(smem_int_ptr),
      "r"(crd1), "r"(crd0)
    : "memory"
  );
}

DEVICE
void tma_reduce_add_2d( 
  CUtensorMap const *tensormap_ptr,
  void const *smem_ptr,
  int32_t const &crd0, 
  int32_t const &crd1
) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tensormap_ptr);
  uint32_t smem_int_ptr = generic_to_shared(smem_ptr);
  asm volatile (
    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%2, %3}], [%1];\n"
    :
    : "l"(gmem_int_desc), "r"(smem_int_ptr),
      "r"(crd1), "r"(crd0)
    : "memory"
  );
}

DEVICE
void tma_store_commit() {
  asm volatile ("cp.async.bulk.commit_group;\n");
}

DEVICE
void tma_wait() {
  asm volatile ("cp.async.bulk.wait_group 0;\n");
}

DEVICE
void ldmatrix_a_b8_32x32(
  SmemPtrS128B const &smem_ptr,
  uint32_t reg[8],
  uint32_t const &row,
  uint32_t const &col,
  uint32_t const &bytes_per_row,
  int const &lane_idx
) {
  /*
  0 | 2
  1 | 3
  4 | 6
  5 | 7
  */
  uint32_t smem_int_ptr = smem_ptr((row + lane_idx % 16) * bytes_per_row + col + 16 * (lane_idx / 16));
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
    : "r"(smem_int_ptr)
    : "memory"
  );
  smem_int_ptr = smem_ptr((16 + row + lane_idx % 16) * bytes_per_row + col + 16 * (lane_idx / 16));
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(reg[4]), "=r"(reg[5]), "=r"(reg[6]), "=r"(reg[7])
    : "r"(smem_int_ptr)
    : "memory"
  );
}


DEVICE
void ldmatrix_b_b8_32x32(
  SmemPtrS128B const &smem_ptr,
  uint32_t reg[8],
  uint32_t const &row,
  uint32_t const &col,
  uint32_t const &bytes_per_row,
  int const &lane_idx
) {
  /*
  0 | 1
  2 | 3
  4 | 5
  6 | 7
  */
  uint32_t smem_int_ptr = smem_ptr((row + (lane_idx / 16) * 8 + lane_idx % 8) * bytes_per_row + col + (lane_idx % 16) / 8 * 16);
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
    : "r"(smem_int_ptr)
    : "memory"
  );
  smem_int_ptr = smem_ptr((row + 16 + (lane_idx / 16) * 8 + lane_idx % 8) * bytes_per_row + col + (lane_idx % 16) / 8 * 16);
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(reg[4]), "=r"(reg[5]), "=r"(reg[6]), "=r"(reg[7])
    : "r"(smem_int_ptr)
    : "memory"
  );
}


DEVICE
void mma_mxfp8_16x8x32_(
  uint32_t const &a0, uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,
  uint32_t const &b0, uint32_t const &b1,
  float          &c0, float          &c1, float          &c2, float          &c3,
  uint32_t const &scale_a,
  uint32_t const &scale_b,
  uint16_t const &thr_id_a,
  uint16_t const &thr_id_b,
  uint16_t const &byte_id
) {
  asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%0, %1, %2, %3},"
    "{%10},"
    "{%11, %12},"
    "{%13},"
    "{%14, %15};\n"
    :  "+f"(c0),  "+f"(c1),  "+f"(c2),  "+f"(c3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "r"(scale_a), "h"(byte_id), "h"(thr_id_a),
        "r"(scale_b), "h"(byte_id), "h"(thr_id_b)
  );
}

DEVICE
void mma_mxfp8_32x32x32(
  uint32_t const regA[8],
  uint32_t const regB[8],
  float regC[32],
  uint32_t const &scale_a,
  uint32_t const &scale_b,
  uint32_t const &byte_id
) {
  #pragma unroll
  for (uint16_t n = 0; n < 4; ++n) {
    #pragma unroll
    for (uint16_t m = 0; m < 2; ++m) {
      mma_mxfp8_16x8x32_(
        regA[0+m*4], regA[1+m*4], regA[2+m*4], regA[3+m*4],
        regB[0+n*2], regB[1+n*2],
        regC[0+m*4+n*8], regC[1+m*4+n*8], regC[2+m*4+n*8], regC[3+m*4+n*8], 
        scale_a, scale_b,
        m, n, byte_id
      );
    }
  }
}

DEVICE
void stmatrix_b16_32x32(
  SmemPtrS128B const &smem_ptr,
  uint32_t reg[16],
  uint32_t const &row,
  uint32_t const &col,
  uint32_t const &bytes_per_row,
  int lane_idx
) {
  /*
  0 | 4 | 8  | 12
  1 | 5 | 9  | 13
  2 | 6 | 10 | 14
  3 | 7 | 11 | 15
  */
  #pragma unroll
  for (size_t n = 0; n < 4; ++n) {
    uint32_t smem_int_ptr = smem_ptr((row + lane_idx) * bytes_per_row + col * 2 + n * 16);
    asm volatile (
      "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3}; \n"
      :
      : "r"(reg[0+n*4]), "r"(reg[1+n*4]), "r"(reg[2+n*4]), "r"(reg[3+n*4]), "r"(smem_int_ptr)
    );
  }
}

DEVICE
void clc_try_cancel(
  int4 *clc_ptr,
  uint64_t *mbar_ptr
) {
  uint32_t clc_int_ptr = generic_to_shared(clc_ptr);
  uint32_t mbar_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile (
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n" 
    :
    : "r"(clc_int_ptr),
      "r"(mbar_int_ptr)
  );
}

DEVICE
std::tuple<dim3, uint32_t> clc_query(
  int4 *clc_ptr
) {
  uint32_t valid = 0;
  dim3 result(0, 0, 0);
  uint32_t clc_int_ptr = generic_to_shared(clc_ptr);
  asm volatile(
    "{\n"
    ".reg .pred p1;\n\t"
    ".reg .b128 clc_result;\n\t"
    "ld.shared.b128 clc_result, [%4];\n\t"
    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"
    "selp.u32 %3, 1, 0, p1;\n\t"
    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_result;\n\t"
    "}\n"
    : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(valid)
    : "r"(clc_int_ptr)
    : "memory"
  );
  ptx::fence_shared_async();
  return std::make_tuple(get_swizzled_blk_coord(result), valid);
}

DEVICE
uint32_t cvt_fp32_to_bf16x2(float const &a, float const &b) {
  uint32_t ret;
  asm volatile (
    "cvt.rn.bf16x2.f32 %0, %1, %2;\n"
    : "=r"(ret)
    : "f"(b), "f"(a)
  );
  return ret;
}

DEVICE
uint32_t cvt_fp32_to_fp16x2(float const &a, float const &b) {
  uint32_t ret;
  asm volatile (
    "cvt.rn.f16x2.f32 %0, %1, %2;\n"
    : "=r"(ret)
    : "f"(b), "f"(a)
  );
  return ret;
}


} // namespace ptx;


template <int Stage>
class Pipeline {
public:
  class State {
  public:
    DEVICE
    State (uint32_t const &init_phase = 0): phase_(init_phase), stage_(0) {}
    DEVICE
    void advance() {
      if (++stage_ == Stage) {
        stage_ = 0;
        phase_ ^= 1;
      }
    }
    DEVICE
    uint32_t const &phase() const { return phase_; }
    DEVICE
    uint32_t const &stage() const { return stage_; }
  private:
    uint32_t phase_;
    uint32_t stage_;
  };

  struct alignas(8) Storage {
    alignas(8) uint64_t full[Stage];
    alignas(8) uint64_t empty[Stage];
  };

  DEVICE
  Pipeline(Storage &storage, uint32_t num_producers, uint32_t num_consumers): storage_(storage) {
    if (threadIdx.x == 0) {
      #pragma unroll
      for (int i = 0; i < Stage; ++i) {
        ptx::mbarrier_init(&storage.full[i], num_producers);
        ptx::mbarrier_init(&storage.empty[i], num_consumers);
      }
    }
    __syncwarp();
  }

  DEVICE
  void producer_arrive_expect_tx(State const &state, uint32_t bytes) {
    ptx::mbarrier_arrive_expect_tx(
      &storage_.full[state.stage()], bytes
    );
  }

  DEVICE
  void producer_arrive_cpasync(State const &state) {
    ptx::mbarrier_arrive_cpasync(&storage_.full[state.stage()]);
  }

  DEVICE
  uint32_t producer_try_wait(State const &state) {
    return ptx::try_wait_barrier(
      &storage_.empty[state.stage()],
      state.phase()
    );
  }

  DEVICE
  void producer_wait(State const &state, uint32_t const &done = 0) {
    ptx::wait_barrier(
      &storage_.empty[state.stage()],
      state.phase(), done
    );
  }
  
  DEVICE
  void producer_commit(State const &state) {
    ptx::mbarrier_arrive(&storage_.full[state.stage()]);
  }

  DEVICE
  uint64_t *producer_get_mbar(State const &state) {
    return &storage_.full[state.stage()];
  }

  DEVICE
  void consumer_arrive(State const &state) {
    ptx::mbarrier_arrive(&storage_.empty[state.stage()]);
  }

  DEVICE
  uint32_t consumer_try_wait(State const &state) {
    return ptx::try_wait_barrier(&storage_.full[state.stage()], state.phase());
  }

  DEVICE
  void consumer_wait(State const &state, uint32_t const &done = 0) {
    ptx::wait_barrier(
      &storage_.full[state.stage()],
      state.phase(), done
    );
  }

private:
  Storage &storage_;
};

struct Params {
  CUtensorMap tensormap_A;
  CUtensorMap tensormap_B;
  CUtensorMap tensormap_D;
  uint32_t *scale_A;
  uint32_t *scale_B;
  void *D;
  int64_t m;
  int64_t n;
  int64_t k;
};

enum class WarpGroup : uint32_t {
  Producer = 0,
  Consumer0 = 1,
  Consumer1 = 2
};

enum class ProducerRole : uint32_t {
  ScaleA = 0,
  TMA = 1,
  ScaleB = 2,
  Scheduler = 3,
  None = 4
};

template <class StageData_, int Stages_, int Alignment = 128>
class DataPipe {
public:
  static constexpr int Stages = Stages_;
  using StageData = StageData_;
  using Pipeline = Pipeline<Stages>;
  struct alignas(Alignment) Storage {
    StageData stages[Stages];
  };
  DEVICE
  DataPipe(Storage &storage, Pipeline &&pipe, typename Pipeline::State &&state): storage_(storage), pipe(std::move(pipe)), state(std::move(state)) {}
  DEVICE
  StageData &data() { return storage_.stages[state.stage()]; }
  Pipeline pipe;
  typename Pipeline::State state;
private:
  Storage &storage_;
};

struct alignas(128) MainloopStageData {
  alignas(128) uint8_t sA[CtaM * CtaK];
  alignas(128) uint8_t sB[CtaM * CtaK];
  alignas(4) uint32_t sSFA[CtaM];
  alignas(4) uint32_t sSFB[CtaN];
};

using MainloopPipe = DataPipe<MainloopStageData, Stages, 16>;

struct alignas(16) CLCloopStageData {
  alignas(16) int4 clc_ret;
};

using CLCloopPipe = DataPipe<CLCloopStageData, 1, 16>;

struct alignas(128) EpilogueStorage {
  alignas(128) uint16_t sC[EpiM * EpiN];
};

struct alignas(16) SchedulerScratch {
  alignas(8) uint64_t mbar;
  alignas(16) int4 clc_ret;
};

struct SmemStorage {
  struct alignas(16) PipelineStorage {
    alignas(8) MainloopPipe::Pipeline::Storage mainloop;
    alignas(8) CLCloopPipe::Pipeline::Storage clcloop;
  };
  struct DataStorage {
    alignas(128) MainloopPipe::Storage mainloop;
    alignas(128) CLCloopPipe::Storage clcloop;
    alignas(128) EpilogueStorage epilogue;
  };
  alignas(16) PipelineStorage pipeline;
  alignas(128) DataStorage data;
  alignas(16) SchedulerScratch scheduler;
};

struct WorkTileInfo {
  uint32_t blk_m;
  uint32_t blk_n;
  uint32_t valid;
};

DEVICE void update_work_info(dim3 coord, bool valid, WorkTileInfo& work_info) {
  work_info.blk_m = coord.y;
  work_info.blk_n = coord.x;
  work_info.valid = valid;
}

template<bool Accum>
__launch_bounds__(128*3, 1)
__global__ void mxfp8_gemm(
  __grid_constant__ Params const params,
  __grid_constant__ const dim3 real_grid_dim
) {
  extern __shared__ uint8_t smem[];

  // role
  int tidx = threadIdx.x;
  int lane_idx = threadIdx.x % NumThreadsPerWarp;
  int widx = threadIdx.x / NumThreadsPerWarp - 4;
  WarpGroup wg = static_cast<WarpGroup>(tidx / NumThreadsPerWarpGroup);
  ProducerRole role = static_cast<ProducerRole>(std::min(tidx / NumThreadsPerWarp, 4));

  // storage
  SmemStorage& storage = *reinterpret_cast<SmemStorage*>(smem);

  // prefetch tensormap
  if (tidx == 0) {
    ptx::prefetch_tensormap(&params.tensormap_A);
    ptx::prefetch_tensormap(&params.tensormap_B);
    ptx::mbarrier_init(&storage.scheduler.mbar, 1);
  }

  // init pipeline
  auto create_mainloop = [&] {
    MainloopPipe::Pipeline mainloop_pipe(storage.pipeline.mainloop, NumThreadsPerWarp * 2 + 1, NumThreadsPerWarp * WorkerRepM * WorkerRepN);
    MainloopPipe::Pipeline::State mainloop_state(wg == WarpGroup::Producer ? 1 : 0);
    return MainloopPipe(storage.data.mainloop, std::move(mainloop_pipe), std::move(mainloop_state));
  };

  auto create_clcloop = [&] {
    CLCloopPipe::Pipeline clc_pipe(storage.pipeline.clcloop, 1, NumThreadsPerWarp * 2 + 1 + NumThreadsPerWarp * WorkerRepM * WorkerRepN);
    CLCloopPipe::Pipeline::State clc_state;
    return CLCloopPipe(storage.data.clcloop, std::move(clc_pipe), std::move(clc_state));
  };

  auto mainloop = create_mainloop();
  auto clcloop = create_clcloop();

  __syncthreads();

  WorkTileInfo work_info;
  update_work_info(get_swizzled_blk_coord(blockIdx), 1, work_info);
  if (work_info.blk_m >= real_grid_dim.y) return;
  int64_t kstages = params.k / 128;

  if (wg == WarpGroup::Producer) {
    ptx::warpgroup_reg_dealloc<80>();

    if (role == ProducerRole::TMA) {
      if (lane_idx == 0) {
        while (work_info.valid) {
          #pragma unroll 1
          for (int k = 0; k < kstages; ++k) {
            mainloop.pipe.producer_wait(mainloop.state);
            mainloop.pipe.producer_arrive_expect_tx(
              mainloop.state,
              CtaM * CtaK * 2 * sizeof(uint8_t) // sA + sB
            );
            ptx::tma_copy_tensor_2d(
              &params.tensormap_A,
              mainloop.data().sA,
              mainloop.pipe.producer_get_mbar(mainloop.state),
              work_info.blk_m * CtaM,
              k * CtaK
            );
            ptx::tma_copy_tensor_2d(
              &params.tensormap_B,
              mainloop.data().sB,
              mainloop.pipe.producer_get_mbar(mainloop.state),
              work_info.blk_n * CtaN,
              k * CtaK
            );
            mainloop.state.advance();
          }
          clcloop.pipe.consumer_arrive(clcloop.state);
          {
            int4 *clc_ret_ptr = &clcloop.data().clc_ret;
            clcloop.pipe.consumer_wait(clcloop.state);
            auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
            update_work_info(ret, valid, work_info);
          }
          clcloop.state.advance();
        }
      }
    } else if (role == ProducerRole::ScaleA) {
      while (work_info.valid) {
        #pragma unroll 1
        for (int k = 0; k < kstages; ++k) {
          mainloop.pipe.producer_wait(mainloop.state);
          #pragma unroll
          for (int i = lane_idx; i < CtaM; i += NumThreadsPerWarp) {
            ptx::cpasync_32b(
              params.scale_A + ((params.k / 128) * work_info.blk_m + k * CtaK / 128) * CtaM + i * CtaK / 128,
              &mainloop.data().sSFA[i]
            );
          }
          mainloop.pipe.producer_arrive_cpasync(mainloop.state);
          mainloop.state.advance();
        }

        clcloop.pipe.consumer_arrive(clcloop.state);
        {
          int4 *clc_ret_ptr = &clcloop.data().clc_ret;
          clcloop.pipe.consumer_wait(clcloop.state);
          auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
          update_work_info(ret, valid, work_info);
        }
        clcloop.state.advance();
      }

    } else if (role == ProducerRole::ScaleB) {
      while (work_info.valid) {
        #pragma unroll 1
        for (int k = 0; k < kstages; ++k) {
          mainloop.pipe.producer_wait(mainloop.state);
          #pragma unroll
          for (int i = lane_idx; i < CtaN; i += NumThreadsPerWarp) {
            ptx::cpasync_32b(
              params.scale_B + ((params.k / 128) * work_info.blk_n + k * CtaK / 128) * CtaN + i * CtaK / 128,
              &mainloop.data().sSFB[i]
            );
          }
          mainloop.pipe.producer_arrive_cpasync(mainloop.state);
          mainloop.state.advance();
        }
        clcloop.pipe.consumer_arrive(clcloop.state);
        {
          int4 *clc_ret_ptr = &clcloop.data().clc_ret;
          clcloop.pipe.consumer_wait(clcloop.state);
          auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
          update_work_info(ret, valid, work_info);
        }
        clcloop.state.advance();

      }
    } else if (role == ProducerRole::Scheduler) {
      if (lane_idx == 0) {
        uint32_t scratch_phase = 0;
        SchedulerScratch* scratch = &storage.scheduler;
        while (work_info.valid) {
          clcloop.pipe.producer_wait(clcloop.state);
          bool found_valid_task = false;
          while (!found_valid_task) {
            ptx::mbarrier_arrive_expect_tx(&scratch->mbar, 16);
            ptx::clc_try_cancel(&scratch->clc_ret, &scratch->mbar);
            ptx::wait_barrier(&scratch->mbar, scratch_phase);
            scratch_phase ^= 1;
            auto [ret, valid] = ptx::clc_query(&scratch->clc_ret);
            if (!valid) {
              found_valid_task = true;
              work_info.valid = false;
            }
            else if (ret.y < real_grid_dim.y) {
              found_valid_task = true;
              work_info.valid = true;
            }
          }
          clcloop.data().clc_ret = scratch->clc_ret;
          clcloop.pipe.producer_commit(clcloop.state);
          clcloop.state.advance();
        }
      }
    }
  } else {
    ptx::warpgroup_reg_alloc<208>();
    size_t m_offset = worker_m_offset(widx);
    size_t n_offset = worker_n_offset(widx);

    uint32_t sfa[AtomRepM], sfb[AtomRepN];
    uint32_t regA[AtomRepK][AtomRepM][AtomRegA];
    uint32_t regB[AtomRepK][AtomRepN][AtomRegB];
    float regC[AtomRepM][AtomRepN][AtomRegC];

    auto ldmatrix = [&] __attribute__((always_inline))(uint32_t const &k_pipe) {
      #pragma unroll
      for (int i = 0; i < AtomRepM; ++i)
        ptx::ldmatrix_a_b8_32x32(
          SmemPtrS128B(mainloop.data().sA),
          regA[k_pipe][i],
          m_offset + i * AtomM,
          k_pipe * AtomK,
          CtaK,
          lane_idx
        );
      #pragma unroll
      for (int j = 0; j < AtomRepN; ++j)
        ptx::ldmatrix_b_b8_32x32(
          SmemPtrS128B(mainloop.data().sB),
          regB[k_pipe][j],
          n_offset + j * AtomN,
          k_pipe * AtomK,
          CtaK,
          lane_idx
        );
    };

    auto mma = [&] __attribute__((always_inline))(uint32_t const &k_pipe) {
      #pragma unroll
      for (int i = 0; i < AtomRepM; ++i) {
        #pragma unroll
        for (int j = 0; j < AtomRepN; ++j) {
          ptx::mma_mxfp8_32x32x32(
            regA[k_pipe][i],
            regB[k_pipe][j],
            regC[i][j],
            sfa[i],
            sfb[j],
            k_pipe
          );
        }
      }
    };

    while (work_info.valid) {
      #pragma unroll
      for (int i = 0; i < AtomRepM; ++i)
        #pragma unroll
        for (int j = 0; j < AtomRepN; ++j)
          #pragma unroll
          for (int k = 0; k < AtomRegC; ++k)
            regC[i][j][k] = 0.0f;

      #pragma unroll 1
      for (int k_block = 0; k_block < kstages; ++k_block) {
        auto token = mainloop.pipe.consumer_try_wait(mainloop.state);
        mainloop.pipe.consumer_wait(mainloop.state, token);


        // load scale
        #pragma unroll
        for (int i = 0; i < AtomRepM; ++i)
          sfa[i] = mainloop.data().sSFA[m_offset + AtomM * i + lane_scale_a_offset_32(lane_idx)];
        #pragma unroll
        for (int j = 0; j < AtomRepN; ++j)
          sfb[j] = mainloop.data().sSFB[n_offset + AtomN * j + lane_scale_b_offset_32(lane_idx)];

        // load data
        ldmatrix(0);
        #pragma unroll
        for (uint32_t k_pipe = 0; k_pipe < AtomRepK; ++k_pipe) {
          if (k_pipe < AtomRepK - 1) {
            ldmatrix(k_pipe + 1);
            if (k_pipe == AtomRepK - 2) {
              // last atom has been loaded into registers
              mainloop.pipe.consumer_arrive(mainloop.state);
            }
          }
          mma(k_pipe);
        }

        mainloop.state.advance();
      }
      clcloop.pipe.consumer_arrive(clcloop.state);
      
      // epilogue
      {
        auto sC = SmemPtrS128B(storage.data.epilogue.sC);
        uint32_t regD[AtomRepM][AtomRepN][AtomRegC/2];
        #pragma unroll
        for (int i = 0; i < AtomRepM; ++i) {
          #pragma unroll
          for (int j = 0; j < AtomRepN; ++j) {
            #pragma unroll
            for (int k = 0; k < AtomRegC/2; ++k) {
              regD[i][j][k] = ptx::cvt_fp32_to_bf16x2(regC[i][j][k * 2], regC[i][j][k * 2 + 1]);
              // regD[i][j][k] = ptx::cvt_fp32_to_fp16x2(regC[i][j][k * 2], regC[i][j][k * 2 + 1]);
            }
          }
        }

        for (int epi_m = 0; epi_m < CtaM / EpiM; ++epi_m) {
          for (int epi_n = 0; epi_n < CtaN / EpiN; ++epi_n) {
            int32_t epi_st_m_idx = epi_m * EpiM;
            int32_t epi_ed_m_idx = epi_st_m_idx + EpiM;
            int32_t epi_st_n_idx = epi_n * EpiN;
            int32_t epi_ed_n_idx = epi_st_n_idx + EpiN;
            if (m_offset >= epi_st_m_idx && m_offset < epi_ed_m_idx &&
                n_offset >= epi_st_n_idx && n_offset < epi_ed_n_idx) {
              int32_t epi_m_offset = m_offset - epi_st_m_idx;
              int32_t epi_n_offset = n_offset - epi_st_n_idx;
              #pragma unroll
              for (int i = 0; i < AtomRepM; ++i) {
                #pragma unroll
                for (int j = 0; j < AtomRepN; ++j) {
                  ptx::stmatrix_b16_32x32(
                    sC,
                    regD[i][j],
                    i * AtomM + epi_m_offset,
                    j * AtomN + epi_n_offset,
                    EpiN * sizeof(uint16_t),
                    lane_idx
                  );
                }
              }
              ptx::fence_shared_async();
            }
            ptx::bar_sync(0, 256);
            if (widx == 0 && lane_idx == 0) {
              if constexpr (Accum) {
                ptx::tma_reduce_add_2d(
                  &params.tensormap_D,
                  storage.data.epilogue.sC,
                  work_info.blk_m * CtaM + epi_st_m_idx,
                  work_info.blk_n * CtaN + epi_st_n_idx
                );
              }
              else {
                ptx::tma_store_2d(
                  &params.tensormap_D,
                  storage.data.epilogue.sC,
                  work_info.blk_m * CtaM + epi_st_m_idx,
                  work_info.blk_n * CtaN + epi_st_n_idx
                );
              }
              ptx::tma_store_commit();
              ptx::tma_wait();
            }
            __syncwarp();
            ptx::bar_sync(0, 256);
          }
        }
      }

      // clc
      {
        int4 *clc_ret_ptr = &clcloop.data().clc_ret;
        clcloop.pipe.consumer_wait(clcloop.state);
        auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
        update_work_info(ret, valid, work_info);
      }
      clcloop.state.advance();
    }
  }
}

__global__ void bf16_to_fp32(
  uint16_t *in, float *out,
  size_t size
) {
  uint64_t thr_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thr_id < size) *(out + thr_id) = __bfloat162float(*((__nv_bfloat16*)in+thr_id));
}

__global__ void fp16_to_fp32(
  uint16_t *in, float *out,
  size_t size
) {
  uint64_t thr_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thr_id < size) *(out + thr_id) = __half2float(*((half*)in+thr_id));
}

template<bool Accum>
void mxfp8_gemm_launch(
  uint8_t* dev_qA, uint8_t* dev_qB,
  uint8_t* dev_SFA, uint8_t* dev_SFB,
  uint16_t* dev_C_bf16,
  const int m, const int n, const int k,
  cudaStream_t stream
) {
  Params params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.scale_A = (uint32_t*)dev_SFA;
  params.scale_B = (uint32_t*)dev_SFB;
  params.tensormap_A = make_tensormap_fp8(dev_qA, m, k, CtaM, CtaK);
  params.tensormap_B = make_tensormap_fp8(dev_qB, n, k, CtaN, CtaK);
  params.tensormap_D = make_tensormap_bf16(dev_C_bf16, m, n, EpiM, EpiN);
  params.D = dev_C_bf16;
  
  cudaFuncSetAttribute(
    mxfp8_gemm<Accum>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SmemStorage)
  );

  constexpr int GM = 32;
  dim3 real_grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
  dim3 grid_dim(real_grid_dim.x, cdiv(real_grid_dim.y, GM) * GM);
  mxfp8_gemm<Accum><<<grid_dim, dim3(128*3), sizeof(SmemStorage), stream>>>(params, real_grid_dim);
}
}

template void fp8::mxfp8_gemm_launch<true>(
  uint8_t* dev_qA, uint8_t* dev_qB,
  uint8_t* dev_SFA, uint8_t* dev_SFB,
  uint16_t* dev_C_bf16,
  const int m, const int n, const int k,
  cudaStream_t stream
);

template void fp8::mxfp8_gemm_launch<false>(
  uint8_t* dev_qA, uint8_t* dev_qB,
  uint8_t* dev_SFA, uint8_t* dev_SFB,
  uint16_t* dev_C_bf16,
  const int m, const int n, const int k,
  cudaStream_t stream
);