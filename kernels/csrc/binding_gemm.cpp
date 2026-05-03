#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/core/ScalarType.h>

namespace fp4 {
  void nvfp4_gemm_launch(
    uint8_t* dev_qA, uint8_t* dev_qB, 
    uint8_t* dev_SFA, uint8_t* dev_SFB,
    float* dev_OuterSFA, float* dev_OuterSFB,
    uint16_t* dev_C_bf16,
    const int m, const int n, const int k,
    cudaStream_t stream
  );
}

namespace fp8 {
  template<bool Accum>
  void mxfp8_gemm_launch(
    uint8_t* dev_qA, uint8_t* dev_qB,
    uint8_t* dev_SFA, uint8_t* dev_SFB,
    uint16_t* dev_C_bf16,
    const int m, const int n, const int k,
    cudaStream_t stream
  );
}

static inline void check_cuda_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be a defined Tensor");
  TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static inline void check_dtype(const torch::Tensor& t, c10::ScalarType dt, const char* name) {
  TORCH_CHECK(t.scalar_type() == dt, name, " dtype mismatch. Expected ",
              c10::toString(dt), ", got ", c10::toString(t.scalar_type()), ".");
}

static inline std::tuple<int, int, int> check_shape(const torch::Tensor& a, const torch::Tensor& b) {
  int m = a.size(-2);
  int n = b.size(-2);
  int k = a.size(-1);
  TORCH_CHECK(b.size(-1) == k, "Matrix shapes mismatch. A is ", m, "x", k, ", B is ", b.size(-1), "x", n);
  return std::make_tuple(m, n, k);
}


torch::Tensor fp4_gemm(
  torch::Tensor a, torch::Tensor b,
  torch::Tensor sfa, torch::Tensor sfb,
  torch::Tensor out_sfa, torch::Tensor out_sfb
) {
    check_cuda_contig(a, "a");
    check_cuda_contig(b, "b");
    check_cuda_contig(sfa, "sfa");
    check_cuda_contig(sfb, "sfb");
    check_cuda_contig(out_sfa, "out_sfa");
    check_cuda_contig(out_sfb, "out_sfb");

    check_dtype(a,       torch::kUInt8,   "a");
    check_dtype(b,       torch::kUInt8,   "b");
    check_dtype(sfa, c10::ScalarType::Float8_e4m3fn, "sfa");
    check_dtype(sfb, c10::ScalarType::Float8_e4m3fn, "sfb");
    check_dtype(out_sfa, torch::kFloat32, "out_sfa");
    check_dtype(out_sfb, torch::kFloat32, "out_sfb");

    auto [m, n, k] = check_shape(a, b);
    k *= 2;

    TORCH_CHECK(a.get_device() == b.get_device(), "a and b must be on same GPU");
    TORCH_CHECK(a.get_device() == sfa.get_device(), "a and sfa must be on same GPU");
    TORCH_CHECK(a.get_device() == sfb.get_device(), "a and sfb must be on same GPU");
    TORCH_CHECK(a.get_device() == out_sfa.get_device(), "a and out_sfa must be on same GPU");
    TORCH_CHECK(a.get_device() == out_sfb.get_device(), "a and out_sfb must be on same GPU");

    auto c = torch::empty({m, n},
                        torch::TensorOptions()
                            .device(a.device())
                            .dtype(torch::kBFloat16));


    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fp4::nvfp4_gemm_launch(
      (uint8_t*)a.data_ptr<uint8_t>(),
      (uint8_t*)b.data_ptr<uint8_t>(),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()), 
      reinterpret_cast<uint8_t*>(sfb.data_ptr()), 
      out_sfa.data_ptr<float>(),
      out_sfb.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(c.data_ptr<at::BFloat16>()),
      m, n, k,
      stream
    );

    return c;
}

torch::Tensor fp8_gemm(
  torch::Tensor a, torch::Tensor b,
  torch::Tensor sfa, torch::Tensor sfb
) {
  check_cuda_contig(a, "a");
  check_cuda_contig(b, "b");
  check_cuda_contig(sfa, "sfa");
  check_cuda_contig(sfb, "sfb");

  check_dtype(a, torch::kFloat8_e4m3fn, "a");
  check_dtype(b, torch::kFloat8_e4m3fn, "b");
  check_dtype(sfa, torch::kFloat8_e8m0fnu, "sfa");
  check_dtype(sfb, torch::kFloat8_e8m0fnu, "sfb");

  TORCH_CHECK(a.get_device() == b.get_device(), "a and b must be on same GPU");
  TORCH_CHECK(a.get_device() == sfa.get_device(), "a and sfa must be on same GPU");
  TORCH_CHECK(a.get_device() == sfb.get_device(), "a and sfb must be on same GPU");

  auto [m, n, k] = check_shape(a, b);

  auto c = torch::empty({m, n}, a.options().dtype(torch::kBFloat16));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  fp8::mxfp8_gemm_launch<false>(
    (uint8_t*)a.data_ptr(),
    (uint8_t*)b.data_ptr(),
    (uint8_t*)sfa.data_ptr(),
    (uint8_t*)sfb.data_ptr(),
    (uint16_t*)c.data_ptr(),
    m, n, k,
    stream
  );

  return c;
}

torch::Tensor fp8_gemm_accum(
  torch::Tensor a, torch::Tensor b,
  torch::Tensor sfa, torch::Tensor sfb,
  torch::Tensor c
) {
  check_cuda_contig(a, "a");
  check_cuda_contig(b, "b");
  check_cuda_contig(sfa, "sfa");
  check_cuda_contig(sfb, "sfb");
  check_cuda_contig(c, "c");

  check_dtype(a, torch::kFloat8_e4m3fn, "a");
  check_dtype(b, torch::kFloat8_e4m3fn, "b");
  check_dtype(sfa, torch::kFloat8_e8m0fnu, "sfa");
  check_dtype(sfb, torch::kFloat8_e8m0fnu, "sfb");
  check_dtype(c, torch::kBFloat16, "c");

  TORCH_CHECK(a.get_device() == b.get_device(), "a and b must be on same GPU");
  TORCH_CHECK(a.get_device() == sfa.get_device(), "a and sfa must be on same GPU");
  TORCH_CHECK(a.get_device() == sfb.get_device(), "a and sfb must be on same GPU");

  auto [m, n, k] = check_shape(a, b);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  fp8::mxfp8_gemm_launch<true>(
    (uint8_t*)a.data_ptr(),
    (uint8_t*)b.data_ptr(),
    (uint8_t*)sfa.data_ptr(),
    (uint8_t*)sfb.data_ptr(),
    (uint16_t*)c.data_ptr(),
    m, n, k,
    stream
  );

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp4_gemm", &fp4_gemm, "NVFP4 GEMM (templated launcher, BF16 output)");
    m.def("fp8_gemm_accum", &fp8_gemm_accum, "MXFP8 GEMM");
    m.def("fp8_gemm", &fp8_gemm);
}