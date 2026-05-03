#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// fp4
void nvfp4_double_quant_scale_only_launch(
    const float* g_x, float* x_q, int n_elements, cudaStream_t stream
);

template<bool Stochastic>
void nvfp4_double_quant_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int n_elements, uint32_t seed, cudaStream_t stream
);

void nvfp4_double_quant_tile_major_dequant_trans_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    float* g_x_dq,
    int m, int n, cudaStream_t stream
);

void nvfp4_dequant_launch(
    const uint8_t* x_q,
    float* x_dq, float* outer_scale, uint8_t* inner_scale,
    int n_elements, cudaStream_t stream
);

void nvfp4_dequant_tile_major_launch(
    const uint8_t* g_x,
    float* x_dq, float* outer_scale, uint8_t* inner_scale,
    int m, int n, cudaStream_t stream
);

void dequant_fp4_rh_requant_tile_major_fused_launch(
    uint8_t* __restrict__ g_x,
    float* __restrict__ g_outer_scale,
    uint8_t* __restrict__ g_inner_scale,
    uint8_t* __restrict__ s_x_q,
    float* __restrict__ s_outer_scale,
    uint8_t* __restrict__ s_inner_scale,
    int m, int n,
    uint32_t seed, cudaStream_t stream
);

void rh_quant_fp4_trans_rh_requant_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    uint8_t* x_t_q, float* t_outer_scale, uint8_t* t_inner_scale,
    int m, int n, uint32_t seed_dx, uint32_t seed_dw, cudaStream_t stream
);

void unpack_fp4_launch(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int n_elements, cudaStream_t stream
);

template<bool Stochastic>
void nvfp4_double_quant_tile_major_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
);

template<bool Stochastic>
void rh_nvfp4_double_quant_tile_major_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
);

void nvfp4_double_quant_tile_major_dequant_trans_rh_requant_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    uint8_t* x_t_q, float* t_outer_scale, uint8_t* t_inner_scale,
    int m, int n, uint32_t seed, cudaStream_t stream
);

void rh_quant_fp4_trans_rh_requant_fp4_fp8_fused_launch(
    const float* g_x,
    uint8_t* x_q, float* outer_scale, uint8_t* inner_scale,
    uint8_t* x_t_fp4_q, float* t_fp4_outer_scale, uint8_t* t_fp4_inner_scale,
    uint8_t* x_t_fp8_q, uint8_t* t_fp8_scale,
    int m, int n, uint32_t seed_dx, uint32_t seed_dw, cudaStream_t stream
);



// fp8
void quant_fp8_launch(
    const float* g_x, uint8_t* x_q, uint8_t* scale,
    int m, int n, cudaStream_t stream
);

void quant_fp8_tile_major_launch(
    const float* g_x, uint8_t* x_q, uint8_t* scale,
    int m, int n, cudaStream_t stream
);

void quant_fp8_dequant_trans_requant_fused_launch(
    const float* g_x, uint8_t* x_q, uint8_t* scale,
    uint8_t* x_t_q, uint8_t* scale_t,
    int m, int n, cudaStream_t stream
);

void dequant_fp8_launch(
  const uint8_t* x_q,
  const uint8_t* g_scale,
  float* x_dq,
  int m, int n, cudaStream_t stream
);

void dequant_fp8_tile_major_launch(
  const uint8_t* x_q,
  const uint8_t* g_scale,
  float* x_dq,
  int m, int n, cudaStream_t stream
);



static inline void check_cuda_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be a defined Tensor");
  TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static inline void check_dtype(const torch::Tensor& t, c10::ScalarType dt, const char* name) {
  TORCH_CHECK(t.scalar_type() == dt, name, " dtype mismatch. Expected ",
              c10::toString(dt), ", got ", c10::toString(t.scalar_type()));
}

torch::Tensor quant_fp4_scale_only(torch::Tensor x) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");

    auto y = torch::empty_like(x);
    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    int n_elements = x.numel();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvfp4_double_quant_scale_only_launch(x_ptr, y_ptr, n_elements, stream);

    return y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
            torch::Tensor, torch::Tensor, torch::Tensor,
            torch::Tensor, torch::Tensor>
rh_quant_fp4_trans_rh_requant_fp4_fp8_fused(torch::Tensor x, uint32_t seed_dx, uint32_t seed_dw) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");
    TORCH_CHECK(x.size(-1) % 128 == 0, "last dimension of x must be able to be devided by 128", x.size(-1));
    TORCH_CHECK(x.sizes().size() == 2, "x must be a 2D matrix");

    int m = x.size(-2);
    int n = x.size(-1);

    auto q_x = torch::empty({m, n / 2}, x.options().dtype(torch::kUInt8));
    auto inner_scale = torch::empty({m, n / 16}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto outer_scale = torch::empty({m, n / 128}, x.options());

    auto q_x_t = torch::empty({n, m / 2}, x.options().dtype(torch::kUInt8));
    auto t_fp4_inner_scale = torch::empty({n, m / 16}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto t_fp4_outer_scale = torch::empty({n, m / 128}, x.options());

    auto q_x_t_fp8 = torch::empty({n, m}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto t_fp8_scale = torch::empty({n, m / 32}, x.options().dtype(torch::kFloat8_e8m0fnu));

    const float* x_ptr = x.data_ptr<float>();
    uint8_t* q_x_ptr = q_x.data_ptr<uint8_t>();
    uint8_t* inner_scale_ptr = (uint8_t*)(inner_scale.data_ptr());
    float* outer_scale_ptr = outer_scale.data_ptr<float>();

    uint8_t* q_x_t_ptr = q_x_t.data_ptr<uint8_t>();
    uint8_t* t_fp4_inner_scale_ptr = (uint8_t*)(t_fp4_inner_scale.data_ptr());
    float* t_fp4_outer_scale_ptr = t_fp4_outer_scale.data_ptr<float>();

    uint8_t* q_x_t_fp8_ptr = (uint8_t*)q_x_t_fp8.data_ptr();
    uint8_t* t_fp8_scale_ptr = (uint8_t*)t_fp8_scale.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rh_quant_fp4_trans_rh_requant_fp4_fp8_fused_launch(
        x_ptr, q_x_ptr, outer_scale_ptr, inner_scale_ptr,
        q_x_t_ptr, t_fp4_outer_scale_ptr, t_fp4_inner_scale_ptr,
        q_x_t_fp8_ptr, t_fp8_scale_ptr,
        m, n, seed_dx, seed_dw, stream
    );

    return std::make_tuple(
        q_x, outer_scale, inner_scale,
        q_x_t, t_fp4_outer_scale, t_fp4_inner_scale,
        q_x_t_fp8, t_fp8_scale
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
            torch::Tensor, torch::Tensor, torch::Tensor>
rh_quant_fp4_trans_rh_requant_fused(torch::Tensor x, uint32_t seed_dx, uint32_t seed_dw) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");
    TORCH_CHECK(x.size(-1) % 128 == 0, "last dimension of x must be able to be devided by 128, got ", x.size(-1));
    TORCH_CHECK(x.sizes().size() == 2, "x must be a 2D matrix");

    int m = x.size(-2);
    int n = x.size(-1);

    auto q_x = torch::empty({m, n / 2}, x.options().dtype(torch::kUInt8));
    auto inner_scale = torch::empty({m, n / 16}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto outer_scale = torch::empty({m, n / 128}, x.options());

    auto q_x_t = torch::empty({n, m / 2}, x.options().dtype(torch::kUInt8));
    auto t_inner_scale = torch::empty({n, m / 16}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto t_outer_scale = torch::empty({n, m / 128}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    uint8_t* q_x_ptr = q_x.data_ptr<uint8_t>();
    uint8_t* inner_scale_ptr = (uint8_t*)(inner_scale.data_ptr());
    float* outer_scale_ptr = outer_scale.data_ptr<float>();

    uint8_t* q_x_t_ptr = q_x_t.data_ptr<uint8_t>();
    uint8_t* t_inner_scale_ptr = (uint8_t*)(t_inner_scale.data_ptr());
    float* t_outer_scale_ptr = t_outer_scale.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rh_quant_fp4_trans_rh_requant_fused_launch(
        x_ptr, q_x_ptr, outer_scale_ptr, inner_scale_ptr,
        q_x_t_ptr, t_outer_scale_ptr, t_inner_scale_ptr,
        m, n, seed_dx, seed_dw, stream
    );

    return std::make_tuple(q_x, outer_scale, inner_scale, q_x_t, t_outer_scale, t_inner_scale);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
            torch::Tensor, torch::Tensor, torch::Tensor>
quant_fp4_dequant_trans_rh_requant_fused(torch::Tensor x, uint32_t seed) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");
    TORCH_CHECK(x.size(-1) % 128 == 0, "last dimension of x must be able to be devided by 128, got ", x.size(-1));
    TORCH_CHECK(x.sizes().size() == 2, "x must be a 2D matrix");

    int m = x.size(-2);
    int n = x.size(-1);

    auto q_x = torch::empty({m, n / 2}, x.options().dtype(torch::kUInt8));
    auto inner_scale = torch::empty({m, n / 16}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto outer_scale = torch::empty({m, n / 128}, x.options());

    auto q_x_t = torch::empty({n, m / 2}, x.options().dtype(torch::kUInt8));
    auto t_inner_scale = torch::empty({n, m / 16}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto t_outer_scale = torch::empty({n, m / 128}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    uint8_t* q_x_ptr = q_x.data_ptr<uint8_t>();
    uint8_t* inner_scale_ptr = (uint8_t*)(inner_scale.data_ptr());
    float* outer_scale_ptr = outer_scale.data_ptr<float>();

    uint8_t* q_x_t_ptr = q_x_t.data_ptr<uint8_t>();
    uint8_t* t_inner_scale_ptr = (uint8_t*)(t_inner_scale.data_ptr());
    float* t_outer_scale_ptr = t_outer_scale.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvfp4_double_quant_tile_major_dequant_trans_rh_requant_fused_launch
        (x_ptr, q_x_ptr, outer_scale_ptr, inner_scale_ptr,
            q_x_t_ptr, t_outer_scale_ptr, t_inner_scale_ptr, m, n, seed, stream);

    return std::make_tuple(q_x, outer_scale, inner_scale, q_x_t, t_outer_scale, t_inner_scale);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
quant_fp4_dequant_trans_fused(torch::Tensor x) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");
    TORCH_CHECK(x.size(-1) % 128 == 0, "last dimension of x must be able to be devided by 128, got", x.size(-1));
    TORCH_CHECK(x.sizes().size() == 2, "x must be a 2D matrix");

    auto q_x_shape = x.sizes().vec();
    q_x_shape.back() /= 2;
    auto inner_scale_shape = x.sizes().vec();
    inner_scale_shape.back() /= 16;
    auto outer_scale_shape = x.sizes().vec();
    outer_scale_shape.back() /= 128;

    auto q_x = torch::empty(q_x_shape, x.options().dtype(torch::kUInt8));
    auto inner_scale = torch::empty(inner_scale_shape, x.options().dtype(torch::kFloat8_e4m3fn));
    auto outer_scale = torch::empty(outer_scale_shape, x.options());
    const float* x_ptr = x.data_ptr<float>();
    uint8_t* q_x_ptr = q_x.data_ptr<uint8_t>();
    uint8_t* inner_scale_ptr = (uint8_t*)(inner_scale.data_ptr());
    float* outer_scale_ptr = outer_scale.data_ptr<float>();
    // int n_elements = x.numel();
    int m = x.size(-2);
    int n = x.size(-1);
    auto x_dq_trans = torch::empty({x.size(1), x.size(0)}, x.options());
    float* x_dq_trans_ptr = x_dq_trans.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvfp4_double_quant_tile_major_dequant_trans_fused_launch
        (x_ptr, q_x_ptr, outer_scale_ptr, inner_scale_ptr, x_dq_trans_ptr, m, n, stream);

    return std::make_tuple(q_x, outer_scale, inner_scale, x_dq_trans);
}

template<bool Stochastic>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quant_fp4(torch::Tensor x, uint32_t seed = 0) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");
    TORCH_CHECK(x.size(-1) % 128 == 0, "last dimension of x must be able to be devided by 128, got", x.size(-1));
    TORCH_CHECK(x.sizes().size() == 2, "x must be a 2D matrix");

    auto q_x_shape = x.sizes().vec();
    q_x_shape.back() /= 2;
    auto inner_scale_shape = x.sizes().vec();
    inner_scale_shape.back() /= 16;
    auto outer_scale_shape = x.sizes().vec();
    outer_scale_shape.back() /= 128;

    auto q_x = torch::empty(q_x_shape, x.options().dtype(torch::kUInt8));
    auto inner_scale = torch::empty(inner_scale_shape, x.options().dtype(torch::kFloat8_e4m3fn));
    auto outer_scale = torch::empty(outer_scale_shape, x.options());
    const float* x_ptr = x.data_ptr<float>();
    uint8_t* q_x_ptr = q_x.data_ptr<uint8_t>();
    uint8_t* inner_scale_ptr = (uint8_t*)(inner_scale.data_ptr());
    float* outer_scale_ptr = outer_scale.data_ptr<float>();
    // int n_elements = x.numel();
    int m = x.size(-2);
    int n = x.size(-1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // nvfp4_double_quant_launch<Stochastic>
    //     (x_ptr, q_x_ptr, outer_scale_ptr, inner_scale_ptr, n_elements, seed, stream);
    nvfp4_double_quant_tile_major_launch<Stochastic>
        (x_ptr, q_x_ptr, outer_scale_ptr, inner_scale_ptr, m, n, seed, stream);

    return std::make_tuple(q_x, outer_scale, inner_scale);
}

template<bool Stochastic>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rh_quant_fp4_fused(torch::Tensor x, uint32_t seed = 0) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");
    TORCH_CHECK(x.size(-1) % 128 == 0, "last dimension of x must be able to be devided by 128, got", x.size(-1));
    TORCH_CHECK(x.sizes().size() == 2, "x must be a 2D matrix");

    auto q_x_shape = x.sizes().vec();
    q_x_shape.back() /= 2;
    auto inner_scale_shape = x.sizes().vec();
    inner_scale_shape.back() /= 16;
    auto outer_scale_shape = x.sizes().vec();
    outer_scale_shape.back() /= 128;

    auto q_x = torch::empty(q_x_shape, x.options().dtype(torch::kUInt8));
    auto inner_scale = torch::empty(inner_scale_shape, x.options().dtype(torch::kFloat8_e4m3fn));
    auto outer_scale = torch::empty(outer_scale_shape, x.options());
    const float* x_ptr = x.data_ptr<float>();
    uint8_t* q_x_ptr = q_x.data_ptr<uint8_t>();
    uint8_t* inner_scale_ptr = (uint8_t*)(inner_scale.data_ptr());
    float* outer_scale_ptr = outer_scale.data_ptr<float>();
    // int n_elements = x.numel();
    int m = x.size(-2);
    int n = x.size(-1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rh_nvfp4_double_quant_tile_major_fused_launch<Stochastic>
        (x_ptr, q_x_ptr, outer_scale_ptr, inner_scale_ptr, m, n, seed, stream);

    return std::make_tuple(q_x, outer_scale, inner_scale);
}

torch::Tensor dequant_fp4(torch::Tensor x_q, torch::Tensor outer_scale, torch::Tensor inner_scale) {
    check_cuda_contig(x_q, "x_q");
    check_cuda_contig(outer_scale, "outer_scale");
    check_dtype(outer_scale, torch::kFloat32, "outer_scale");
    check_cuda_contig(inner_scale, "inner_scale");
    check_dtype(inner_scale, torch::kFloat8_e4m3fn, "inner_scale");

    std::vector<int64_t> x_shape = x_q.sizes().vec();
    x_shape.back() *= 2;

    auto x_dq = torch::empty(x_shape, x_q.options().dtype(torch::kFloat32));
    const uint8_t* x_ptr = x_q.data_ptr<uint8_t>();
    float* x_dq_ptr = x_dq.data_ptr<float>();
    float* outer_scale_ptr = outer_scale.data_ptr<float>();
    uint8_t* inner_scale_ptr = (uint8_t*)(inner_scale.data_ptr());
    // int n_elements = x_dq.numel();
    int m = x_q.size(-2);
    int n = x_q.size(-1) * 2;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // nvfp4_dequant_launch(x_ptr, x_dq_ptr, outer_scale_ptr, inner_scale_ptr, n_elements, stream);
    nvfp4_dequant_tile_major_launch(x_ptr, x_dq_ptr, outer_scale_ptr, inner_scale_ptr, m, n, stream);

    return x_dq;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dequant_fp4_rh_requant_fused(
    torch::Tensor x_q, torch::Tensor outer_scale, torch::Tensor inner_scale, uint32_t seed
) {
    check_cuda_contig(x_q, "x_q");
    check_cuda_contig(outer_scale, "outer_scale");
    check_dtype(outer_scale, torch::kFloat32, "outer_scale");
    check_cuda_contig(inner_scale, "inner_scale");
    check_dtype(inner_scale, torch::kFloat8_e4m3fn, "inner_scale");

    auto s_x_q = torch::empty_like(x_q, x_q.options());
    auto s_os = torch::empty_like(outer_scale, outer_scale.options());
    auto s_is = torch::empty_like(inner_scale, inner_scale.options());

    uint8_t* x_q_ptr = x_q.data_ptr<uint8_t>();
    float* outer_scale_ptr = outer_scale.data_ptr<float>();
    uint8_t* inner_scale_ptr = (uint8_t*)(inner_scale.data_ptr());

    uint8_t* s_x_q_ptr = s_x_q.data_ptr<uint8_t>();
    float* s_os_ptr = s_os.data_ptr<float>();
    uint8_t* s_is_ptr = (uint8_t*)s_is.data_ptr();
    
    int m = x_q.size(-2);
    int n = x_q.size(-1) * 2;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dequant_fp4_rh_requant_tile_major_fused_launch(
        x_q_ptr, outer_scale_ptr, inner_scale_ptr, s_x_q_ptr, s_os_ptr, s_is_ptr, m, n, seed, stream
    );

    return std::make_tuple(s_x_q, s_os, s_is);
}

torch::Tensor unpack_fp4(torch::Tensor x) {
    check_cuda_contig(x, "x");
    int n_elements = x.numel();
    std::vector<int64_t> output_shape = x.sizes().vec();
    output_shape.back() *= 2;
    auto output = torch::empty(output_shape, x.options().dtype(torch::kFloat32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    unpack_fp4_launch((uint8_t*)(x.data_ptr()), output.data_ptr<float>(), n_elements, stream);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, 
    torch::Tensor, torch::Tensor>
quant_fp8_dequant_trans_requant_fused(torch::Tensor x) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");
    TORCH_CHECK(x.size(-1) % 128 == 0, "last dimension of x must be able to be devided by 128, got", x.size(-1));
    TORCH_CHECK(x.sizes().size() == 2, "x must be a 2D matrix");
    int m = x.size(-2);
    int n = x.size(-1);

    auto q_x = torch::empty({m, n}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto scale = torch::empty({m, n / 32}, x.options().dtype(torch::kFloat8_e8m0fnu));
    auto q_x_t = torch::empty({n, m}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto scale_t = torch::empty({n, m / 32}, x.options().dtype(torch::kFloat8_e8m0fnu));

    const float* x_ptr = x.data_ptr<float>();
    uint8_t* q_x_ptr = (uint8_t*)q_x.data_ptr();
    uint8_t* scale_ptr = (uint8_t*)(scale.data_ptr());
    uint8_t* q_x_t_ptr = (uint8_t*)q_x_t.data_ptr();
    uint8_t* scale_t_ptr = (uint8_t*)(scale_t.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    quant_fp8_dequant_trans_requant_fused_launch
        (x_ptr, q_x_ptr, scale_ptr, q_x_t_ptr, scale_t_ptr, m, n, stream);  

    return std::make_tuple(q_x, scale, q_x_t, scale_t);
}

std::tuple<torch::Tensor, torch::Tensor> quant_fp8(torch::Tensor x) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat32, "x");
    TORCH_CHECK(x.size(-1) % 128 == 0, "last dimension of x must be able to be devided by 128, got", x.size(-1));
    TORCH_CHECK(x.sizes().size() == 2, "x must be a 2D matrix");

    auto scale_shape = x.sizes().vec();
    scale_shape.back() /= 32;

    auto q_x = torch::empty_like(x, x.options().dtype(torch::kFloat8_e4m3fn));
    auto scale = torch::empty(scale_shape, x.options().dtype(torch::kFloat8_e8m0fnu));
    const float* x_ptr = x.data_ptr<float>();
    uint8_t* q_x_ptr = (uint8_t*)q_x.data_ptr();
    uint8_t* scale_ptr = (uint8_t*)(scale.data_ptr());
    // int n_elements = x.numel();
    int m = x.size(-2);
    int n = x.size(-1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    quant_fp8_tile_major_launch(x_ptr, q_x_ptr, scale_ptr, m, n, stream);  

    return std::make_tuple(q_x, scale);
}

torch::Tensor dequant_fp8(torch::Tensor x, torch::Tensor scale) {
    check_cuda_contig(x, "x");
    check_dtype(x, torch::kFloat8_e4m3fn, "x");
    check_cuda_contig(scale, "scale");
    check_dtype(scale, torch::kFloat8_e8m0fnu, "scale");

    auto x_dq = torch::empty_like(x, x.options().dtype(torch::kFloat32));
    const uint8_t* x_ptr = (uint8_t*)x.data_ptr();
    const uint8_t* scale_ptr = (uint8_t*)scale.data_ptr();
    float* x_dq_ptr = x_dq.data_ptr<float>();
    // int n_elements = x.numel();
    int m = x.size(-2);
    int n = x.size(-1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dequant_fp8_tile_major_launch(x_ptr, scale_ptr, x_dq_ptr, m, n, stream);

    return x_dq;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_fp4_scale_only", &quant_fp4_scale_only);
    m.def("quant_fp4", &quant_fp4<false>);
    m.def("quant_fp4_stochastic", &quant_fp4<true>);
    m.def("unpack_fp4", &unpack_fp4);
    m.def("dequant_fp4", &dequant_fp4);
    m.def("rh_quant_fp4_stochastic_fused", &rh_quant_fp4_fused<true>);
    m.def("quant_fp4_dequant_trans_fused", &quant_fp4_dequant_trans_fused);
    m.def("quant_fp4_dequant_trans_rh_requant_fused", &quant_fp4_dequant_trans_rh_requant_fused);
    m.def("dequant_fp4_rh_requant_fused", &dequant_fp4_rh_requant_fused);
    m.def("rh_quant_fp4_trans_rh_requant_fused", &rh_quant_fp4_trans_rh_requant_fused);
    m.def("rh_quant_fp4_trans_rh_requant_fp4_fp8_fused", &rh_quant_fp4_trans_rh_requant_fp4_fp8_fused);

    m.def("quant_fp8", &quant_fp8);
    m.def("dequant_fp8", &dequant_fp8);
    m.def("quant_fp8_dequant_trans_requant_fused", &quant_fp8_dequant_trans_requant_fused);
}