import torch

from baseline_fakequant.nvfp4_quant_fused import nvfp4_scale_only, nvfp4_double_quantize_fused

import tetrajetv2

def cos_sim(X: torch.Tensor, Y: torch.Tensor):
    X = X.contiguous().flatten().to(torch.float64)
    Y = Y.contiguous().flatten().to(torch.float64)
    return (X * Y).sum() / (X.norm(dtype=torch.float64) * Y.norm(dtype=torch.float64))

def test_gemm_mix(dim=4096, p = 8):
    print("-----------------------------------------")
    print("      Testing gemm (mixed precision)     ")
    print("-----------------------------------------")

    torch.manual_seed(42)
    m = dim
    n = dim
    k = dim
    k_fp8 = dim // p
    k_fp4 = k - k_fp8
    a_fp4 = torch.randn(m, k_fp4, device='cuda', dtype=torch.float32)
    b_fp4 = torch.randn(n, k_fp4, device='cuda', dtype=torch.float32)
    a_fp8 = torch.randn(m, k_fp8, device='cuda', dtype=torch.float32)
    b_fp8 = torch.randn(n, k_fp8, device='cuda', dtype=torch.float32)

    print(f"1/{p} in fp8")
    if k_fp8 < 128:
        print(f"k_fp8 is {k_fp8}, which is too small!")
        return

    q_a_fp4, outer_scale_a_fp4, inner_scale_a_fp4 = tetrajetv2.quant_fp4(a_fp4, 0)
    q_a_fp8, scale_a_fp8 = tetrajetv2.quant_fp8(a_fp8)
    q_b_fp4, outer_scale_b_fp4, inner_scale_b_fp4 = tetrajetv2.quant_fp4(b_fp4, 0)
    q_b_fp8, scale_b_fp8 = tetrajetv2.quant_fp8(b_fp8)

    warm_up = 10
    run_iter = 100

    for _ in range(warm_up):
        c = tetrajetv2.fp4_gemm(q_a_fp4, q_b_fp4,
                                    inner_scale_a_fp4, inner_scale_b_fp4,
                                    outer_scale_a_fp4, outer_scale_b_fp4)
        c = tetrajetv2.fp8_gemm_accum(q_a_fp8, q_b_fp8, scale_a_fp8, scale_b_fp8, c)
        
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(run_iter):
        c = tetrajetv2.fp4_gemm(q_a_fp4, q_b_fp4,
                                    inner_scale_a_fp4, inner_scale_b_fp4,
                                    outer_scale_a_fp4, outer_scale_b_fp4)
        c = tetrajetv2.fp8_gemm_accum(q_a_fp8, q_b_fp8, scale_a_fp8, scale_b_fp8, c)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / run_iter
    ops = m * n * k * 2.0
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"the ops is: {ops}")
    print(f"the time is: {ms}ms")
    print(f"the tflops is:{tflops}")
    print(c.shape)



def test_quant_stochastic():
    print("-----------------------------------------")
    print("        Testing Stochastic Quant         ")
    print("-----------------------------------------")
    torch.manual_seed(42)
    dim = 4096
    x = torch.randn(dim, dim, device='cuda', dtype=torch.float32)
    x_avg = torch.zeros_like(x, dtype=torch.float32)
    x_avg_baseline = torch.zeros_like(x, dtype=torch.float32)
    run_times = 1000
    for _ in range(run_times):
        seed = torch.randint(0, 2**32, size=(), dtype=torch.uint32)
        q_x_stochastic, outer_scale, inner_scale = tetrajetv2.quant_fp4_stochastic(x, seed)
        x_st_baseline = nvfp4_double_quantize_fused(x, stochastic=True)
        x_stochastic = tetrajetv2.dequant_fp4(q_x_stochastic, outer_scale, inner_scale)

        x_avg += x_stochastic
        x_avg_baseline += x_st_baseline
    x_avg /= run_times
    x_avg_baseline /= run_times

    print(f"All close: {torch.allclose(x_avg, x_avg_baseline, 1e-4)}")
    print(f"{x_avg[:32]}")
    print(f"{x_avg_baseline[:32]}")

    diff = torch.abs(x_avg - x_avg_baseline)
    max_abs_error = torch.max(diff)
    print(f"Max diff: {max_abs_error.item()}")

    cos_sim_val = cos_sim(x_avg, x_avg_baseline)
    print(f"Cos Similarity: {cos_sim_val}")

    mse = torch.mean((x_avg - x_avg_baseline) ** 2)
    print(f"MSE: {mse}")

def test_quant_fp4():
    print("-----------------------------------------")
    print("             Testing Quant               ")
    print("-----------------------------------------")
    torch.manual_seed(42)
    dim = 4096
    x = torch.randn(dim, dim, device='cuda', dtype=torch.float32)
    
    q_x, outer_scale, inner_scale = tetrajetv2.quant_fp4(x, 0)
    
    x_dequant = tetrajetv2.dequant_fp4(q_x, outer_scale, inner_scale)

    torch.set_printoptions(precision=7)

    print(f"Cuda: {x_dequant[:16]}")

    x_dequant_baseline = nvfp4_double_quantize_fused(x)
    print(f"Baseline: {x_dequant_baseline[:16]}")

    is_equal = torch.equal(x_dequant, x_dequant_baseline)
    print(f"Equal: {is_equal}")

    if not is_equal:
        diff = torch.abs(x_dequant - x_dequant_baseline)
        max_abs_error = torch.max(diff)
        print(f"Max diff: {max_abs_error.item()}")

def test_scale_only():
    print("-----------------------------------------")
    print("          Testing Scale Only             ")
    print("-----------------------------------------")

    # torch.manual_seed(42)
    N = 4096 * 4096
    x = torch.randn(N, device='cuda', dtype=torch.float32)

    y_cuda = tetrajetv2.quant_fp4_scale_only(x)
    y_baseline = nvfp4_scale_only(x)

    print(f"First 16 Input: {x[:16]}")

    y_cuda = y_cuda.reshape(-1, 16)
    y_baseline = y_baseline.reshape(-1, 16)
    print(f"Output: {y_cuda[:8, 0]}")
    print(f"Baseline: {y_baseline[:8, 0]}")

    print(f"Equal: {torch.equal(y_cuda, y_baseline)}")

    #tetrajetv2.quant(x, 0)

def test_quant_gemm():
    print("-----------------------------------------")
    print("        Testing Quant and gemm           ")
    print("-----------------------------------------")

    torch.manual_seed(42)
    m = 4096
    n = 4096
    k = 4096
    a = torch.randn(m, k, device='cuda', dtype=torch.float32)
    b = torch.randn(n, k, device='cuda', dtype=torch.float32)
    
    q_a, outer_scale_a, inner_scale_a = tetrajetv2.quant_fp4(a, 0)
    q_b, outer_scale_b, inner_scale_b = tetrajetv2.quant_fp4(b, 0)

    print("q_a dtype is:", q_a.dtype)
    print("outer_scale_a dtype is:", outer_scale_a.dtype)
    print("innner_scale_a dtype is:", inner_scale_a.dtype)

    result = tetrajetv2.fp4_gemm(q_a, q_b,
                                inner_scale_a, inner_scale_b,
                                outer_scale_a, outer_scale_b)

    base_a = nvfp4_double_quantize_fused(a)
    base_b = nvfp4_double_quantize_fused(b)
    base_a = base_a.reshape(m, k)
    base_b = base_b.reshape(n, k)

    base_result = base_a @ base_b.T

    torch.set_printoptions(precision=7)
    is_equal = torch.equal(result, base_result)
    print(f"Equal: {is_equal}")
    if not is_equal:
        cos_sim_val = cos_sim(result, base_result)
        print(f"Cos similarity is: {cos_sim_val}")
        diff = torch.abs(result - base_result)
        max_abs_error = torch.max(diff)
        print(f"Max diff: {max_abs_error.item()}")
        threshold = 0.8
        num_large_diff = (diff > threshold).sum().item()
        print(f"Number of elements with diff > {threshold}: {num_large_diff}")

def test_gemm(dim=4096):
    print("-----------------------------------------")
    print("        Testing Gemm FLOPS          ")
    print("-----------------------------------------")

    torch.manual_seed(42)
    m = dim
    n = dim
    k = dim
    a = torch.randn(m, k, device='cuda', dtype=torch.float32)
    b = torch.randn(n, k, device='cuda', dtype=torch.float32)
    q_a, outer_scale_a, inner_scale_a = tetrajetv2.quant_fp4(a, 0)
    q_b, outer_scale_b, inner_scale_b = tetrajetv2.quant_fp4(b, 0)

    warm_up = 10
    run_iter = 100

    for _ in range(warm_up):
        result = tetrajetv2.fp4_gemm(q_a, q_b,
                                    inner_scale_a, inner_scale_b,
                                    outer_scale_a, outer_scale_b)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(run_iter):
        result = tetrajetv2.fp4_gemm(q_a, q_b,
                                    inner_scale_a, inner_scale_b,
                                    outer_scale_a, outer_scale_b)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / run_iter
    ops = m * n * k * 2.0
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"the ops is: {ops}")
    print(f"the time is: {ms}ms")
    print(f"the tflops is: {tflops}")
    print(result.shape)
 