import torch
from collections import defaultdict
from contextlib import contextmanager

import tetrajetv2
from tetrajetv2.fp4_layers_speedtest import FP4TransformerBlock
from tetrajetv2.fp8_layers_speedtest import FP8TransformerBlock
from tetrajetv2.performance_probe import TIME_STATS
from tetrajetv2.mixed_precision_layers_speedtest import MixedTransformerBlock

from tests.test_basics import *

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

from torch.profiler import profile, ProfilerActivity
from datetime import datetime

def cos_sim(X: torch.Tensor, Y: torch.Tensor):
    X = X.contiguous().flatten().to(torch.float64)
    Y = Y.contiguous().flatten().to(torch.float64)
    return (X * Y).sum() / (X.norm(dtype=torch.float64) * Y.norm(dtype=torch.float64))


warm_up = 20
run_iter = 100

class GPUTimer:
    def __init__(self):
        self.timings = defaultdict(float)
    
    @contextmanager
    def measure(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        self.timings[name] += start.elapsed_time(end)

    def print_stats(self, iterations):
        print("\n=== Component Breakdown (Avg ms per iter) ===")
        total_time = sum(self.timings.values())
        sorted_stats = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for name, total_ms in sorted_stats:
            avg_ms = total_ms / iterations
            percentage = (total_ms / total_time) * 100
            print(f"{name:<15}: {avg_ms:>8.4f} ms ({percentage:>5.1f}%)")
        print("=============================================\n")

@torch.no_grad()
def test_fwd(dim=4096, timer_on=False):
    print("-----------------------------------------")
    print("            Testing Forward              ")
    print("-----------------------------------------")

    torch.manual_seed(42)
    m = dim
    n = dim
    k = dim
    x = torch.randn(m, k, device='cuda', dtype=torch.float32)
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)
    q_w, outer_scale_w, inner_scale_w = tetrajetv2.quant_fp4(w, 0)
    seed_dw = torch.randint(0, 2**32, size=(), dtype=torch.uint32)

    def run_forward_step():
        q_x, outer_scale_x, inner_scale_x, _, _, _ = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x, seed_dw)
        y = tetrajetv2.fp4_gemm(
            q_x, q_w,
            inner_scale_x, inner_scale_w,
            outer_scale_x, outer_scale_w
        )
        return y
    
    def run_forward_step_with_timer(timer):
        ctx = timer.measure
        with ctx("quant"):
            q_x, outer_scale_x, inner_scale_x, _, _, _ = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x, seed_dw)

        with ctx("gemm"):
            y = tetrajetv2.fp4_gemm(
                q_x, q_w,
                inner_scale_x, inner_scale_w,
                outer_scale_x, outer_scale_w,
            )
        return y

    warm_up = 10
    run_iter = 100

    for _ in range(warm_up):
        y = run_forward_step()

    if dim < 32768:
        y_base = x @ w.T
        print(f"Cos sim of y: {cos_sim(y, y_base)}")
        
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    timer = GPUTimer()

    if timer_on:
        run = lambda: run_forward_step_with_timer(timer)
    else:
        run = run_forward_step

    start.record()
    for _ in range(run_iter):
        y = run()
    end.record()
    torch.cuda.synchronize()

    if timer_on:
        timer.print_stats(run_iter)

    ms = start.elapsed_time(end) / run_iter
    ops = m * n * k * 2.0
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"ops: {ops}")
    print(f"time: {ms}ms")
    print(f"tflops: {tflops}")
    print(y.shape)
    print(f"| {y.shape[0]} | {ms:.5f} | {tflops:.2f} |")

@torch.no_grad()
def test_fwd_mix(dim=4096, p = 8, timer_on=False):
    print("-----------------------------------------")
    print("    Testing Forward (mixed precision)    ")
    print("-----------------------------------------")

    torch.manual_seed(42)
    m = dim
    n = dim
    k = dim
    k_fp8 = k // p
    k_fp8 = (k_fp8 + 127) // 128 * 128
    k_fp4 = k - k_fp8
    print(f"1/{p} in fp8")
    if k_fp8 < 128:
        print(f"k_fp8 is {k_fp8}, which is too small!")
        return
    x_fp4 = torch.randn(m, k_fp4, device='cuda', dtype=torch.float32)
    w_fp4 = torch.randn(n, k_fp4, device='cuda', dtype=torch.float32)
    x_fp8 = torch.randn(m, k_fp8, device='cuda', dtype=torch.float32)
    w_fp8 = torch.randn(n, k_fp8, device='cuda', dtype=torch.float32)
    q_w_fp4, outer_scale_w_fp4, inner_scale_w_fp4 = tetrajetv2.quant_fp4(w_fp4, 0)
    q_w_fp8, scale_w_fp8 = tetrajetv2.quant_fp8(w_fp8)

    stream_fp8 = torch.cuda.Stream()

    def run_forward_step():
        q_x_fp4, outer_scale_x_fp4, inner_scale_x_fp4, _, _, _ = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x_fp4, 0)
        with torch.cuda.stream(stream_fp8):
            q_x_fp8, scale_x_fp8, _, _ = tetrajetv2.quant_fp8_dequant_trans_requant_fused(x_fp8)
        y = tetrajetv2.fp4_gemm(q_x_fp4, q_w_fp4,
                                    inner_scale_x_fp4, inner_scale_w_fp4,
                                    outer_scale_x_fp4, outer_scale_w_fp4)
        torch.cuda.current_stream().wait_stream(stream_fp8)
        y = tetrajetv2.fp8_gemm_accum(q_x_fp8, q_w_fp8, scale_x_fp8, scale_w_fp8, y)

        return y

    def run_forward_step_with_timer(timer): # without stream parallel
        ctx = timer.measure
        with ctx("quant_fp4"):
            q_x_fp4, outer_scale_x_fp4, inner_scale_x_fp4, _, _, _ = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x_fp4, 0)
        
        with ctx("fp4_gemm"):
            y = tetrajetv2.fp4_gemm(q_x_fp4, q_w_fp4,
                                        inner_scale_x_fp4, inner_scale_w_fp4,
                                        outer_scale_x_fp4, outer_scale_w_fp4)
        with ctx("quant_fp8"):
            q_x_fp8, scale_x_fp8, _, _ = tetrajetv2.quant_fp8_dequant_trans_requant_fused(x_fp8)
        
        with ctx("fp8_gemm"):
            y = tetrajetv2.fp8_gemm_accum(q_x_fp8, q_w_fp8, scale_x_fp8, scale_w_fp8, y)

        return y
    
    timer = GPUTimer()
    if timer_on:
        run = lambda: run_forward_step_with_timer(timer)
    else:
        run = run_forward_step

    warm_up = 10
    run_iter = 100

    for _ in range(warm_up):
        y = run()

    if dim < 32768:
        y_base = x_fp4 @ w_fp4.T + x_fp8 @ w_fp8.T
        print(f"Cos sim of y: {cos_sim(y_base, y)}")
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(run_iter):
        y = run()
    end.record()
    torch.cuda.synchronize()

    if timer_on:
        timer.print_stats(run_iter)
    
    ms = start.elapsed_time(end) / run_iter
    ops = m * n * k * 2.0
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"ops: {ops}")
    print(f"time: {ms}ms")
    print(f"tflops:{tflops}")
    print(y.shape)
    print(f"| {y.shape[0]} | {ms:.5f} | {tflops:.2f} |")

def test_fp8_baseline_fwd(dim=4096):
    print("-----------------------------------------")
    print("        Testing TE FP8 forward           ")
    print("-----------------------------------------")
    m = dim
    n = dim
    k = dim
    device = torch.device('cuda')
    dtype = torch.float32
    x = torch.randn(m, k, device=device, dtype=dtype)
    te_linear = te.Linear(k, n, bias=False).to(device)

    recipe = DelayedScaling(margin=0, fp8_format=Format.E4M3)

    warm_up = 10
    run_iter = 100

    for _ in range(warm_up):
        with torch.autocast('cuda', torch.bfloat16):
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = te_linear(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        x @ x.T

    start.record()
    for _ in range(run_iter):
        with torch.autocast('cuda', torch.bfloat16):
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = te_linear(x)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / run_iter
    ops = m * n * k * 2.0
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"the ops is: {ops}")
    print(f"the time is: {ms}ms")
    print(f"the tflops is:{tflops}")
    print(y.shape)
    print(f"| {y.shape[0]} | {ms:.5f} | {tflops:.2f} |")

def test_fp8_baseline_bwd(dim=4096):
    print("-----------------------------------------")
    print("        Testing TE FP8 backward           ")
    print("-----------------------------------------")
    m = dim
    n = dim
    k = dim
    device = torch.device('cuda')
    x = torch.randn(m, k, device=device, dtype=torch.float32, requires_grad=True)
    dy = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    te_linear = te.Linear(k, n, bias=False).to(device)

    recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID)

    warm_up = 10
    run_iter = 100

    for _ in range(warm_up):
        with torch.autocast('cuda', torch.bfloat16):
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = te_linear(x)
            y.backward(dy)
            x.grad = None
            te_linear.weight.grad = None
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    bw_time = 0.0

    for _ in range(5):
        x @ x.T

    for _ in range(run_iter):
        with torch.autocast('cuda', torch.bfloat16):
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                y = te_linear(x)
            torch.cuda.synchronize()
            start.record()
            y.backward(dy)
            end.record()
            torch.cuda.synchronize()
            bw_time += start.elapsed_time(end)
            x.grad = None
            te_linear.weight.grad = None

    ms = bw_time / run_iter
    ops = m * n * k * 4.0
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"the ops is: {ops}")
    print(f"the time is: {ms}ms")
    print(f"the tflops is:{tflops}")
    print(y.shape)
    print(f"| {y.shape[0]} | {ms:.5f} | {tflops:.2f} |")

@torch.no_grad()
def test_bwd(dim=4096, timer_on=False):
    print("-----------------------------------------")
    print("           Testing backward              ")
    print("-----------------------------------------")
    print(f"dim is: {dim}")
    if dim >= 32768:
        print("too large!")
        return

    m = dim
    n = dim
    k = dim

    x = torch.randn(m, k, device='cuda', dtype=torch.float32)
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)
    dy = torch.randn(m, n, device='cuda', dtype=torch.float32)

    seed_dx = torch.randint(0, 2**32, size=(), dtype=torch.uint32)
    seed_dw = torch.randint(0, 2**32, size=(), dtype=torch.uint32)

    q_x, outer_scale_x, inner_scale_x, x_t_sq, x_t_os, x_t_is = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x, seed_dw)
    
    q_w, outer_scale_w, inner_scale_w, w_t = tetrajetv2.quant_fp4_dequant_trans_fused(w)

    w_dq = tetrajetv2.dequant_fp4(q_w, outer_scale_w, inner_scale_w)

    def run_backward_step():
        dy_sq, dy_os, dy_is, dy_t_sq, dy_t_os, dy_t_is = tetrajetv2.rh_quant_fp4_trans_rh_requant_fused(dy, seed_dx, seed_dw)
        w_t_sq, w_t_s_os, w_t_s_is = tetrajetv2.rh_quant_fp4_stochastic_fused(w_t, seed_dx)

        dx = tetrajetv2.fp4_gemm(dy_sq, w_t_sq, dy_is, w_t_s_is, dy_os, w_t_s_os)

        dw = tetrajetv2.fp4_gemm(dy_t_sq, x_t_sq, dy_t_is, x_t_is, dy_t_os, x_t_os)

        return dx, dw
    
    def run_backward_step_with_timer(timer=None):
        if timer is None:
            from contextlib import nullcontext
            ctx = lambda x: nullcontext()
        else:
            ctx = timer.measure
        
        with ctx("hadamard&quant"):
            dy_sq, dy_os, dy_is, dy_t_sq, dy_t_os, dy_t_is = tetrajetv2.rh_quant_fp4_trans_rh_requant_fused(dy, seed_dx, seed_dw)
            w_t_sq, w_t_s_os, w_t_s_is = tetrajetv2.rh_quant_fp4_stochastic_fused(w_t, seed_dx)

        with ctx("gemm"):
            dx = tetrajetv2.fp4_gemm(dy_sq, w_t_sq, dy_is, w_t_s_is, dy_os, w_t_s_os)

        with ctx("gemm"):
            dw = tetrajetv2.fp4_gemm(dy_t_sq, x_t_sq, dy_t_is, x_t_is, dy_t_os, x_t_os)

        return dx, dw
 
    warm_up = 200
    run_iter = 100

    dx = torch.zeros_like(x, dtype=torch.float32)
    dw = torch.zeros_like(w, dtype=torch.float32)

    for _ in range(warm_up):
        seed_dx = torch.randint(0, 2**32, size=(), dtype=torch.uint32)
        seed_dw = torch.randint(0, 2**32, size=(), dtype=torch.uint32)
        q_x, outer_scale_x, inner_scale_x, x_t_sq, x_t_os, x_t_is = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x, seed_dw)
        dx_tmp, dw_tmp = run_backward_step()
        dx += dx_tmp
        dw += dw_tmp

    dx /= warm_up
    dw /= warm_up
    dx_base = dy @ w_dq
    x_dq = tetrajetv2.dequant_fp4(q_x, outer_scale_x, inner_scale_x)
    dw_base = dy.T @ x_dq
    if dim <= 16384:
        print(f"dx cos sim: {cos_sim(dx, dx_base)}")
        print(f"dw cos sim: {cos_sim(dw, dw_base)}")
        
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    timer = GPUTimer()

    if timer_on:
        run = lambda: run_backward_step_with_timer(timer)
    else:
        run = run_backward_step

    start.record()
    for _ in range(run_iter):
        run()
    end.record()
    torch.cuda.synchronize()

    if timer_on:
        timer.print_stats(run_iter)

    ms = start.elapsed_time(end) / run_iter
    ops = m * n * k * 4.0
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"ops: {ops}")
    print(f"time: {ms}ms")
    print(f"tflops: {tflops}")
    print(f"| {x.shape[0]} | {ms:.5f} | {tflops:.2f} |")

@torch.no_grad()
def test_bwd_mix(dim=4096, p=8, timer_on=False):
    print("-----------------------------------------")
    print("   Testing backward (mixed precision)    ")
    print("-----------------------------------------")
    print(f"dim is: {dim}")
    if dim >= 32768:
        print("too large!")
        return

    m = dim
    n = dim
    k = dim
    k_fp8 = k // p
    k_fp8 = (k_fp8 + 127) // 128 * 128
    print(f"1/{p} in fp8")
    print(f"fp8 channels: {k_fp8}")
    k_fp4 = k - k_fp8

    if k_fp8 < 128:
        print(f"k_fp8 is {k_fp8}, which is too small!")
        return

    x_fp4 = torch.randn(m, k_fp4, device='cuda', dtype=torch.float32)
    w_fp4 = torch.randn(n, k_fp4, device='cuda', dtype=torch.float32)
    x_fp8 = torch.randn(m, k_fp8, device='cuda', dtype=torch.float32)
    w_fp8 = torch.randn(n, k_fp8, device='cuda', dtype=torch.float32)
    w = torch.cat((w_fp4, w_fp8), dim=1)
    dy = torch.randn(m, n, device='cuda', dtype=torch.float32)

    seed_dx = torch.randint(0, 2**32, size=(), dtype=torch.uint32)
    seed_dw = torch.randint(0, 2**32, size=(), dtype=torch.uint32)

    q_x_fp4, fp4_outer_scale_x, fp4_inner_scale_x, fp4_x_t_sq, fp4_x_t_os, fp4_x_t_is = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x_fp4, seed_dw)
    q_x_fp8, fp8_scale_x, q_x_t_fp8, fp8_scale_x_t = tetrajetv2.quant_fp8_dequant_trans_requant_fused(x_fp8)

    q_w, outer_scale_w, inner_scale_w, w_t = tetrajetv2.quant_fp4_dequant_trans_fused(w)

    w_dq = tetrajetv2.dequant_fp4(q_w, outer_scale_w, inner_scale_w)
    stream_cur = torch.cuda.current_stream()
    stream_w_t = torch.cuda.Stream()

    def run_backward_step():
        dy_sq, dy_os, dy_is, dy_t_fp4_sq, dy_t_fp4_os, dy_t_fp4_is, dy_t_fp8, dy_t_fp8_s = tetrajetv2.rh_quant_fp4_trans_rh_requant_fp4_fp8_fused(dy, seed_dx, seed_dw)
        
        with torch.cuda.stream(stream_w_t):
            w_t_sq, w_t_s_os, w_t_s_is = tetrajetv2.rh_quant_fp4_stochastic_fused(w_t, seed_dx)

        stream_cur.wait_stream(stream_w_t)
        dx = tetrajetv2.fp4_gemm(dy_sq, w_t_sq, dy_is, w_t_s_is, dy_os, w_t_s_os)

        dw_fp4 = tetrajetv2.fp4_gemm(dy_t_fp4_sq, fp4_x_t_sq, dy_t_fp4_is, fp4_x_t_is, dy_t_fp4_os, fp4_x_t_os)

        dw_fp8 = tetrajetv2.fp8_gemm(dy_t_fp8, q_x_t_fp8, dy_t_fp8_s, fp8_scale_x_t)

        return dx, dw_fp4, dw_fp8
    
    def run_backward_step_with_timer(timer=None): # without stream parallel
        if timer is None:
            from contextlib import nullcontext
            ctx = lambda x: nullcontext()
        else:
            ctx = timer.measure
        
        with ctx("dy_quant_fuse"):
            dy_sq, dy_os, dy_is, dy_t_fp4_sq, dy_t_fp4_os, dy_t_fp4_is, dy_t_fp8, dy_t_fp8_s = tetrajetv2.rh_quant_fp4_trans_rh_requant_fp4_fp8_fused(dy, seed_dx, seed_dw)
        with ctx("w_t_quant"):
            w_t_sq, w_t_s_os, w_t_s_is = tetrajetv2.rh_quant_fp4_stochastic_fused(w_t, seed_dx)

        with ctx("fp4_gemm"):
            dx = tetrajetv2.fp4_gemm(dy_sq, w_t_sq, dy_is, w_t_s_is, dy_os, w_t_s_os)
            dw_fp4 = tetrajetv2.fp4_gemm(dy_t_fp4_sq, fp4_x_t_sq, dy_t_fp4_is, fp4_x_t_is, dy_t_fp4_os, fp4_x_t_os)

        with ctx("fp8_gemm"):
            dw_fp8 = tetrajetv2.fp8_gemm(dy_t_fp8, q_x_t_fp8, dy_t_fp8_s, fp8_scale_x_t)

        return dx, dw_fp4, dw_fp8

    warm_up = 20
    run_iter = 100

    dx = torch.zeros(m, k, dtype=torch.float32, device='cuda')
    dw_fp4 = torch.zeros(n, k_fp4, dtype=torch.float32, device='cuda')

    for _ in range(warm_up):
        seed_dx = torch.randint(0, 2**32, size=(), dtype=torch.uint32)
        seed_dw = torch.randint(0, 2**32, size=(), dtype=torch.uint32)
        q_x_fp4, fp4_outer_scale_x, fp4_inner_scale_x, fp4_x_t_sq, fp4_x_t_os, fp4_x_t_is = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x_fp4, seed_dw)
        stream_w_t.wait_stream(stream_cur)
        dx_tmp, dw_fp4_tmp, dw_fp8 = run_backward_step()
        dx += dx_tmp
        dw_fp4 += dw_fp4_tmp
        
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    timer = GPUTimer()

    if timer_on:
        run = lambda: run_backward_step_with_timer(timer)
    else:
        run = run_backward_step

    start.record()
    for _ in range(run_iter):
        run()
    end.record()
    torch.cuda.synchronize()

    if timer_on:
        timer.print_stats(run_iter)

    dx /= warm_up
    dw_fp4 /= warm_up
    dx_base = dy @ w_dq
    x_fp4_dq = tetrajetv2.dequant_fp4(q_x_fp4, fp4_outer_scale_x, fp4_inner_scale_x)
    dw_fp4_base = dy.T @ x_fp4_dq
    x_fp8_dq = tetrajetv2.dequant_fp8(q_x_fp8, fp8_scale_x)
    dw_fp8_base = dy.T @ x_fp8_dq
    if dim <= 16384:
        print(f"dx cos sim: {cos_sim(dx, dx_base)}")
        print(f"dw_fp4 cos sim: {cos_sim(dw_fp4, dw_fp4_base)}")
        print(f"dw_fp8 cos sim: {cos_sim(dw_fp8, dw_fp8_base)}")

    ms = start.elapsed_time(end) / run_iter
    ops = m * n * k * 4.0
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"ops: {ops}")
    print(f"time: {ms}ms")
    print(f"tflops: {tflops}")
    print(f"| {m} | {ms:.5f} | {tflops:.2f} |")

def print_probe_result(run_iter):
    print(f"{'Layer':<15} | {'Phase':<10} | {'Time (ms)':<10}")
    print("-" * 45)

    tot_time = 0.0
    for layer_name, stats in TIME_STATS.items():
        fwd_times = []
        for s, e in stats['fwd']:
            fwd_times.append(s.elapsed_time(e))
        
        bwd_times = []
        for s, e in stats['bwd']:
            bwd_times.append(s.elapsed_time(e))
            
        import numpy as np
        avg_fwd_times = np.sum(fwd_times) / run_iter
        avg_bwd_times = np.sum(bwd_times) / run_iter
        tot_time += avg_fwd_times
        tot_time += avg_bwd_times
        print(f"{layer_name:<15} | Forward    | {avg_fwd_times:.4f}")
        print(f"{layer_name:<15} | Backward   | {avg_bwd_times:.4f}")
    print(f"total time for tested items: {tot_time}")
    return avg_fwd_times, avg_bwd_times

def test_fp8_baseline_e2e(dim=4096, batch_size=4, seq_len=1024, probe_on=False):
    print("-----------------------------------------")
    print("      Testing TE FP8 end to end          ")
    print("-----------------------------------------")
    if dim > 16384:
        print(f"too large! {dim}")
        return
    
    hidden_size = dim
    intermediate_size = int(2 * hidden_size)
    num_heads = 32
    print(f"batch size: {batch_size}")
    print(f"seq_len: {seq_len}")
    print(f"hidden_size: {hidden_size}")
    print(f"intermidiate_size: {intermediate_size*2}")
    print(f"num_heads: {num_heads}")
    device = torch.device("cuda")

    model = FP8TransformerBlock(hidden_size, num_heads, intermediate_size, probe_on).to(device)
    model.train()

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    dy = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    a = torch.randn(4096, 4096, device=device, dtype=torch.float32)

    with torch.autocast('cuda', torch.bfloat16):
        for _ in range(warm_up):
            y = model(x)
            y.backward(dy)
            for param in model.parameters():
                param.grad = None
            x.grad = None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True) as prof:
        with torch.autocast('cuda', torch.bfloat16):
            y = model(x)
            y.backward(dy)
    for param in model.parameters():
        param.grad = None
    x.grad = None
    prof.export_chrome_trace(f"log/base_end2end_{timestamp}.json")

    torch.cuda.synchronize()
    for k in TIME_STATS:
        TIME_STATS[k]['fwd'] = []
        TIME_STATS[k]['bwd'] = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        a @ a

    start.record()
    with torch.autocast('cuda', torch.bfloat16):
        for _ in range(run_iter):
            y = model(x)
            y.backward(dy)
            for param in model.parameters():
                param.grad = None
            x.grad = None
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / run_iter
    tokens = batch_size * seq_len
    linear_attn_ops = 4 * 2 * tokens * (hidden_size ** 2)
    linear_mlp_ops = 3 * 2 * tokens * hidden_size * intermediate_size
    attn_score_ops = 4 * batch_size * (seq_len ** 2) * hidden_size
    fwd_ops = linear_attn_ops + linear_mlp_ops + attn_score_ops
    bwd_ops = 2 * fwd_ops
    ops = fwd_ops + bwd_ops
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"ops: {ops}")
    print(f"time: {ms}ms")
    print(f"tflops: {tflops}")

    if probe_on:
        lf, lb = print_probe_result(run_iter)
        print(f"| {hidden_size} | {batch_size} | {ms:.5f} | {lf:.5f} | {lb:.5f} | {ms - lf - lb:.5f} | {tflops:.2f} |")

def test_e2e(dim=4096, batch_size=4, seq_len=1024, probe_on=False):
    print("-----------------------------------------")
    print("          Testing end to end             ")
    print("-----------------------------------------")
    if dim > 8192 * 2:
        print("too large")
        return
    
    hidden_size = dim
    intermediate_size = int(2 * hidden_size)
    num_heads = 32
    print(f"batch size: {batch_size}")
    print(f"seq_len: {seq_len}")
    print(f"hidden_size: {hidden_size}")
    print(f"intermidiate_size: {intermediate_size*2}")
    print(f"num_heads: {num_heads}")
    dtype = torch.float32
    device = torch.device("cuda")

    model = FP4TransformerBlock(hidden_size, num_heads, intermediate_size, probe_on).to(device)
    model.train()

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    dy = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    for _ in range(warm_up):
        y = model(x)
        y.backward(dy)
        for param in model.parameters():
            param.grad = None
        x.grad = None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True) as prof:
        y = model(x)
        y.backward(dy)
    for param in model.parameters():
        param.grad = None
    x.grad = None
    prof.export_chrome_trace(f"log/end2end_{timestamp}.json")
        
    torch.cuda.synchronize()
    for k in TIME_STATS:
        TIME_STATS[k]['fwd'] = []
        TIME_STATS[k]['bwd'] = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(run_iter):
        y = model(x)
        y.backward(dy)
        for param in model.parameters():
            param.grad = None
        x.grad = None
    end.record()
    torch.cuda.synchronize()
    
    ms = start.elapsed_time(end) / run_iter
    tokens = batch_size * seq_len
    linear_attn_ops = 4 * 2 * tokens * (hidden_size ** 2)
    linear_mlp_ops = 3 * 2 * tokens * hidden_size * intermediate_size
    attn_score_ops = 4 * batch_size * (seq_len ** 2) * hidden_size
    fwd_ops = linear_attn_ops + linear_mlp_ops + attn_score_ops
    bwd_ops = 2 * fwd_ops
    ops = fwd_ops + bwd_ops
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"ops: {ops}")
    print(f"time: {ms}ms")
    print(f"tflops: {tflops}")

    if probe_on:
        lf, lb = print_probe_result(run_iter)
        print(f"| {hidden_size} | {batch_size} | {ms:.5f} | {lf:.5f} | {lb:.5f} | {ms - lf - lb:.5f} | {tflops:.2f} |")


def test_e2e_mix(dim=4096, p=8, batch_size=4, seq_len=1024, probe_on=False):
    print("-----------------------------------------")
    print("   Testing end to end (mixed precision)    ")
    print("-----------------------------------------")
    if dim > 8192 * 2:
        print("too large")
        return
    
    hidden_size = dim
    intermediate_size = int(2 * hidden_size)
    num_heads = 32
    print(f"batch size: {batch_size}")
    print(f"seq_len: {seq_len}")
    print(f"hidden_size: {hidden_size}")
    print(f"intermidiate_size: {intermediate_size*2}")
    print(f"num_heads: {num_heads}")
    print(f"proportion in fp8: 1/{p}")
    dtype = torch.float32
    device = torch.device("cuda")

    model = MixedTransformerBlock(hidden_size, num_heads, intermediate_size, p, probe_on).to(device)
    model.train()

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    dy = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    for _ in range(warm_up):
        y = model(x)
        y.backward(dy)
        for param in model.parameters():
            param.grad = None
        x.grad = None
        
    torch.cuda.synchronize()
    for k in TIME_STATS:
        TIME_STATS[k]['fwd'] = []
        TIME_STATS[k]['bwd'] = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(run_iter):
        y = model(x)
        y.backward(dy)
        for param in model.parameters():
            param.grad = None
        x.grad = None
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / run_iter
    tokens = batch_size * seq_len
    linear_attn_ops = 4 * 2 * tokens * (hidden_size ** 2)
    linear_mlp_ops = 3 * 2 * tokens * hidden_size * intermediate_size
    attn_score_ops = 4 * batch_size * (seq_len ** 2) * hidden_size
    fwd_ops = linear_attn_ops + linear_mlp_ops + attn_score_ops
    bwd_ops = 2 * fwd_ops
    ops = fwd_ops + bwd_ops
    tflops = ops * 1e-12 / (ms / 1000)
    print(f"ops: {ops}")
    print(f"time: {ms}ms")
    print(f"tflops: {tflops}")
    
    if probe_on:
        lf, lb = print_probe_result(run_iter)
        print(f"| {hidden_size} | {batch_size} | {ms:.5f} | {lf:.5f} | {lb:.5f} | {ms - lf - lb:.5f} | {tflops:.2f} |")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    
    test_choices = [
        "scale_only", "quant_fp4", "quant_stochastic",
        "gemm", "gemm_mix", "quant_gemm",
        "base_fwd", "base_bwd", "base_e2e",
        "fwd", "fwd_mix",
        "bwd", "bwd_mix",
        "e2e", "e2e_mix"
    ]

    parser.add_argument("test_name", choices=test_choices, help="test to run")
    
    parser.add_argument("-d", "--dims", type=int, nargs="+", 
                        default=[1024, 2048, 4096, 8192],
                        help="dimensions for test, e.g. --dims 1024 2048")
    
    parser.add_argument("--p_vals", type=int, nargs="+", 
                        default=[8, 10, 16],
                        help="proportion of fp8 in mixed precision test, e.g. --p_vals 8")
    
    parser.add_argument("--bs", type=int,
                        default=4,
                        help="batch size")
    parser.add_argument("--sl", "--seq_len", dest="seq_len", type=int,
                        default=1024,
                        help="sequence length")
    
    parser.add_argument("-t", "--timer", action="store_true", help="test time for each component")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: No CUDA found.")
        sys.exit(1)

    print(f"Running test: {args.test_name}")

    if args.test_name == "scale_only":
        test_scale_only()
    elif args.test_name == "quant_fp4":
        test_quant_fp4()
    elif args.test_name == "quant_stochastic":
        test_quant_stochastic()
    elif args.test_name == "quant_gemm":
        test_quant_gemm()
    
    elif args.test_name == "fwd":
        for i in args.dims:
            test_fwd(dim=i, timer_on=args.timer)
    elif args.test_name == "bwd":
        for i in args.dims:
            test_bwd(dim=i, timer_on=args.timer)
    elif args.test_name == "gemm":
        for i in args.dims:
            test_gemm(dim=i)
    elif args.test_name == "base_fwd":
        for i in args.dims:
            test_fp8_baseline_fwd(dim=i)
    elif args.test_name == "base_bwd":
        for i in args.dims:
            test_fp8_baseline_bwd(dim=i)
    elif args.test_name == "e2e":
        for i in args.dims:
            test_e2e(dim=i, batch_size=args.bs, seq_len=args.seq_len, probe_on=args.timer)
    elif args.test_name == "base_e2e":
        for i in args.dims:
            test_fp8_baseline_e2e(dim=i, batch_size=args.bs, seq_len=args.seq_len, probe_on=args.timer)
            
    elif args.test_name == "fwd_mix":
        for i in args.dims:
            for p in args.p_vals:
                test_fwd_mix(dim=i, p=p, timer_on=args.timer)
    elif args.test_name == "gemm_mix":
        for i in args.dims:
            for p in args.p_vals:
                test_gemm_mix(dim=i, p=p)
    elif args.test_name == "bwd_mix":
        for i in args.dims:
            for p in args.p_vals:
                test_bwd_mix(dim=i, p=p, timer_on=args.timer)
    elif args.test_name == "e2e_mix":
        for i in args.dims:
            for p in args.p_vals:
                test_e2e_mix(dim=i, p=p, batch_size=args.bs, seq_len=args.seq_len, probe_on=args.timer)
