import torch
from torch.autograd import Function
import tetrajetv2

class FP4MatMulTrainFunction(Function):
    @staticmethod
    def forward(ctx, x, w):
        seed_dw = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())
        w_q, w_os, w_is, w_t = tetrajetv2.quant_fp4_dequant_trans_fused(w)

        x_q, x_os, x_is, x_t_rhq, x_t_os, x_t_is = (
            tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x, seed_dw)
        )
        y = tetrajetv2.fp4_gemm(
            x_q, w_q,
            x_is, w_is,
            x_os, w_os
        )

        ctx.seed_dw = seed_dw
        ctx.save_for_backward(x_t_rhq, x_t_os, x_t_is, w_t)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        dy = dy.float()
        seed_dw = ctx.seed_dw
        seed_dx = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())

        x_t_rhq, x_t_os, x_t_is, w_t = ctx.saved_tensors

        dy_rhq, dy_os, dy_is, dy_t_rhq, dy_t_os, dy_t_is = tetrajetv2.rh_quant_fp4_trans_rh_requant_fused(dy, seed_dx, seed_dw)
        w_t_rhq, w_t_s_os, w_t_s_is = tetrajetv2.rh_quant_fp4_stochastic_fused(w_t, seed_dx)

        dx = tetrajetv2.fp4_gemm(dy_rhq, w_t_rhq, dy_is, w_t_s_is, dy_os, w_t_s_os)
        dw = tetrajetv2.fp4_gemm(dy_t_rhq, x_t_rhq, dy_t_is, x_t_is, dy_t_os, x_t_os)
        return dx, dw

class FP4MatMulFunction(Function):
    @staticmethod
    def forward(ctx, x, w_q, w_os, w_is, w_t):
        seed_dw = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())

        x_q, x_os, x_is, x_t_rhq, x_t_os, x_t_is = (
            tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x, seed_dw)
        )
        y = tetrajetv2.fp4_gemm(
            x_q, w_q,
            x_is, w_is,
            x_os, w_os
        )

        ctx.seed_dw = seed_dw
        ctx.save_for_backward(x_t_rhq, x_t_os, x_t_is, w_t)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        dy = dy.float()
        seed_dw = ctx.seed_dw
        seed_dx = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())

        x_t_rhq, x_t_os, x_t_is, w_t = ctx.saved_tensors

        dy_rhq, dy_os, dy_is, dy_t_rhq, dy_t_os, dy_t_is = tetrajetv2.rh_quant_fp4_trans_rh_requant_fused(dy, seed_dx, seed_dw)
        w_t_rhq, w_t_s_os, w_t_s_is = tetrajetv2.rh_quant_fp4_stochastic_fused(w_t, seed_dx)

        dx = tetrajetv2.fp4_gemm(dy_rhq, w_t_rhq, dy_is, w_t_s_is, dy_os, w_t_s_os)
        dw = tetrajetv2.fp4_gemm(dy_t_rhq, x_t_rhq, dy_t_is, x_t_is, dy_t_os, x_t_os)
        return dx, dw, None, None, None, None, None

class MixedMatMulFunction(Function):
    @staticmethod
    def forward(ctx, x, w_q_fp4, w_os_fp4, w_is_fp4, w_t, w_q_fp8, w_s_fp8, p):
        k = x.size(-1)
        k_fp8 = k // p
        k_fp8 = (k_fp8 + 127) // 128 * 128
        k_fp4 = k - k_fp8
        x_fp4, x_fp8 = torch.split(x, [k_fp4, k_fp8], dim=-1)
        x_fp4 = x_fp4.contiguous()
        x_fp8 = x_fp8.contiguous()

        seed_dw = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())

        x_q_fp4, x_os_fp4, x_is_fp4, x_t_rhq_fp4, x_t_os_fp4, x_t_is_fp4 = (
            tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x_fp4, seed_dw)
        )
        x_q_fp8, x_s_fp8, x_t_q_fp8, x_t_s_fp8 = tetrajetv2.quant_fp8_dequant_trans_requant_fused(x_fp8)
        y = tetrajetv2.fp4_gemm(
            x_q_fp4, w_q_fp4,
            x_is_fp4, w_is_fp4,
            x_os_fp4, w_os_fp4
        )
        y = tetrajetv2.fp8_gemm_accum(x_q_fp8, w_q_fp8, x_s_fp8, w_s_fp8, y)

        ctx.seed_dw = seed_dw
        ctx.save_for_backward(
            x_t_rhq_fp4, x_t_os_fp4, x_t_is_fp4, w_t, x_t_q_fp8, x_t_s_fp8
        )
        return y
    
    @staticmethod
    def backward(ctx, dy):
        dy = dy.float()
        seed_dw = ctx.seed_dw
        seed_dx = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())

        x_t_rhq_fp4, x_t_os_fp4, x_t_is_fp4, w_t, x_t_q_fp8, x_t_s_fp8 = (
            ctx.saved_tensors
        )
        dy_rhq, dy_os, dy_is, dy_t_rhq_fp4, dy_t_os_fp4, dy_t_is_fp4, dy_t_fp8, dy_t_s_fp8 = (
            tetrajetv2.rh_quant_fp4_trans_rh_requant_fp4_fp8_fused(dy, seed_dx, seed_dw)
        )
        
        w_t_rhq_fp4, w_t_rhq_os_fp4, w_t_rhq_is_fp4 = (
            tetrajetv2.rh_quant_fp4_stochastic_fused(w_t, seed_dx)
        )

        dx = tetrajetv2.fp4_gemm(dy_rhq, w_t_rhq_fp4, dy_is, w_t_rhq_is_fp4, dy_os, w_t_rhq_os_fp4)
        dw_fp4 = tetrajetv2.fp4_gemm(dy_t_rhq_fp4, x_t_rhq_fp4, dy_t_is_fp4, x_t_is_fp4, dy_t_os_fp4, x_t_os_fp4)
        dw_fp8 = tetrajetv2.fp8_gemm(dy_t_fp8, x_t_q_fp8, dy_t_s_fp8, x_t_s_fp8)
        return dx, dw_fp4, None, None, None, dw_fp8, None, None