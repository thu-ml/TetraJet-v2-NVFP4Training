import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from typing import Optional

from dataclasses import asdict, dataclass, field
from olmo.config import NVFP4_QuantizerConfig
import tetrajetv2
from .utils import *
from torch.amp import custom_bwd, custom_fwd

import logging
log = logging.getLogger(__name__)

class NVFP4_Linear_RealQuant_WeightQuantOnline(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 device=None, dtype=None,
                 q_cfg: NVFP4_QuantizerConfig=field(default_factory=NVFP4_QuantizerConfig), 
                 layer_type=''):
        super(NVFP4_Linear_RealQuant_WeightQuantOnline, self).__init__(
            in_features, out_features, bias,
            device=device, dtype=dtype)
        
        assert (
            q_cfg is not None
            and q_cfg.inner_blocksize == 16
            and q_cfg.outer_blocksize == 128
            and q_cfg.enable_qweight_buffer_when_real_quantize == False
        )
        assert (
            q_cfg.enabled_quantizers == [1, 2, 3, 4, 5, 6]
            and q_cfg.backward_double_quant == True
            and q_cfg.backward_stochastic == True
        )
        assert (
            q_cfg.hadamard_bwd
            and q_cfg.hadamard_for_dX_if_bwd and q_cfg.hadamard_for_dW_if_bwd
        )
        assert (
            not q_cfg.outlier_conf.X_chan_select 
            and not q_cfg.outlier_conf.X_chan_select_bwd
        )
        
        self.q_cfg = q_cfg
        self.MM_func = NVFP4_TetraJetv2_MatMulFunc_BwdAllRHT16SQ_Padding128.apply
        
        self.in_features = in_features
        self.out_features = out_features
        
        quantize_flag = format_string_with_condition(layer_type, shape=self.weight.shape, condition=True)
        log.info(quantize_flag + f" : RealQuant{self.q_cfg.enabled_quantizers}")

    def forward(self, x: torch.Tensor,
                is_first_microbatch: Optional[bool] = None
        ) -> torch.Tensor:
        
        y = self.MM_func(
            x, self.weight
        )
        return y

class NVFP4_TetraJetv2_MatMulFunc_BwdAllRHT16SQ_Padding128(Function):
    @staticmethod
    @torch.compiler.disable
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, x, w):
        # Perform dimension checks and apply padding to tokens
        x, pad_len, x_shape = validate_and_pad(x, w, alignment=128)
        out_features = w.shape[0]
        
        seed_dw = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())
        seed_dx = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())

        # Quantization and transformation (Internal kernels handle the padded x)
        (
            x_q1, x_q1_os, x_q1_is, 
            x_q1dq_t_rhq6, x_q1dq_t_rhq6_os, x_q1dq_t_rhq6_is
        ) = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x, seed_dw)
        
        (
            w_q2, w_q2_os, w_q2_is, 
            w_q2dq_t_rhq4, w_q2dq_t_rhq4_os, w_q2dq_t_rhq4_is
        ) = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(w, seed_dx)
        
        ctx.seed_dw = seed_dw
        ctx.seed_dx = seed_dx
        ctx.pad_len = pad_len
        ctx.x_shape = x_shape
        
        # Save tensors needed for backward
        # Note: we save the padded/transformed versions
        ctx.save_for_backward(
            x_q1dq_t_rhq6, x_q1dq_t_rhq6_os, x_q1dq_t_rhq6_is, 
            w_q2dq_t_rhq4, w_q2dq_t_rhq4_os, w_q2dq_t_rhq4_is
        )
        
        # Core GEMM calculation
        y = tetrajetv2.fp4_gemm(
            x_q1, w_q2,
            x_q1_is, w_q2_is,
            x_q1_os, w_q2_os
        )
        
        # Remove padding and restore original shape for the output
        y = unpad_and_reshape(y, pad_len, x_shape, out_features)
        
        return y
    
    @staticmethod
    @torch.compiler.disable
    def backward(ctx, dy):
        dy = dy.float()
        seed_dw = ctx.seed_dw
        seed_dx = ctx.seed_dx
        pad_len = ctx.pad_len
        x_shape = ctx.x_shape
        
        (
            x_q1dq_t_rhq6, x_q1dq_t_rhq6_os, x_q1dq_t_rhq6_is,
            w_q2dq_t_rhq4, w_q2dq_t_rhq4_os, w_q2dq_t_rhq4_is
        ) = ctx.saved_tensors

        # Pad dy to align with the padded x used during forward
        dy = pad_gradient(dy, pad_len)

        # Fused quantization for gradients
        (
            dy_rhq3, dy_rhq3_os, dy_rhq3_is, 
            dy_t_rhq5, dy_t_rhq5_os, dy_t_rhq5_is
        ) = tetrajetv2.rh_quant_fp4_trans_rh_requant_fused(dy, seed_dx, seed_dw) 

        # Compute gradients for input (dx) and weights (dw)
        dx = tetrajetv2.fp4_gemm(
            dy_rhq3, w_q2dq_t_rhq4, 
            dy_rhq3_is, w_q2dq_t_rhq4_is, 
            dy_rhq3_os, w_q2dq_t_rhq4_os
        )
        dw = tetrajetv2.fp4_gemm(
            dy_t_rhq5, x_q1dq_t_rhq6, 
            dy_t_rhq5_is, x_q1dq_t_rhq6_is, 
            dy_t_rhq5_os, x_q1dq_t_rhq6_os
        )
        dx = unpad_and_reshape(dx, pad_len, x_shape)

        # Clean up dx: remove padding and reshape back to input x shape
        return dx, dw
