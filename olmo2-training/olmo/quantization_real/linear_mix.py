import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from typing import Optional

from dataclasses import asdict, dataclass, field
from olmo.config import NVFP4_QuantizerConfig

import torch.distributed as dist

import tetrajetv2
from .utils import *
from .linear import NVFP4_TetraJetv2_MatMulFunc_BwdAllRHT16SQ_Padding128
from torch.amp import custom_bwd, custom_fwd

import logging
log = logging.getLogger(__name__)

class NVFP4_mix_MXFP8_Linear_RealQuant_WeightQuantOnline(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 device=None, dtype=None,
                 q_cfg: NVFP4_QuantizerConfig=field(default_factory=NVFP4_QuantizerConfig), 
                 layer_type=''):
        super(NVFP4_mix_MXFP8_Linear_RealQuant_WeightQuantOnline, self).__init__(
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
            q_cfg.outlier_conf.X_chan_select 
            and q_cfg.outlier_conf.X_chan_select_bwd
            and q_cfg.outlier_conf.outlier_fp8_matmul
        )
        
        self.q_cfg = q_cfg
        self.layer_type = layer_type
        
        self.MM_func_no_mix = NVFP4_TetraJetv2_MatMulFunc_BwdAllRHT16SQ_Padding128.apply
        # self.MM_func_no_mix = F.linear
        self.MM_func = NVFP4_mix_MXFP8_TetraJetv2_MatMulFunc_BwdAllRHT16SQ_Padding128.apply
        
        self.in_features = in_features
        self.out_features = out_features
        
        # NOTE: Here, we use `fp8_features=ceil(ratio*n_channel/128)*128` to avoid padding_overhead
        #       In the paper, we reported the simulated-accuracy result with `fp8_features=round(ratio*n_channel)`
        self.in_features_fp8 = min(in_features // 128 * 128, 
                                      (int(in_features * q_cfg.outlier_conf.X_top_channels_ratio) + 127) // 128 * 128)
        self.in_features_fp4 = self.in_features - self.in_features_fp8
        assert self.in_features_fp4 > 0 and self.in_features_fp8 > 0
        
        self.register_buffer("fp8_channel_idx", self.weight.new_zeros(size=[self.in_features_fp8], 
                                                                      dtype=torch.int,
                                                                      requires_grad=False), persistent=True)
        self.register_buffer("fp4_channel_idx", self.weight.new_zeros(size=[self.in_features_fp4], 
                                                                      dtype=torch.int,
                                                                      requires_grad=False), persistent=True)
        self.register_buffer("mix_channel_done", self.weight.new_tensor(False, 
                                                                        device='cpu',
                                                                        dtype=torch.bool,
                                                                        requires_grad=False), persistent=True)
        self.accumulating_x_norm = False
        
        quantize_flag = format_string_with_condition(layer_type, shape=self.weight.shape, condition=True)
        log.info(quantize_flag + f" : RealQuant{self.q_cfg.enabled_quantizers}")


    def start_channel_outlier_accumulate(self):
        if not hasattr(self, 'act_norm_channels_sum'):
            self.register_buffer("act_norm_channels_sum", 
                                 self.weight.new_zeros(size=[self.weight.shape[1]],
                                                       dtype=torch.float,
                                                       requires_grad=False), persistent=False)
        else:
            self.act_norm_channels_sum.zero_()
        self.accumulating_x_norm = True
        log.info(f"Layer {self.layer_type}: start accumulating channel norms")
    
    def sync_channel_accumulate_and_update_idx(self):
        if int(os.environ.get('RANK', -1)) != -1:
            dist.all_reduce(self.act_norm_channels_sum, op=dist.ReduceOp.SUM)
        
        # _, topk_indices = torch.topk(self.act_norm_channels_sum, k=self.in_features_fp8, largest=True)
        sorted_indices = torch.argsort(self.act_norm_channels_sum, descending=True)
        
        k_fp8 = self.in_features_fp8
        self.fp8_channel_idx.copy_(sorted_indices[:k_fp8])
        self.fp4_channel_idx.copy_(sorted_indices[k_fp8:])
        
        self.accumulating_x_norm = False
        self.mix_channel_done.fill_(True)
        delattr(self, "act_norm_channels_sum")
        
        log.info(f"Update Layer {self.layer_type} with channel norms: "
                 f"selected {self.fp8_channel_idx.numel():<4}channels out of {self.weight.shape[-1]:<5}")
    
    @torch.compiler.disable
    def forward(self, x: torch.Tensor,
                is_first_microbatch: Optional[bool] = None
        ) -> torch.Tensor:
        
        if self.accumulating_x_norm and self.training:
            assert hasattr(self, 'act_norm_channels_sum')
            assert x.shape[-1] == self.act_norm_channels_sum.shape[0], "Shape not match"
            self.act_norm_channels_sum += x.detach().view(-1, x.shape[-1]).norm(p=2, dim=0)
        
        if not self.mix_channel_done.item():
            y = self.MM_func_no_mix(
                x, self.weight
            )
        else:
            y = self.MM_func(
                x, self.weight,
                self.fp8_channel_idx, self.fp4_channel_idx
            )
        return y

class NVFP4_mix_MXFP8_TetraJetv2_MatMulFunc_BwdAllRHT16SQ_Padding128(Function):
    @staticmethod
    @torch.compiler.disable
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, x_in, w, fp8_channel_idx, fp4_channel_idx):
        # Perform dimension checks and apply padding to tokens
        x, pad_len, x_shape = validate_and_pad(x_in, w, alignment=128)
        x_fp8 = x[:, fp8_channel_idx].contiguous()
        x_fp4 = x[:, fp4_channel_idx].contiguous()
        w_fp8 = w[:, fp8_channel_idx].contiguous()
        w_fp4 = w[:, fp4_channel_idx].contiguous()
        
        out_features = w.shape[0]
        seed_dw = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())
        seed_dx = int(torch.randint(0, 2**32, size=(), dtype=torch.uint32).item())

        # Quantization and transformation (Internal kernels handle the padded x)
        (
            x_q1, x_q1_os, x_q1_is, 
            x_q1dq_t_rhq6, x_q1dq_t_rhq6_os, x_q1dq_t_rhq6_is
        ) = tetrajetv2.quant_fp4_dequant_trans_rh_requant_fused(x_fp4, seed_dw)
        
        (
            w_q2, w_q2_os, w_q2_is, 
            w_q2dq_t
        ) = tetrajetv2.quant_fp4_dequant_trans_fused(w_fp4)
        
        x_q_fp8, x_s_fp8, x_t_q_fp8, x_t_s_fp8 = tetrajetv2.quant_fp8_dequant_trans_requant_fused(x_fp8)
        w_q_fp8, w_s_fp8 = tetrajetv2.quant_fp8(w_fp8)
        
        # Core GEMM calculation
        y = tetrajetv2.fp4_gemm(
            x_q1, w_q2,
            x_q1_is, w_q2_is,
            x_q1_os, w_q2_os
        )
        y = tetrajetv2.fp8_gemm_accum(
            x_q_fp8, w_q_fp8, 
            x_s_fp8, w_s_fp8, 
            y
        )
        
        ctx.seed_dw = seed_dw
        ctx.seed_dx = seed_dx
        ctx.pad_len = pad_len
        ctx.x_shape = x_shape
                
        w_t = w.new_empty(w.shape[1], w.shape[0], requires_grad=False)
        w_t[fp4_channel_idx] = w_q2dq_t
        w_t[fp8_channel_idx] = tetrajetv2.dequant_fp8(w_q_fp8, w_s_fp8).T.contiguous()
        
        # Save tensors needed for backward
        # Note: we save the padded/transformed versions
        ctx.save_for_backward(
            x_q1dq_t_rhq6, x_q1dq_t_rhq6_os, x_q1dq_t_rhq6_is, 
            x_t_q_fp8, x_t_s_fp8,
            w_t,
            fp8_channel_idx, fp4_channel_idx
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
            x_t_q_fp8, x_t_s_fp8,
            w_t,
            fp8_channel_idx, fp4_channel_idx
        ) = ctx.saved_tensors

        # Pad dy to align with the padded x used during forward
        dy = pad_gradient(dy, pad_len)

        # Fused quantization for gradients
        (
            dy_rhq3, dy_rhq3_os, dy_rhq3_is, 
            dy_t_rhq5, dy_t_rhq5_os, dy_t_rhq5_is,
            dy_t_fp8, dy_t_s_fp8
        ) = tetrajetv2.rh_quant_fp4_trans_rh_requant_fp4_fp8_fused(dy, seed_dx, seed_dw) 
        
        w_t_rhq_fp4, w_t_rhq_os_fp4, w_t_rhq_is_fp4 = (
            tetrajetv2.rh_quant_fp4_stochastic_fused(w_t, seed_dx)
        )

        # Compute gradients for input (dx) and weights (dw)
        dx = tetrajetv2.fp4_gemm(
            dy_rhq3, w_t_rhq_fp4, 
            dy_rhq3_is, w_t_rhq_is_fp4, 
            dy_rhq3_os, w_t_rhq_os_fp4
        )
        dx = unpad_and_reshape(dx, pad_len, x_shape)
        
        dw = w_t.new_empty(w_t.shape[1], w_t.shape[0], requires_grad=False,
                           dtype=torch.bfloat16)
        
        dw[:, fp4_channel_idx] = tetrajetv2.fp4_gemm(
            dy_t_rhq5, x_q1dq_t_rhq6, 
            dy_t_rhq5_is, x_q1dq_t_rhq6_is, 
            dy_t_rhq5_os, x_q1dq_t_rhq6_os
        )
        dw[:, fp8_channel_idx] = tetrajetv2.fp8_gemm(
            dy_t_fp8, x_t_q_fp8, 
            dy_t_s_fp8, x_t_s_fp8, 
        )

        # Clean up dx: remove padding and reshape back to input x shape
        return dx, dw, None, None
