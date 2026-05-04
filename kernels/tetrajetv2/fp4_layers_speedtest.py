import torch
import torch.nn as nn
import torch.nn.functional as F
import tetrajetv2
from tetrajetv2.ops_speedtest import FP4MatMulFunction
from tetrajetv2.performance_probe import PerformanceProbe

import transformer_engine.pytorch as te

class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('w_q', torch.zeros((out_features, in_features // 2), dtype=torch.uint8,
                                                device='cuda'))
        self.register_buffer('w_os', torch.zeros((out_features, in_features // 128), dtype=torch.float,
                                                device='cuda'))
        self.register_buffer('w_is', torch.zeros((out_features, in_features // 16), dtype=torch.float8_e4m3fn,
                                                 device='cuda'))
        self.register_buffer('w_t', torch.zeros((in_features, out_features), device='cuda'))
        
        w = torch.empty(out_features, in_features, device='cuda')
        torch.nn.init.normal_(w, mean=0.0, std=0.023)
        # Speedtest assumption: weights are pre-quantized offline after the
        # optimizer step. The current OLMo training path quantizes weights
        # online; TODO: add offline weight quantization there.
        self.w_q, self.w_os, self.w_is, self.w_t = tetrajetv2.quant_fp4_dequant_trans_fused(w)
    
    def forward(self, x):
        x = x.float()
        x_shape = x.shape
        if len(x_shape) > 2:
            x = x.view(-1, x_shape[-1])
        y = FP4MatMulFunction.apply(
            x, self.w_q, self.w_os, self.w_is, self.w_t
        )
        if len(x_shape) > 2:
            y = y.view(*x_shape[:-1], self.out_features)
        return y
    
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, probe_on):
        super().__init__()
        if probe_on:
            self.gate_up_proj = PerformanceProbe(FP4Linear(hidden_size, intermediate_size * 2), "Linear")
            self.down_proj = PerformanceProbe(FP4Linear(intermediate_size, hidden_size), "Linear")
        else:
            self.gate_up_proj = FP4Linear(hidden_size, intermediate_size * 2)
            self.down_proj = FP4Linear(intermediate_size, hidden_size)
        self.intermediate_size = intermediate_size
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_up: torch.Tensor = self.gate_up_proj(x)
        gate, up = gate_up.split([self.intermediate_size] * 2, dim=-1)
        x = self.act_fn(gate) * up
        return self.down_proj(x)
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, probe_on):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        if probe_on:
            self.qkv_proj = PerformanceProbe(FP4Linear(hidden_size, hidden_size * 3), "Linear")
            self.o_proj = PerformanceProbe(FP4Linear(hidden_size, hidden_size), "Linear")
        else:
            self.qkv_proj = FP4Linear(hidden_size, hidden_size * 3)
            self.o_proj = FP4Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        
        qkv: torch.Tensor = self.qkv_proj(hidden_states).view(bsz, q_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # bsz, num_heads, q_len, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class FP4TransformerBlock(nn.Module):
    def __init__(self, hidden_size=4096, num_heads=32, intermediate_size=16384, probe_on=False):
        super().__init__()
        self.input_layernorm = te.RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, probe_on)
        self.post_attention_layernorm = te.RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, probe_on)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
