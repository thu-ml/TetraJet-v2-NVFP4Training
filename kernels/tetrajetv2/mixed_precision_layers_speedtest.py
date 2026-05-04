import torch
import torch.nn as nn
import torch.nn.functional as F
import tetrajetv2
from tetrajetv2.ops_speedtest import MixedMatMulFunction
from tetrajetv2.performance_probe import PerformanceProbe

import transformer_engine.pytorch as te

class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, p):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        k_fp8 = in_features // p
        k_fp8 = (k_fp8 + 127) // 128 * 128
        k_fp4 = in_features - k_fp8
        self.register_buffer('w_q_fp4', torch.zeros((out_features, k_fp4 // 2), dtype=torch.uint8))
        self.register_buffer('w_os_fp4', torch.zeros((out_features, k_fp4 // 128), dtype=torch.float))
        self.register_buffer('w_is_fp4', torch.zeros((out_features, k_fp4 // 16), dtype=torch.float8_e4m3fn))
        self.register_buffer('w_t', torch.zeros((in_features, out_features), dtype=torch.float))
        self.register_buffer('w_q_fp8', torch.zeros((out_features, k_fp8), dtype=torch.float8_e4m3fn))
        self.register_buffer('w_s_fp8', torch.zeros((out_features, k_fp8 // 32), dtype=torch.float8_e8m0fnu))
        
        w_full = torch.empty(out_features, in_features, device='cuda')
        torch.nn.init.normal_(w_full, mean=0.0, std=0.023)
        w_fp4 = w_full[:, :k_fp4].contiguous()
        w_fp8 = w_full[:, k_fp4:in_features].contiguous()
        # Speedtest assumption: weights are pre-quantized offline after the
        # optimizer step. The current OLMo training path quantizes weights
        # online; TODO: add offline weight quantization there.
        self.w_q_fp4, self.w_os_fp4, self.w_is_fp4, w_t_fp4 = tetrajetv2.quant_fp4_dequant_trans_fused(w_fp4)
        self.w_q_fp8, self.w_s_fp8 = tetrajetv2.quant_fp8(w_fp8)
        
        self.w_t = torch.empty(in_features, out_features, device='cuda')
        self.w_t[:k_fp4] = w_t_fp4
        self.w_t[k_fp4:in_features] = tetrajetv2.dequant_fp8(self.w_q_fp8, self.w_s_fp8).T.contiguous()
        
    
    def forward(self, x):
        x = x.float()
        x_shape = x.shape
        if len(x_shape) > 2:
            x = x.view(-1, x_shape[-1])
        y = MixedMatMulFunction.apply(
            x, self.w_q_fp4, self.w_os_fp4, self.w_is_fp4, self.w_t,
            self.w_q_fp8, self.w_s_fp8, self.p
        )
        if len(x_shape) > 2:
            y = y.view(*x_shape[:-1], self.out_features)
        return y
 
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, p, probe_on):
        super().__init__()
        if probe_on:
            self.gate_up_proj = PerformanceProbe(MixedLinear(hidden_size, intermediate_size * 2, p), "Linear")
            self.down_proj = PerformanceProbe(MixedLinear(intermediate_size, hidden_size, p), "Linear")
        else:
            self.gate_up_proj = MixedLinear(hidden_size, intermediate_size * 2, p)
            self.down_proj = MixedLinear(intermediate_size, hidden_size, p)
        self.intermediate_size = intermediate_size
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_up: torch.Tensor = self.gate_up_proj(x)
        gate, up = gate_up.split([self.intermediate_size] * 2, dim=-1)
        x = self.act_fn(gate) * up
        return self.down_proj(x)
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, p, probe_on):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        if probe_on:
            self.qkv_proj = PerformanceProbe(MixedLinear(hidden_size, hidden_size * 3, p), "Linear")
            self.o_proj = PerformanceProbe(MixedLinear(hidden_size, hidden_size, p), "Linear")
        else:
            self.qkv_proj = MixedLinear(hidden_size, hidden_size * 3, p)
            self.o_proj = MixedLinear(hidden_size, hidden_size, p)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        
        qkv: torch.Tensor = self.qkv_proj(hidden_states).view(bsz, q_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class MixedTransformerBlock(nn.Module):
    def __init__(self, hidden_size=4096, num_heads=32, intermediate_size=16384, p=8, probe_on=False):
        super().__init__()
        self.input_layernorm = te.RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, p, probe_on)
        self.post_attention_layernorm = te.RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, p, probe_on)

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
