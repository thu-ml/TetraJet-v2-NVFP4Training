import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from tetrajetv2.performance_probe import PerformanceProbe

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, probe_on):
        super().__init__()
        if probe_on:
            self.gate_up_proj = PerformanceProbe(te.Linear(hidden_size, intermediate_size * 2, bias=False), "Linear")
            self.down_proj = PerformanceProbe(te.Linear(intermediate_size, hidden_size, bias=False), "Linear")
        else:
            self.gate_up_proj = te.Linear(hidden_size, intermediate_size * 2, bias=False)
            self.down_proj = te.Linear(intermediate_size, hidden_size, bias=False)
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
            self.qkv_proj = PerformanceProbe(te.Linear(hidden_size, hidden_size * 3, bias=False), "Linear")
            self.o_proj = PerformanceProbe(te.Linear(hidden_size, hidden_size, bias=False), "Linear")
        else:
            self.qkv_proj = te.Linear(hidden_size, hidden_size * 3, bias=False)
            self.o_proj = te.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        
        qkv: torch.Tensor = self.qkv_proj(hidden_states).view(bsz, q_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class FP8TransformerBlock(nn.Module):
    def __init__(self, hidden_size=4096, num_heads=32, intermediate_size=16384, probe_on=False):
        super().__init__()
        self.input_layernorm = te.RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, probe_on)
        self.post_attention_layernorm = te.RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, probe_on)
        self.fp8_recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID)

    def forward(self, hidden_states):
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states)
            hidden_states = hidden_states + residual

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = hidden_states + residual

        return hidden_states