from contextlib import contextmanager
import torch
from olmo.config import NVFP4_QuantizerConfig, OscillationPerturb_Config

import tetrajetv2
from .utils import maybe_summon_full_params
from .utils_type import check_quant_linear, check_quant_mix_linear

import torch.distributed as dist
import logging

log = logging.getLogger(__name__)

def _get_bin_width(w_q_fp4val: torch.Tensor):
    FP4_abs_values = w_q_fp4val.new_tensor([0,   0.5, 1,   1.5, 2,    3, 4,   6])
    FP4_bin_widths = w_q_fp4val.new_tensor([0.5, 0.5, 0.5, 0.5, 0.75, 1, 1.5, 2])
    tol = 1e-1
    
    index = torch.zeros_like(w_q_fp4val, dtype=torch.int)
    w_q_fp4val = w_q_fp4val.abs()
    
    for i, v in enumerate(FP4_abs_values):
        index[torch.isclose(w_q_fp4val, v, atol=tol)] = i
    
    return FP4_bin_widths[index]

def fp4_elements_centralize_BinEdge(model, centralize_thrd=0.495):
    sum_weight_numel = 0
    sum_osci_numel = 0
    
    with torch.no_grad():
        for _, module in model.named_modules():
            if check_quant_linear(module):
                if check_quant_mix_linear(module) and module.mix_channel_done:
                    channel_idx = module.fp4_channel_idx
                    w = module.weight.detach()[:, channel_idx]
                else:
                    channel_idx = None
                    w = module.weight.detach()
                
                w_q, w_os, w_is = tetrajetv2.quant_fp4(w, 0)
                w_qdq = tetrajetv2.dequant_fp4(w_q, w_os, w_is)
                w_scale_fp = tetrajetv2.quant_fp4_scale_only(w)
                
                w_lat = w / w_scale_fp
                w_fp4val = w_qdq / w_scale_fp
                w_binwidth = _get_bin_width(w_fp4val)
                
                reset_mask = (w_lat - w_fp4val).abs() > w_binwidth * centralize_thrd
                sum_osci_numel += reset_mask.sum().item()
                sum_weight_numel += reset_mask.numel()
                
                w[reset_mask] = w_qdq[reset_mask]
                
                if channel_idx is not None:
                    module.weight.data.index_copy_(1, channel_idx.to(torch.long), w)
                else:
                    module.weight.copy_(w)
                   
    if sum_weight_numel != 0:
        log.info(f"[BinEdge] selected {sum_osci_numel / sum_weight_numel * 100:.3f}% oscillating elements")
    else:
        log.info(f"[BinEdge] No QLinear elements")


_last_linear_weights = {}
_distance_Wfp = {}
_distance_Wqdq = {}
_mix_channel_done = {}

def check_mix_channel_done(name):
    return _mix_channel_done[name]

def query_quant_risk_ratio(name):
    ret = _distance_Wqdq[name] / _distance_Wfp[name]
    ret[_distance_Wfp[name] == 0] = 0
    return ret

def quant_weight_dist_init_snapshots(model):
    log.info(f'Initializing snapshots.')
    with torch.no_grad():
        for name, module in model.named_modules():
            if check_quant_linear(module):
                if check_quant_mix_linear(module) and module.mix_channel_done:
                    _mix_channel_done[name] = True
                    channel_idx = module.fp4_channel_idx
                    w = module.weight.detach()[:, channel_idx]
                else:
                    _mix_channel_done[name] = False
                    w = module.weight.detach()
                
                _last_linear_weights[name] = w.clone()
                _distance_Wqdq[name] = torch.zeros_like(w)
                _distance_Wfp[name]  = torch.zeros_like(w)

def quant_weight_dist_track(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if check_quant_linear(module):
                if check_quant_mix_linear(module) and check_mix_channel_done(name):
                    channel_idx = module.fp4_channel_idx
                    w = module.weight.detach()[:, channel_idx]
                else:
                    w = module.weight.detach()
                    
                w_prev = _last_linear_weights[name]
                w_qdq = tetrajetv2.dequant_fp4(*tetrajetv2.quant_fp4(w, 0))
                w_prev_qdq = tetrajetv2.dequant_fp4(*tetrajetv2.quant_fp4(w_prev, 0))
                
                _distance_Wqdq[name] += (w_qdq - w_prev_qdq).abs()
                _distance_Wfp[name]  += (w - w_prev).abs()
                _last_linear_weights[name] = w.clone()

def quant_weight_dist_clear_snapshots():
    global _last_linear_weights
    global _distance_Wfp
    global _distance_Wqdq
    
    for _, v in _last_linear_weights.items(): del v
    for _, v in _distance_Wqdq.items(): del v
    for _, v in _distance_Wfp.items(): del v
    
    _last_linear_weights.clear()
    _distance_Wqdq.clear()
    _distance_Wfp.clear()
    _mix_channel_done.clear()
    
    torch.cuda.empty_cache()
    
def fp4_elements_centralize_DistRatio(model, osci_ratio_thrd: float =16.):
    sum_weight_numel = 0
    sum_osci_numel = 0
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if check_quant_linear(module):
                if check_quant_mix_linear(module) and check_mix_channel_done(name):
                    channel_idx = module.fp4_channel_idx
                    w = module.weight.detach()[:, channel_idx]
                else:
                    channel_idx = None
                    w = module.weight
                
                w_q, w_os, w_is = tetrajetv2.quant_fp4(w, 0)
                w_qdq = tetrajetv2.dequant_fp4(w_q, w_os, w_is)
                
                reset_mask = query_quant_risk_ratio(name) >= osci_ratio_thrd
                sum_osci_numel += reset_mask.sum().item()
                sum_weight_numel += reset_mask.numel()
                
                w[reset_mask] = w_qdq[reset_mask]
                
                if channel_idx is not None:
                    module.weight.data.index_copy_(1, channel_idx.to(torch.long), w)
                else:
                    module.weight.copy_(w)
                   
    if sum_weight_numel != 0:
        log.info(f"[DistRatio] Selected {sum_osci_numel / sum_weight_numel * 100:.3f}% oscillating elements")
    else:
        log.info(f"No QLinear elements")
