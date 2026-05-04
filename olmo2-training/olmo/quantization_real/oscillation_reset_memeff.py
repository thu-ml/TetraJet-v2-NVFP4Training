import logging
import math

import torch

import tetrajetv2
from .utils_type import check_quant_linear, check_quant_mix_linear

log = logging.getLogger(__name__)


def _get_bin_width(w_q_fp4val: torch.Tensor):
    FP4_abs_values = w_q_fp4val.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    FP4_bin_widths = w_q_fp4val.new_tensor([0.5, 0.5, 0.5, 0.5, 0.75, 1, 1.5, 2])
    tol = 1e-1

    index = torch.zeros_like(w_q_fp4val, dtype=torch.int)
    w_q_fp4val = w_q_fp4val.abs()

    for i, v in enumerate(FP4_abs_values):
        index[torch.isclose(w_q_fp4val, v, atol=tol)] = i

    return FP4_bin_widths[index]


def _pick_track_indices_by_bin_center_distance(w: torch.Tensor, sample_percent: float) -> torch.Tensor:
    numel = w.numel()
    if numel == 0:
        return torch.empty(0, dtype=torch.long, device=w.device)

    sample_percent = float(sample_percent)
    if sample_percent <= 0:
        return torch.empty(0, dtype=torch.long, device=w.device)
    if sample_percent >= 100:
        return torch.arange(numel, device=w.device, dtype=torch.long)

    w_q, w_os, w_is = tetrajetv2.quant_fp4(w, 0)
    w_qdq = tetrajetv2.dequant_fp4(w_q, w_os, w_is)
    w_scale_fp = tetrajetv2.quant_fp4_scale_only(w)

    w_lat = w / w_scale_fp
    w_fp4val = w_qdq / w_scale_fp
    w_binwidth = _get_bin_width(w_fp4val)
    score = (w_lat - w_fp4val).abs() / w_binwidth
    score = score.reshape(-1)

    k = min(numel, max(1, int(math.ceil(numel * sample_percent / 100.0))))
    return torch.topk(score, k=k, largest=True, sorted=False).indices.to(torch.long)


_last_linear_weights = {}
_last_linear_weights_qdq = {}
_distance_Wfp = {}
_distance_Wqdq = {}
_track_indices = {}
_mix_channel_done = {}


def check_mix_channel_done_memeff(name):
    return _mix_channel_done[name]


def query_quant_risk_ratio_memeff(name):
    ret = _distance_Wqdq[name] / _distance_Wfp[name]
    ret[_distance_Wfp[name] == 0] = 0
    return ret


def query_track_indices_memeff(name):
    return _track_indices[name]


def quant_weight_dist_init_snapshots_memeff(model, sample_percent: float = 10.0):
    log.info(f"Initializing mem-efficient snapshots with sample_percent={sample_percent:.2f}%.")
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

                idx = _pick_track_indices_by_bin_center_distance(w, sample_percent=sample_percent)
                w_flat = w.reshape(-1)
                w_snap = w_flat[idx].clone()
                w_qdq = tetrajetv2.dequant_fp4(*tetrajetv2.quant_fp4(w, 0))
                w_qdq_snap = w_qdq.reshape(-1)[idx].clone()

                _track_indices[name] = idx
                _last_linear_weights[name] = w_snap
                _last_linear_weights_qdq[name] = w_qdq_snap
                _distance_Wqdq[name] = torch.zeros_like(w_snap)
                _distance_Wfp[name] = torch.zeros_like(w_snap)


def quant_weight_dist_track_memeff(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if check_quant_linear(module):
                if check_quant_mix_linear(module) and check_mix_channel_done_memeff(name):
                    channel_idx = module.fp4_channel_idx
                    w = module.weight.detach()[:, channel_idx]
                else:
                    w = module.weight.detach()

                idx = _track_indices[name]
                w_cur = w.reshape(-1)[idx]
                w_prev = _last_linear_weights[name]
                w_prev_qdq = _last_linear_weights_qdq[name]
                w_qdq_full = tetrajetv2.dequant_fp4(*tetrajetv2.quant_fp4(w, 0))
                w_qdq = w_qdq_full.reshape(-1)[idx]

                _distance_Wqdq[name] += (w_qdq - w_prev_qdq).abs()
                _distance_Wfp[name] += (w_cur - w_prev).abs()
                _last_linear_weights[name] = w_cur.clone()
                _last_linear_weights_qdq[name] = w_qdq.clone()


def quant_weight_dist_clear_snapshots_memeff():
    global _last_linear_weights
    global _last_linear_weights_qdq
    global _distance_Wfp
    global _distance_Wqdq
    global _track_indices

    for _, v in _last_linear_weights.items():
        del v
    for _, v in _last_linear_weights_qdq.items():
        del v
    for _, v in _distance_Wqdq.items():
        del v
    for _, v in _distance_Wfp.items():
        del v
    for _, v in _track_indices.items():
        del v

    _last_linear_weights.clear()
    _last_linear_weights_qdq.clear()
    _distance_Wqdq.clear()
    _distance_Wfp.clear()
    _track_indices.clear()
    _mix_channel_done.clear()

    torch.cuda.empty_cache()


def fp4_elements_centralize_DistRatio_memeff(model, osci_ratio_thrd: float = 16.0):
    sum_tracked_numel = 0
    sum_osci_numel = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            if check_quant_linear(module):
                if check_quant_mix_linear(module) and check_mix_channel_done_memeff(name):
                    channel_idx = module.fp4_channel_idx
                    w = module.weight.detach()[:, channel_idx]
                else:
                    channel_idx = None
                    w = module.weight.detach()

                idx = _track_indices[name]
                if idx.numel() == 0:
                    continue

                w_q, w_os, w_is = tetrajetv2.quant_fp4(w, 0)
                w_qdq = tetrajetv2.dequant_fp4(w_q, w_os, w_is)

                ratio = query_quant_risk_ratio_memeff(name)
                reset_mask = ratio >= osci_ratio_thrd

                sum_osci_numel += reset_mask.sum().item()
                sum_tracked_numel += reset_mask.numel()

                reset_idx = idx[reset_mask]
                if reset_idx.numel() > 0:
                    w_flat = w.reshape(-1)
                    w_qdq_flat = w_qdq.reshape(-1)
                    w_flat[reset_idx] = w_qdq_flat[reset_idx]

                if channel_idx is not None:
                    module.weight.data.index_copy_(1, channel_idx.to(torch.long), w)
                else:
                    module.weight.copy_(w)

    if sum_tracked_numel != 0:
        log.info(
            f"[DistRatio-MemEff] Selected {sum_osci_numel / sum_tracked_numel * 100:.3f}% "
            f"oscillating elements among tracked subset"
        )
    else:
        log.info("[DistRatio-MemEff] No tracked QLinear elements")
