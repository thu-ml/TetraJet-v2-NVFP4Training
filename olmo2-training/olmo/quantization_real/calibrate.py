import torch

from olmo.config import NVFP4_QuantizerConfig, NVFP4_Outlier_Selection_Config

from .oscillation_reset import *
from .oscillation_reset_memeff import *

from .utils_type import *
from .utils import *

def linear_mixprec_record(qconfig: NVFP4_QuantizerConfig, iter_num, model):
    if (not qconfig.enable_nvfp4_training
        or 1 not in qconfig.enabled_quantizers
        or not qconfig.outlier_conf.X_chan_select):
        return

    outlier_conf: NVFP4_Outlier_Selection_Config = qconfig.outlier_conf
    
    if outlier_conf.X_chan_select_mode not in ['norm2_record']:
        raise NotImplementedError(f"Not implemented X_chan_select_mode: '{outlier_conf.X_chan_select_mode}'")
    
    if iter_num == outlier_conf.X_chan_record_start_iter:
        with maybe_summon_full_params(model):
            with torch.no_grad():
                for _, module in model.named_modules():
                    if check_quant_mix_linear(module):
                        module.start_channel_outlier_accumulate()
        
    if iter_num == outlier_conf.X_chan_record_start_iter + outlier_conf.X_chan_record_iters:
        with maybe_summon_full_params(model):
            with torch.no_grad():
                for _, module in model.named_modules():
                    if check_quant_mix_linear(module):
                        module.sync_channel_accumulate_and_update_idx()

def linear_oscillation_reset_with_mix(qconfig: NVFP4_QuantizerConfig, iter_num, model):
    if (qconfig is None) or \
        (qconfig.enabled_quantizers is None) or \
        (2 not in qconfig.enabled_quantizers):
        return
    if not isinstance(qconfig.osci_perturb_setting, OscillationPerturb_Config) or \
        not qconfig.osci_perturb_setting.apply_perturb:
        return
    
    osci_conf: OscillationPerturb_Config = qconfig.osci_perturb_setting
    
    if osci_conf.detect_type == 'bin_edge':
        if iter_num % osci_conf.perturb_period == 0 and iter_num >= osci_conf.perturb_start_iter:
            
            log.info(f"[step #{iter_num}] doing RealQ-BinEdge-OsciReset with thrd={osci_conf.detect_BinEdge_thrd}")
            with maybe_summon_full_params(model, writeback=True):
                fp4_elements_centralize_BinEdge(model=model, 
                                                centralize_thrd=osci_conf.detect_BinEdge_thrd)
        
    elif osci_conf.detect_type == 'distance_ratio':
        
        # [TODO] Better Oscillation Detection Method
        if iter_num >= osci_conf.perturb_start_iter:
            if iter_num % osci_conf.perturb_period == 0:  # (1) init
                with maybe_summon_full_params(model):
                    log.info(f"[step #{iter_num}] Start RealQ-DistRatio-OsciReduce Initialize")
                    quant_weight_dist_init_snapshots(model)
            
            elif iter_num % osci_conf.perturb_period <= osci_conf.detect_DistRatio_iters: # (2) track dist
                with maybe_summon_full_params(model):
                    quant_weight_dist_track(model)
                
            elif iter_num % osci_conf.perturb_period == osci_conf.detect_DistRatio_iters + 1: # (3) centralize
                with maybe_summon_full_params(model, writeback=True):
                    log.info(f"[step #{iter_num}] doing RealQ-DistRatio-OsciReduce with thrd={osci_conf.detect_DistRatio_thrd}")
            
                    fp4_elements_centralize_DistRatio(model=model,
                                                      osci_ratio_thrd=osci_conf.detect_DistRatio_thrd)
                    quant_weight_dist_clear_snapshots()
    
    elif osci_conf.detect_type == 'distance_ratio_memeff':
        if iter_num % osci_conf.perturb_period == 0:  # (1) init
            sample_percent = getattr(osci_conf, "detect_DistRatio_sample_percent", 10.0)
            with maybe_summon_full_params(model):
                log.info(f"[step #{iter_num}] Start RealQ-DistRatio-MemEff-OsciReduce Initialize"
                         f"(sample_percent={sample_percent})")
                quant_weight_dist_init_snapshots_memeff(model, sample_percent=sample_percent)
        
        elif iter_num % osci_conf.perturb_period <= osci_conf.detect_DistRatio_iters: # (2) track dist
            with maybe_summon_full_params(model):
                quant_weight_dist_track_memeff(model)
            
        elif iter_num % osci_conf.perturb_period == osci_conf.detect_DistRatio_iters + 1: # (3) centralize
            with maybe_summon_full_params(model, writeback=True):
                log.info(f"[step #{iter_num}] doing RealQ-DistRatio-MemEff-OsciReduce "
                         f"with thrd={osci_conf.detect_DistRatio_thrd}")
        
                fp4_elements_centralize_DistRatio_memeff(model=model,
                                                    osci_ratio_thrd=osci_conf.detect_DistRatio_thrd)
                quant_weight_dist_clear_snapshots_memeff()
            
    else:
        raise NotImplementedError(f'not implemented osci_conf.detect_type: {osci_conf.detect_type}')

