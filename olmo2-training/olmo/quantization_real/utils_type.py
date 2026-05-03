from .linear import NVFP4_Linear_RealQuant_WeightQuantOnline
from .linear_mix import NVFP4_mix_MXFP8_Linear_RealQuant_WeightQuantOnline
from olmo.config import NVFP4_QuantizerConfig

def check_quant_linear(module):
    quant_linear_modules = [NVFP4_Linear_RealQuant_WeightQuantOnline,
                            NVFP4_mix_MXFP8_Linear_RealQuant_WeightQuantOnline]
    for valid_mod in quant_linear_modules:
        if isinstance(module, valid_mod):
            return True
    return False

def check_quant_mix_linear(module):
    quant_mix_linear_modules = [NVFP4_mix_MXFP8_Linear_RealQuant_WeightQuantOnline]
    for valid_mod in quant_mix_linear_modules:
        if isinstance(module, valid_mod):
            return True
    return False