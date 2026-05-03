import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

try:
    from .nvfp4_quant_fused_kernel \
        import _E2M1_DoubleScale_fake_quantize_stochastic_fused_kernel, \
            _E2M1_DoubleScale_fake_quantize_fused_kernel, \
            _DoubleScale_NVQuant_ScaleOnly_kernel
except:
    from nvfp4_quant_fused_kernel \
        import _E2M1_DoubleScale_fake_quantize_stochastic_fused_kernel, \
            _E2M1_DoubleScale_fake_quantize_fused_kernel, \
            _DoubleScale_NVQuant_ScaleOnly_kernel

@torch.no_grad()
def nvfp4_double_quantize_fused(x: torch.Tensor, 
                                quantize_enabled=True,
                                stochastic=False,
                                truncation_free_scale=False, # TODO,
                                blocksize_inner = 16,
                                blocksize_outer = 128):
    if not x.is_contiguous():
        x = x.contiguous()
    if not quantize_enabled:
        return x
    
    # output: quantize-dequantized x
    assert x.shape[-1] % blocksize_outer == 0, f"the shape of x is {x.shape}, x.shape[-1] % {blocksize_outer} != 0"
    assert truncation_free_scale == False, "TODO"
    
    y = torch.empty_like(x)
    n_elements = x.numel()
    
    if blocksize_outer != -1:
        grid = (triton.cdiv(n_elements, blocksize_outer),)
        if stochastic:
            noise = x.new(x.shape).uniform_(-0.5, 0.5)
            _E2M1_DoubleScale_fake_quantize_stochastic_fused_kernel[grid](
                x, noise, 
                y, 
                n_elements, blocksize_outer, blocksize_inner # type: ignore
            )
        else:
            _E2M1_DoubleScale_fake_quantize_fused_kernel[grid](
                x, 
                y, 
                n_elements, blocksize_outer, blocksize_inner # type: ignore
            )
    else:
        raise NotImplementedError("Per-tensor outer-scaling not implemented yet")

    return y

@torch.no_grad()
def nvfp4_scale_only(x: torch.Tensor, 
                     blocksize_inner = 16,
                     blocksize_outer = 128):
    # output: scale shape_like x
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.shape[-1] % blocksize_outer == 0, \
        f"the shape of x is {x.shape}, x.shape[-1] % {blocksize_outer} != 0"
    
    y = torch.empty_like(x)
    n_elements = x.numel()
    
    if blocksize_outer != -1:
        grid = (triton.cdiv(n_elements, blocksize_outer),)
        _DoubleScale_NVQuant_ScaleOnly_kernel[grid](
            x, 
            y, 
            n_elements, blocksize_outer, blocksize_inner # type: ignore
        )
    else:
        raise NotImplementedError("Per-tensor outer-scaling not implemented yet")

    return y

