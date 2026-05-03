import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _E2M1_DoubleScale_fake_quantize_fused_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCKSIZE_outer: tl.constexpr,
    BLOCKSIZE_inner: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCKSIZE_outer
    offsets = block_start + tl.arange(0, BLOCKSIZE_outer)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    
    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    
    #### Double Scale #####
    # first scale: per-block (1x128) FP32
    scale_global = tl.max(x_abs) / 448 / 6
    scale_global = tl.where(scale_global == 0.0, 1, scale_global)
    # second scale: per-block (1x16) uE4M3 (nvfp4)
    x_abs_blocks = x_abs.reshape(BLOCKSIZE_outer // BLOCKSIZE_inner, BLOCKSIZE_inner)
    scale_blocks = tl.max(x_abs_blocks, axis=1, keep_dims=True) / scale_global / 6
    scale_blocks = scale_blocks.to(tl.float8e4nv).to(tl.float32)
    
    scale_product = scale_blocks * scale_global
    x_abs = tl.where(scale_product == 0, 0., x_abs_blocks / scale_product)
    
    ####  Quantize: Exponent-2bit   #####
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=0, max=2)

    ####  Quantize: Mantissa-1bit  #####
    mant      = x_abs / tl.exp2(expo)
    mant_int  = tl.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * 2.0
    # mant_frac = mant_frac + noise
    mant_frac = libdevice.rint(mant_frac)

    #### Rescale ####
    mant_q = mant_int + mant_frac / 2.0
    
    y          = tl.exp2(expo) * mant_q
    y_rescaled = tl.clamp(y, min=0, max=6) * scale_product
    y_return   = tl.ravel(y_rescaled) * sign
    
    tl.store(y_ptr + offsets, y_return.to(x_ptr.dtype.element_ty), mask=mask)

@triton.jit
def _E2M1_DoubleScale_fake_quantize_stochastic_fused_kernel(
    x_ptr, noise_ptr, y_ptr,
    n_elements,
    BLOCKSIZE_outer: tl.constexpr,
    BLOCKSIZE_inner: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCKSIZE_outer
    offsets = block_start + tl.arange(0, BLOCKSIZE_outer)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    
    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    
    noise = tl.load(noise_ptr + offsets, mask=mask, other=0.0) \
              .reshape(BLOCKSIZE_outer // BLOCKSIZE_inner, BLOCKSIZE_inner)
    
    #### Double Scale #####
    # first scale: per-block (1x128) FP32
    scale_global = tl.max(x_abs) / 448 / 6
    scale_global = tl.where(scale_global == 0.0, 1, scale_global)
    # second scale: per-block (1x16) uE4M3 (nvfp4)
    x_abs_blocks = x_abs.reshape(BLOCKSIZE_outer // BLOCKSIZE_inner, BLOCKSIZE_inner)
    scale_blocks = tl.max(x_abs_blocks, axis=1, keep_dims=True) / scale_global / 6
    scale_blocks = scale_blocks.to(tl.float8e4nv).to(tl.float32)
    # avoid "zero", 2**(-9) is the min-subnormal of E4M3
    scale_blocks = tl.where(scale_blocks == 0.0, 2 ** (-9), scale_blocks)

    scale_product = scale_blocks * scale_global
    x_abs = tl.where(scale_product == 0, 0., x_abs_blocks / scale_product)
    
    ####  Quantize: Exponent-2bit   #####
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=0, max=2)

    ####  Quantize: Mantissa-1bit  #####
    mant      = x_abs / tl.exp2(expo)
    mant_int  = tl.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * 2.0
    
    mant_frac = mant_frac + noise
    
    mant_frac = libdevice.rint(mant_frac)

    #### Rescale ####
    mant_q = mant_int + mant_frac / 2.0
    
    y          = tl.exp2(expo) * mant_q
    y_rescaled = tl.clamp(y, min=0, max=6) * scale_product
    y_return   = tl.ravel(y_rescaled) * sign
    
    tl.store(y_ptr + offsets, y_return.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _DoubleScale_NVQuant_ScaleOnly_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCKSIZE_outer: tl.constexpr,
    BLOCKSIZE_inner: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCKSIZE_outer
    offsets = block_start + tl.arange(0, BLOCKSIZE_outer)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    
    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    
    #### Double Scale #####
    # first scale: per-block (1x128) FP32
    scale_global = tl.max(x_abs) / 448 / 6
    scale_global = tl.where(scale_global == 0.0, 1, scale_global)
    # second scale: per-block (1x16) uE4M3 (nvfp4)
    x_abs_blocks = x_abs.reshape(BLOCKSIZE_outer // BLOCKSIZE_inner, BLOCKSIZE_inner)
    scale_blocks = tl.max(x_abs_blocks, axis=1, keep_dims=True) / scale_global / 6
    scale_blocks = scale_blocks.to(tl.float8e4nv).to(tl.float32)
    # avoid "zero", 2**(-9) is the min-subnormal of E4M3
    scale_blocks = tl.where(scale_blocks == 0.0, 2 ** (-9), scale_blocks)

    scale_product = scale_blocks * scale_global
    scale_per_element = tl.broadcast_to(
        scale_product,
        (BLOCKSIZE_outer // BLOCKSIZE_inner, BLOCKSIZE_inner)
    )
    
    scale_per_element = tl.ravel(scale_per_element)
    tl.store(y_ptr + offsets, scale_per_element, mask=mask)
