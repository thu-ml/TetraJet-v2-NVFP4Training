from torch import Tensor

def fp4_gemm(
    a: Tensor, b: Tensor,
    sfa: Tensor, sfb: Tensor,
    out_sfa: Tensor, out_sfb: Tensor
) -> Tensor: ...

def fp8_gemm(
    a: Tensor, b: Tensor,
    sfa: Tensor, sfb: Tensor
) -> Tensor: ...

def fp8_gemm_accum(
    a: Tensor, b: Tensor,
    sfa: Tensor, sfb: Tensor
) -> Tensor: ...