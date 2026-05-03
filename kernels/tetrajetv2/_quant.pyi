from torch import Tensor

# quant
def quant_fp4(x: Tensor, seed: int) -> tuple[Tensor, Tensor, Tensor]: ...

def quant_fp8(x: Tensor) -> tuple[Tensor, Tensor]: ...

def quant_fp4_dequant_trans_rh_requant_fused(
    x: Tensor, seed: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

def quant_fp8_dequant_trans_requant_fused(
    x: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

def quant_fp4_dequant_trans_fused(
    x: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

def rh_quant_fp4_trans_rh_requant_fused(
    x: Tensor,
    seed_dx: int,
    seed_dw: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

def rh_quant_fp4_trans_rh_requant_fp4_fp8_fused(
    x: Tensor,
    seed_dx: int,
    seed_dw: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

# dequant
def dequant_fp4(a: Tensor, out_sfa: Tensor, sfa: Tensor) -> Tensor: ...

def dequant_fp4_rh_requant_fused(
    a: Tensor, out_sfa: Tensor, sfa: Tensor, seed: int
) -> tuple[Tensor, Tensor, Tensor]: ...

def dequant_fp8(a: Tensor, scale: Tensor) -> Tensor: ...