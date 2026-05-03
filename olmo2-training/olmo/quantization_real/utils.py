import torch
import torch.nn.functional as F
from contextlib import contextmanager
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F

def cos_sim(X: torch.Tensor, Y: torch.Tensor):
    X = X.contiguous().flatten().to(torch.float64)
    Y = Y.contiguous().flatten().to(torch.float64)
    return (X * Y).sum() / (X.norm(dtype=torch.float64) * Y.norm(dtype=torch.float64))

@contextmanager
def maybe_summon_full_params(model, writeback=False):
    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, writeback=writeback):
            yield
    else:
        yield

def format_string_with_condition(input_string, shape, condition, input_pad=15, ):
    padded_string = input_string.ljust(input_pad)
    if condition:
        output_string = padded_string + "Quant Enabled".ljust(15)  + f"{shape}".ljust(20)
    else:
        output_string = padded_string + "Quant Disabled".ljust(15) + f"{shape}".ljust(20)

    return output_string

def validate_and_pad(x, w, alignment=128):
    """
    Checks dimensions and pads the token dimension of x to a multiple of alignment.
    Args:
        x: Input tensor of shape [..., in_features]
        w: Weight tensor of shape [out_features, in_features]
        alignment: Required multiple for dimensions
    Returns:
        x_padded: Padded 2D tensor [N_padded, in_features]
        pad_len: Number of padded tokens
        original_shape: Original shape of x
    """
    out_features, in_features = w.shape
    original_shape = x.shape
    
    # 1. Dimension Check
    if original_shape[-1] != in_features:
        raise ValueError(f"Dimension mismatch: x.shape[-1] ({original_shape[-1]}) "
                         f"must match w.shape[1] ({in_features})")
    
    if out_features % alignment != 0 or in_features % alignment != 0:
        raise ValueError(f"Weight dimensions ({out_features}, {in_features}) "
                         f"must be multiples of {alignment}")

    # 2. Reshape and Padding
    x_reshaped = x.view(-1, in_features)
    num_tokens = x_reshaped.shape[0]
    
    pad_len = (alignment - num_tokens % alignment) % alignment
    if pad_len > 0:
        # Pad only the token dimension (dim 0)
        x_padded = F.pad(x_reshaped, (0, 0, 0, pad_len))
    else:
        x_padded = x_reshaped
        
    return x_padded, pad_len, original_shape

def unpad_and_reshape(tensor, pad_len, original_shape, out_features=None):
    """
    Removes padding and restores the original batch/spatial dimensions.
    """
    # Remove padding from the token dimension
    if pad_len > 0:
        tensor = tensor[:-pad_len, :]
    
    # Determine the new shape
    # If out_features is provided, we are processing the output Y
    # If None, we are processing DX (which matches original_shape)
    if out_features is not None:
        new_shape = list(original_shape[:-1]) + [out_features]
    else:
        new_shape = original_shape
        
    return tensor.view(*new_shape)

def pad_gradient(dy, pad_len):
    """
    Pads the gradient dy to match the padded token count used in forward.
    """
    dy_reshaped = dy.view(-1, dy.shape[-1])
    if pad_len > 0:
        return F.pad(dy_reshaped, (0, 0, 0, pad_len))
    return dy_reshaped