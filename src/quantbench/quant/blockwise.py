"""Group-wise (Block-wise) Quantization Primitives.

Partitions tensors into independent contiguous blocks of size G (e.g. 32, 64, 128)
to isolate channel outliers and preserve precision across wide transformer layers.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from quantbench.quant.affine import (
    dequantize_absmax,
    dequantize_affine,
    quantize_absmax,
    quantize_affine,
)


def quantize_blockwise(
    x: torch.Tensor,
    group_size: int = 128,
    num_bits: int = 4,
    symmetric: bool = True,
    signed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Quantize a tensor into independent group-wise blocks.

    Divides flattened tensor into blocks of size `group_size`. If the tensor element
    count is not an exact multiple of `group_size`, pads with zeros.

    Args:
        x: Input tensor of arbitrary shape.
        group_size: Number of consecutive elements per block (32, 64, 128).
        num_bits: Bit precision per element (e.g. 4 or 8).
        symmetric: If True, uses symmetric AbsMax per block (zero_point = 0).
                   If False, uses asymmetric affine (scale and zero_point).
        signed: Whether integer values are signed (only used when symmetric=False).

    Returns:
        Tuple of (q_blocks, scales, zero_points, metadata_dict).
    """
    orig_shape = tuple(x.shape)
    flat_x = x.contiguous().view(-1)
    n_elements = flat_x.numel()

    # Calculate padding if necessary
    remainder = n_elements % group_size
    pad_len = (group_size - remainder) % group_size
    if pad_len > 0:
        flat_x = torch.nn.functional.pad(flat_x, (0, pad_len), value=0.0)

    # Reshape into blocks: [num_blocks, group_size]
    blocked_x = flat_x.view(-1, group_size)

    if symmetric:
        q_blocks, scales = quantize_absmax(
            blocked_x,
            num_bits=num_bits,
            dim=1,
            symmetric_range=True,
        )
        zero_points = None
    else:
        q_blocks, scales, zero_points = quantize_affine(
            blocked_x,
            num_bits=num_bits,
            dim=1,
            signed=signed,
        )

    meta = {
        "original_shape": orig_shape,
        "n_elements": n_elements,
        "pad_len": pad_len,
        "group_size": group_size,
        "num_bits": num_bits,
        "symmetric": symmetric,
        "dtype": x.dtype,
        "device": x.device,
    }

    return q_blocks, scales, zero_points, meta


def dequantize_blockwise(
    q_blocks: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    meta: Dict[str, Any],
) -> torch.Tensor:
    """Reconstruct an original floating-point tensor from block-wise quantized representations.

    Args:
        q_blocks: Quantized integer blocks of shape [num_blocks, group_size].
        scales: Scale factors of shape [num_blocks, 1].
        zero_points: Optional zero-points of shape [num_blocks, 1].
        meta: Metadata dictionary returned by quantize_blockwise.

    Returns:
        Reconstructed tensor matching original shape and device.
    """
    if meta["symmetric"]:
        x_recon_blocked = dequantize_absmax(q_blocks, scales)
    else:
        assert zero_points is not None, (
            "zero_points required for asymmetric blockwise dequantization"
        )
        x_recon_blocked = dequantize_affine(q_blocks, scales, zero_points)

    flat_recon = x_recon_blocked.view(-1)
    # Remove padding if added during quantization
    if meta["pad_len"] > 0:
        flat_recon = flat_recon[: meta["n_elements"]]

    return flat_recon.view(meta["original_shape"]).to(meta["dtype"])
