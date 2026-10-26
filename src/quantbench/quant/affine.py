"""Uniform Affine and Symmetric AbsMax Quantization Primitives.

Implements the fundamental mathematical mappings between continuous floating-point
representations and discrete integer quantization grids.
"""

from typing import Optional, Tuple, Union

import torch


def quantize_affine(
    x: torch.Tensor,
    num_bits: int = 8,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    signed: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a floating-point tensor using uniform affine (asymmetric) quantization.

    Maps [x_min, x_max] to integer range [q_min, q_max] using:
        scale = (x_max - x_min) / (2^num_bits - 1)
        zero_point = round(-x_min / scale) + q_min
        q = clamp(round(x / scale) + zero_point, q_min, q_max)

    Args:
        x: Input floating-point tensor.
        num_bits: Target integer precision (typically 4 or 8).
        dim: Dimension(s) along which to reduce for per-channel scaling.
             If None, computes a single scalar per-tensor.
        signed: Whether the integer grid is signed (e.g. [-128, 127]) or unsigned ([0, 255]).
        eps: Small epsilon to prevent division by zero for constant tensors.

    Returns:
        Tuple of (quantized_tensor, scale, zero_point).
    """
    if signed:
        q_min = -(1 << (num_bits - 1))
        q_max = (1 << (num_bits - 1)) - 1
    else:
        q_min = 0
        q_max = (1 << num_bits) - 1

    if dim is None:
        x_min = x.min()
        x_max = x.max()
    else:
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)

    # Standard zero-preserving affine quantization
    # (ensures 0.0 maps exactly to an integer zero_point)
    zeros = torch.zeros_like(x_min)
    x_min_adj = torch.minimum(x_min, zeros)
    x_max_adj = torch.maximum(x_max, zeros)

    # Calculate scale factor
    scale = (x_max_adj - x_min_adj) / float(q_max - q_min)
    scale = torch.clamp(scale, min=eps)

    # Calculate zero-point
    zero_point = torch.round(-x_min_adj / scale) + q_min
    zero_point = torch.clamp(zero_point, min=q_min, max=q_max)

    # Quantize and clamp
    q = torch.round(x / scale) + zero_point
    q = torch.clamp(q, min=q_min, max=q_max)

    # Cast to integer representation
    dtype = torch.int8 if signed else torch.uint8
    if num_bits <= 8:
        q = q.to(dtype)
    else:
        q = q.to(torch.int32)

    return q, scale, zero_point


def dequantize_affine(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize an affine-quantized integer tensor back to floating-point.

    Computes:
        x_hat = (q - zero_point) * scale

    Args:
        q: Integer quantized tensor.
        scale: Floating-point scale tensor (broadcastable to q).
        zero_point: Integer or floating-point zero-point tensor.

    Returns:
        Reconstructed floating-point tensor.
    """
    return (q.to(scale.dtype) - zero_point.to(scale.dtype)) * scale


def quantize_absmax(
    x: torch.Tensor,
    num_bits: int = 8,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    symmetric_range: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating-point tensor using symmetric AbsMax quantization (zero_point = 0).

    Maps [-max(|x|), max(|x|)] to signed integer range [-q_max, q_max] using:
        scale = max(|x|) / q_max
        q = clamp(round(x / scale), q_min, q_max)

    Args:
        x: Input floating-point tensor.
        num_bits: Target bit precision (typically 4 or 8).
        dim: Dimension(s) along which to reduce for per-channel scaling.
             If None, computes per-tensor scale.
        symmetric_range: If True, restricts range to [-(2^(b-1)-1), 2^(b-1)-1] to preserve
                         exact symmetry around zero. If False, allows [-2^(b-1), 2^(b-1)-1].
        eps: Epsilon to avoid division by zero.

    Returns:
        Tuple of (quantized_tensor, scale).
    """
    q_max = (1 << (num_bits - 1)) - 1
    q_min = -q_max if symmetric_range else -(1 << (num_bits - 1))

    if dim is None:
        max_val = x.abs().max()
    else:
        max_val = x.abs().amax(dim=dim, keepdim=True)

    scale = max_val / float(q_max)
    scale = torch.clamp(scale, min=eps)

    q = torch.round(x / scale)
    q = torch.clamp(q, min=q_min, max=q_max)

    if num_bits <= 8:
        q = q.to(torch.int8)
    else:
        q = q.to(torch.int32)

    return q, scale


def dequantize_absmax(
    q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize an AbsMax-quantized tensor back to floating-point.

    Computes:
        x_hat = q * scale

    Args:
        q: Integer quantized tensor.
        scale: Floating-point scale tensor (broadcastable to q).

    Returns:
        Reconstructed floating-point tensor.
    """
    return q.to(scale.dtype) * scale


def check_quantization_invariants(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    scale: torch.Tensor,
    tolerance_factor: float = 0.51,
) -> bool:
    """Verify the fundamental quantization invariant: error <= scale / 2.

    For uniform round-to-nearest quantization, the maximum discretization error
    for any element within the clamp range is bounded by scale / 2.

    Args:
        x: Original floating-point tensor.
        x_hat: Dequantized floating-point tensor.
        scale: Scale tensor used for quantization.
        tolerance_factor: Discretization error threshold multiplier (default 0.51).

    Returns:
        True if all elements within dynamic range satisfy the invariant.
    """
    err = (x - x_hat).abs()
    max_allowed_err = scale * tolerance_factor
    return bool(torch.all(err <= max_allowed_err + 1e-6))
