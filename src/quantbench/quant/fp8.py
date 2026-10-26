"""OCP FP8 (E4M3FN and E5M2) Emulation and Simulation Primitives.

Implements bit-exact simulation of the two standardized 8-bit floating-point formats:
1. E4M3FN: 1 sign, 4 exponent, 3 mantissa (bias 7, max ±448.0, no infinity, higher precision).
2. E5M2: 1 sign, 5 exponent, 2 mantissa (bias 15, max ±57344.0, IEEE inf/nan, wider dynamic range).
"""

from enum import Enum
from typing import Optional, Tuple

import torch


class FP8Format(str, Enum):
    """Standardized Open Compute Project (OCP) 8-bit floating point formats."""

    E4M3 = "e4m3fn"
    E5M2 = "e5m2"


FP8_MAX_VALUES = {
    FP8Format.E4M3: 448.0,
    FP8Format.E5M2: 57344.0,
}


def _emulate_fp8(
    x: torch.Tensor,
    exp_bits: int,
    mantissa_bits: int,
    bias: int,
    max_val: float,
    has_inf: bool = False,
) -> torch.Tensor:
    """Analytical bit-level emulator for floating-point formats with arbitrary bitwidths.

    Simulates the exact IEEE / OCP quantization, subnormal behavior, and saturation.
    """
    sign = torch.sign(x)
    abs_x = torch.abs(x)

    # Saturate / clamp to max representable value
    abs_x = torch.clamp(abs_x, max=max_val)

    # Handle zeros
    is_zero = abs_x == 0.0

    # Calculate base-2 exponent
    # e = floor(log2(x))
    log2_x = torch.log2(torch.clamp(abs_x, min=1e-30))
    exp = torch.floor(log2_x)

    # Minimum normal exponent
    min_exp = 1 - bias
    max_exp = (1 << exp_bits) - 1 - bias

    # Separate normals from subnormals
    is_subnormal = exp < min_exp
    exp = torch.clamp(exp, min=min_exp, max=max_exp)

    # Mantissa quantization step size
    # For normals: step = 2^(exp - mantissa_bits)
    # For subnormals: step = 2^(min_exp - mantissa_bits)
    eff_exp = torch.where(is_subnormal, torch.full_like(exp, min_exp), exp)
    step = torch.pow(2.0, eff_exp - mantissa_bits)

    # Discretize mantissa via round-to-nearest
    q_abs = torch.round(abs_x / step) * step

    # Re-apply clamp
    q_abs = torch.clamp(q_abs, max=max_val)

    # Zero out true zeros
    q_abs = torch.where(is_zero, torch.zeros_like(q_abs), q_abs)

    return sign * q_abs


def quantize_fp8(
    x: torch.Tensor,
    format: FP8Format = FP8Format.E4M3,
    scale: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating-point tensor to FP8 with dynamic or user-specified scaling.

    Computes:
        scale = max(|x|) / max_representable_fp8 (if scale is None)
        scaled_x = x / scale
        q_fp8 = quantize(scaled_x)

    Args:
        x: Input tensor.
        format: FP8Format.E4M3 or FP8Format.E5M2.
        scale: Optional precomputed scale factor tensor.
        eps: Small value to avoid division by zero.

    Returns:
        Tuple of (quantized_fp8_tensor, scale).
    """
    max_repr = FP8_MAX_VALUES[format]

    if scale is None:
        max_val = x.abs().max()
        scale = max_val / float(max_repr)
        scale = torch.clamp(scale, min=eps)

    scaled_x = x / scale

    # Check if native torch float8 dtype is supported and available
    native_dtype = None
    if hasattr(torch, "float8_e4m3fn") and format == FP8Format.E4M3:
        native_dtype = torch.float8_e4m3fn
    elif hasattr(torch, "float8_e5m2") and format == FP8Format.E5M2:
        native_dtype = torch.float8_e5m2

    if native_dtype is not None:
        try:
            # Native PyTorch conversion
            q = scaled_x.to(native_dtype)
            return q, scale
        except Exception:
            pass

    # Analytical fallback emulation
    if format == FP8Format.E4M3:
        q = _emulate_fp8(
            scaled_x,
            exp_bits=4,
            mantissa_bits=3,
            bias=7,
            max_val=448.0,
            has_inf=False,
        )
    else:
        q = _emulate_fp8(
            scaled_x,
            exp_bits=5,
            mantissa_bits=2,
            bias=15,
            max_val=57344.0,
            has_inf=True,
        )

    return q, scale


def dequantize_fp8(
    q_fp8: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to target floating-point precision.

    Computes:
        x_hat = q_fp8 * scale

    Args:
        q_fp8: Quantized FP8 tensor.
        scale: Scale factor tensor.
        target_dtype: Desired output floating-point precision (e.g. float16, float32).

    Returns:
        Dequantized tensor.
    """
    return q_fp8.to(target_dtype) * scale.to(target_dtype)
