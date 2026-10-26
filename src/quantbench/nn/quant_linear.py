"""QuantLinear: Drop-in PyTorch Linear layer with sub-byte packed weights.

Replaces standard torch.nn.Linear by storing weights as packed integer buffers:
- INT4: Vectorized uint8 packed nibbles (2 weights/byte) + blockwise/channel scales.
- BitNet Ternary: 2-bit offset-binary packed uint8 buffers (4 weights/byte) + scale.
- 2:4 Sparsity: 50% condensed float values + 2-bit position metadata nibbles.
- INT8: Signed 8-bit integers + per-channel or per-tensor scales.

Weights are registered as persistent buffers rather than trainable nn.Parameters,
eliminating optimizer memory overhead and enabling direct serialization into safetensors.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantbench.quant.affine import (
    dequantize_absmax,
    dequantize_affine,
    quantize_absmax,
    quantize_affine,
)
from quantbench.quant.blockwise import dequantize_blockwise, quantize_blockwise
from quantbench.quant.packing import (
    decompress_2_4,
    pack_int4,
    pack_ternary,
    prune_and_compress_2_4,
    unpack_int4,
    unpack_ternary,
)


class QuantLinear(nn.Module):
    """Drop-in quantized linear layer storing packed weights in memory buffers.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        scheme: Quantization scheme: 'int4', 'int8', 'ternary', 'sparse24'.
        group_size: Group size for blockwise quantization (-1 for per-channel).
        bias: Whether to include an additive bias parameter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scheme: str = "int4",
        group_size: int = 64,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scheme = scheme.lower()
        self.group_size = group_size

        if self.scheme not in ("int4", "int8", "ternary", "sparse24"):
            raise ValueError(
                f"Unknown scheme '{scheme}'. Supported: 'int4', 'int8', 'ternary', 'sparse24'."
            )

        # Register bias as parameter if present
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Buffers are allocated upon calling from_linear or loading state_dict
        self.register_buffer("packed_weight", None)
        self.register_buffer("scales", None)
        self.register_buffer("zero_points", None)
        self.register_buffer("condensed_weight", None)
        self.register_buffer("packed_metadata", None)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Register incoming serialized buffer tensors dynamically during load_state_dict."""
        for buf_name in (
            "packed_weight",
            "scales",
            "zero_points",
            "condensed_weight",
            "packed_metadata",
        ):
            full_key = prefix + buf_name
            if full_key in state_dict and state_dict[full_key] is not None:
                self.register_buffer(buf_name, state_dict[full_key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        scheme: str = "int4",
        group_size: int = 64,
        symmetric: bool = True,
    ) -> QuantLinear:
        """Create a QuantLinear module from an unquantized torch.nn.Linear layer.

        Args:
            linear: Source torch.nn.Linear layer.
            scheme: Target quantization scheme: 'int4', 'int8', 'ternary', 'sparse24'.
            group_size: Group size for blockwise quantization (e.g. 32, 64, 128, or -1).
            symmetric: Whether to use symmetric quantization (zero-point = 0).

        Returns:
            QuantLinear instance with compressed weight buffers.
        """
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        quant_layer = cls(
            in_features=in_features,
            out_features=out_features,
            scheme=scheme,
            group_size=group_size,
            bias=has_bias,
        )

        w = linear.weight.data.clone().to(torch.float32)
        if has_bias:
            quant_layer.bias.data.copy_(linear.bias.data)

        scheme_lower = scheme.lower()

        if scheme_lower == "int4":
            if group_size > 0:
                # Group-wise INT4
                q_blocks, s, zp, _ = quantize_blockwise(
                    w, group_size=group_size, num_bits=4, symmetric=symmetric
                )
                # Pack each block along dim=-1
                packed_w, _ = pack_int4(q_blocks, signed=symmetric, dim=-1)
                quant_layer.register_buffer("packed_weight", packed_w)
                quant_layer.register_buffer("scales", s)
                if zp is not None:
                    quant_layer.register_buffer("zero_points", zp)
            else:
                # Per-channel INT4 along output channels (reduce across input dim 1)
                if symmetric:
                    q_w, s = quantize_absmax(w, num_bits=4, dim=1)
                    zp = None
                else:
                    q_w, s, zp = quantize_affine(w, num_bits=4, dim=1)
                packed_w, _ = pack_int4(q_w, signed=symmetric, dim=-1)
                quant_layer.register_buffer("packed_weight", packed_w)
                quant_layer.register_buffer("scales", s)
                if zp is not None:
                    quant_layer.register_buffer("zero_points", zp)

        elif scheme_lower == "int8":
            if symmetric:
                q_w, s = quantize_absmax(w, num_bits=8, dim=1)
                zp = None
            else:
                q_w, s, zp = quantize_affine(w, num_bits=8, dim=1)
            quant_layer.register_buffer("packed_weight", q_w.to(torch.int8))
            quant_layer.register_buffer("scales", s)
            if zp is not None:
                quant_layer.register_buffer("zero_points", zp)

        elif scheme_lower == "ternary":
            # BitNet ternary scaling: gamma = mean(|W|)
            gamma = w.abs().mean().clamp(min=1e-8)
            scaled_w = w / gamma
            ternary_w = torch.clamp(torch.round(scaled_w), -1.0, 1.0)
            packed_w, _ = pack_ternary(ternary_w, mode="2bit", dim=-1)
            quant_layer.register_buffer("packed_weight", packed_w)
            quant_layer.register_buffer("scales", gamma.view(1))

        elif scheme_lower in ("sparse24", "2:4"):
            condensed, metadata = prune_and_compress_2_4(w)
            quant_layer.register_buffer("condensed_weight", condensed)
            quant_layer.register_buffer("packed_metadata", metadata)

        return quant_layer

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize packed buffers back to a standard floating-point weight matrix."""
        if self.scheme == "int4":
            is_signed = self.zero_points is None
            if self.group_size > 0:
                # Group-wise reconstruction
                unpacked_q = unpack_int4(
                    self.packed_weight, original_len=self.group_size, signed=is_signed, dim=-1
                )
                n_elements = self.out_features * self.in_features
                remainder = n_elements % self.group_size
                pad_len = (self.group_size - remainder) % self.group_size
                meta = {
                    "original_shape": (self.out_features, self.in_features),
                    "n_elements": n_elements,
                    "pad_len": pad_len,
                    "group_size": self.group_size,
                    "symmetric": is_signed,
                    "dtype": torch.float32,
                    "device": self.packed_weight.device,
                }
                zp = self.zero_points if self.zero_points is not None else None
                return dequantize_blockwise(unpacked_q, self.scales, zp, meta)
            else:
                # Per-channel reconstruction
                unpacked_q = unpack_int4(
                    self.packed_weight, original_len=self.in_features, signed=is_signed, dim=-1
                )
                if self.zero_points is not None:
                    return dequantize_affine(unpacked_q, self.scales, self.zero_points)
                else:
                    return dequantize_absmax(unpacked_q, self.scales)

        elif self.scheme == "int8":
            if self.zero_points is not None:
                return dequantize_affine(self.packed_weight, self.scales, self.zero_points)
            else:
                return dequantize_absmax(self.packed_weight, self.scales)

        elif self.scheme == "ternary":
            unpacked_t = unpack_ternary(
                self.packed_weight, original_len=self.in_features, mode="2bit", dim=-1
            )
            return unpacked_t.to(torch.float32) * self.scales

        elif self.scheme in ("sparse24", "2:4"):
            return decompress_2_4(
                self.condensed_weight,
                self.packed_metadata,
                original_shape=(self.out_features, self.in_features),
            )

        raise RuntimeError(f"Unsupported scheme '{self.scheme}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic vector dequantization.

        Args:
            x: Input activation tensor with shape (..., in_features).

        Returns:
            Projected activation tensor with shape (..., out_features).
        """
        w_hat = self.dequantize_weight().to(x.dtype)
        return F.linear(x, w_hat, self.bias)

    def memory_footprint_bytes(self) -> dict[str, float]:
        """Compute exact memory footprint in bytes and compression factor relative to FP16.

        Returns:
            Dictionary containing 'compressed_bytes', 'fp16_bytes', and 'compression_ratio'.
        """
        compressed_bytes = 0
        for buf in self.buffers():
            if buf is not None:
                compressed_bytes += buf.nelement() * buf.element_size()
        if self.bias is not None:
            compressed_bytes += self.bias.nelement() * self.bias.element_size()

        fp16_bytes = (self.in_features * self.out_features * 2) + (
            (self.out_features * 2) if self.bias is not None else 0
        )
        ratio = fp16_bytes / max(compressed_bytes, 1)

        return {
            "compressed_bytes": compressed_bytes,
            "fp16_bytes": fp16_bytes,
            "compression_ratio": ratio,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scheme={self.scheme}, group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )
