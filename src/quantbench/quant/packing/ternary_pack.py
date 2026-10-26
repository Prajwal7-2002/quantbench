"""BitNet 1.58b ternary bit-packing runtime for sub-byte quantization.

Packs ternary weights in {-1, 0, +1} into compact uint8 byte containers.
Supports two complementary packing representations:
1. '2bit' (Default, hardware SIMD-friendly):
   - Maps {-1 -> 0, 0 -> 1, +1 -> 2} (biased offset by +1).
   - Packs 4 ternary values into 1 byte (2 bits per value).
   - Enables fast parallel bitshift and masking instructions on CPU/GPU.
   - Delivers an 8x memory reduction over FP16 and 4x over INT8.

2. 'base3' (High-density Radix-3):
   - Because 3^5 = 243 < 256, packs 5 ternary values into 1 byte.
   - Theoretical information-theoretic density limit (~1.58 bits per weight).
"""

from __future__ import annotations

import torch


def pack_ternary_quad(v0: int, v1: int, v2: int, v3: int) -> int:
    """Pack four ternary values {-1, 0, +1} into a single 8-bit unsigned byte.

    Uses biased representation: u = v + 1 in {0, 1, 2}.
    """
    for v in (v0, v1, v2, v3):
        if v not in (-1, 0, 1):
            raise ValueError(f"Ternary values must be in {{-1, 0, 1}}, got {v}.")
    u0 = v0 + 1
    u1 = v1 + 1
    u2 = v2 + 1
    u3 = v3 + 1
    return (u0 & 0x03) | ((u1 & 0x03) << 2) | ((u2 & 0x03) << 4) | ((u3 & 0x03) << 6)


def unpack_ternary_quad(byte: int) -> tuple[int, int, int, int]:
    """Unpack a single 8-bit unsigned byte into four ternary values {-1, 0, +1}."""
    u0 = byte & 0x03
    u1 = (byte >> 2) & 0x03
    u2 = (byte >> 4) & 0x03
    u3 = (byte >> 6) & 0x03
    return u0 - 1, u1 - 1, u2 - 1, u3 - 1


def pack_ternary(
    tensor: torch.Tensor,
    mode: str = "2bit",
    dim: int = -1,
) -> tuple[torch.Tensor, int]:
    """Vectorized packing of ternary {-1, 0, +1} weights into a uint8 tensor.

    Args:
        tensor: Integer or float tensor containing ternary values {-1, 0, 1}.
        mode: Packing mode:
            - '2bit': 4 ternary weights per byte (standard SIMD bitshift).
            - 'base3': 5 ternary weights per byte (Radix-3 maximal density).
        dim: Dimension along which to pack. Defaults to last dimension (-1).

    Returns:
        Tuple of (packed_uint8_tensor, original_dim_length).
    """
    if mode not in ("2bit", "base3"):
        raise ValueError(f"Unknown packing mode '{mode}'. Choose '2bit' or 'base3'.")

    t_int = tensor.round().to(torch.int32)
    valid_mask = (t_int == -1) | (t_int == 0) | (t_int == 1)
    if not valid_mask.all():
        raise ValueError("Input tensor contains values outside the ternary set {-1, 0, 1}.")

    dim = dim if dim >= 0 else tensor.ndim + dim
    orig_len = tensor.shape[dim]

    if dim != tensor.ndim - 1:
        t_int = t_int.transpose(dim, -1)

    # Offset to non-negative range: {-1, 0, 1} -> {0, 1, 2}
    u = t_int + 1

    if mode == "2bit":
        chunk_size = 4
        rem = orig_len % chunk_size
        if rem != 0:
            pad_len = chunk_size - rem
            pad_shape = list(u.shape[:-1]) + [pad_len]
            # Pad with 1 (which maps to ternary 0 after offset)
            pad_tensor = torch.ones(pad_shape, dtype=u.dtype, device=u.device)
            u = torch.cat([u, pad_tensor], dim=-1)

        v0 = u[..., 0::4]
        v1 = u[..., 1::4]
        v2 = u[..., 2::4]
        v3 = u[..., 3::4]

        packed = (v0 & 0x03) | ((v1 & 0x03) << 2) | ((v2 & 0x03) << 4) | ((v3 & 0x03) << 6)
        packed_uint8 = packed.to(torch.uint8)

    else:  # base3: 5 weights per byte
        chunk_size = 5
        rem = orig_len % chunk_size
        if rem != 0:
            pad_len = chunk_size - rem
            pad_shape = list(u.shape[:-1]) + [pad_len]
            pad_tensor = torch.ones(pad_shape, dtype=u.dtype, device=u.device)
            u = torch.cat([u, pad_tensor], dim=-1)

        v0 = u[..., 0::5]
        v1 = u[..., 1::5]
        v2 = u[..., 2::5]
        v3 = u[..., 3::5]
        v4 = u[..., 4::5]

        packed = v0 + (3 * v1) + (9 * v2) + (27 * v3) + (81 * v4)
        packed_uint8 = packed.to(torch.uint8)

    if dim != tensor.ndim - 1:
        packed_uint8 = packed_uint8.transpose(dim, -1)

    return packed_uint8, orig_len


def unpack_ternary(
    packed: torch.Tensor,
    original_len: int | None = None,
    mode: str = "2bit",
    dim: int = -1,
) -> torch.Tensor:
    """Vectorized unpacking of packed uint8 tensor back into ternary {-1, 0, +1} values.

    Args:
        packed: Tensor of dtype torch.uint8 containing packed ternary values.
        original_len: Optional original dimension length to trim padding.
        mode: Packing mode used ('2bit' or 'base3').
        dim: Dimension along which elements were packed.

    Returns:
        Unpacked tensor of dtype torch.int8 containing values strictly in {-1, 0, 1}.
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"Packed tensor must have dtype torch.uint8, got {packed.dtype}.")

    dim = dim if dim >= 0 else packed.ndim + dim
    if dim != packed.ndim - 1:
        packed = packed.transpose(dim, -1)

    p32 = packed.to(torch.int32)

    if mode == "2bit":
        u0 = p32 & 0x03
        u1 = (p32 >> 2) & 0x03
        u2 = (p32 >> 4) & 0x03
        u3 = (p32 >> 6) & 0x03

        # Subtract offset: {0, 1, 2} -> {-1, 0, 1}
        v0 = u0 - 1
        v1 = u1 - 1
        v2 = u2 - 1
        v3 = u3 - 1

        interleaved = torch.stack([v0, v1, v2, v3], dim=-1)
        flat_shape = list(interleaved.shape[:-2]) + [interleaved.shape[-2] * 4]
        unpacked = interleaved.view(flat_shape).to(torch.int8)

    elif mode == "base3":
        # Extract via radix-3 integer division and modulo
        u0 = p32 % 3
        rem1 = p32 // 3
        u1 = rem1 % 3
        rem2 = rem1 // 3
        u2 = rem2 % 3
        rem3 = rem2 // 3
        u3 = rem3 % 3
        u4 = rem3 // 3

        v0 = u0 - 1
        v1 = u1 - 1
        v2 = u2 - 1
        v3 = u3 - 1
        v4 = u4 - 1

        interleaved = torch.stack([v0, v1, v2, v3, v4], dim=-1)
        flat_shape = list(interleaved.shape[:-2]) + [interleaved.shape[-2] * 5]
        unpacked = interleaved.view(flat_shape).to(torch.int8)

    else:
        raise ValueError(f"Unknown packing mode '{mode}'. Choose '2bit' or 'base3'.")

    if original_len is not None:
        unpacked = unpacked[..., :original_len]

    if dim != packed.ndim - 1:
        unpacked = unpacked.transpose(dim, -1)

    return unpacked
