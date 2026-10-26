"""Vectorized INT4 bit-packing runtime for sub-byte quantization.

Packs two 4-bit values (either unsigned [0, 15] or signed [-8, 7]) into a single uint8
byte container, delivering a 4x memory footprint reduction over FP16 and 2x over INT8.

Bit layout within each packed uint8 byte:
- Bits [0:3] (Low nibble): Even element index (2k)
- Bits [4:7] (High nibble): Odd element index (2k + 1)

Mathematical mapping:
    byte = (v_even & 0x0F) | ((v_odd & 0x0F) << 4)
"""

from __future__ import annotations

import torch


def pack_nibbles(low: int, high: int) -> int:
    """Pack two 4-bit integer scalars into a single 8-bit unsigned byte.

    Args:
        low: Integer value for low nibble (bits 0..3), range [0, 15] or [-8, 7].
        high: Integer value for high nibble (bits 4..7), range [0, 15] or [-8, 7].

    Returns:
        Unsigned 8-bit integer in range [0, 255].
    """
    u_low = low & 0x0F
    u_high = high & 0x0F
    return (u_low) | (u_high << 4)


def unpack_nibbles(byte: int, signed: bool = False) -> tuple[int, int]:
    """Unpack a single 8-bit unsigned byte into two 4-bit integer scalars.

    Args:
        byte: Unsigned 8-bit integer in range [0, 255].
        signed: If True, sign-extends 4-bit values into signed range [-8, 7].
            If False, returns unsigned values in range [0, 15].

    Returns:
        Tuple of (low_nibble, high_nibble).
    """
    low = byte & 0x0F
    high = (byte >> 4) & 0x0F
    if signed:
        if low >= 8:
            low -= 16
        if high >= 8:
            high -= 16
    return low, high


def pack_int4(
    tensor: torch.Tensor,
    signed: bool = False,
    dim: int = -1,
) -> tuple[torch.Tensor, int]:
    """Vectorized bit-shift packing of INT4 values into a uint8 tensor.

    Args:
        tensor: Integer PyTorch tensor containing 4-bit values.
            If signed=False: values must be in [0, 15].
            If signed=True: values must be in [-8, 7].
        signed: Whether the input values are signed two's complement.
        dim: Dimension along which to pack adjacent pairs. Defaults to last dimension (-1).

    Returns:
        Tuple of (packed_uint8_tensor, original_dim_length).
        The packed tensor will have size ceil(original_length / 2) along `dim`.
    """
    if not tensor.dtype.is_floating_point and tensor.dtype != torch.bool:
        t_int = tensor.to(torch.int32)
    else:
        t_int = tensor.round().to(torch.int32)

    # Validate bounds
    if signed:
        if (t_int < -8).any() or (t_int > 7).any():
            raise ValueError("Signed INT4 values must fall strictly within range [-8, 7].")
    else:
        if (t_int < 0).any() or (t_int > 15).any():
            raise ValueError("Unsigned INT4 values must fall strictly within range [0, 15].")

    # Normalize dim
    dim = dim if dim >= 0 else tensor.ndim + dim
    orig_len = tensor.shape[dim]

    # Convert to 4-bit unsigned representation
    u4 = t_int & 0x0F

    # Pad by 1 if odd length along dim
    if orig_len % 2 != 0:
        pad_shape = list(u4.shape)
        pad_shape[dim] = 1
        pad_tensor = torch.zeros(pad_shape, dtype=u4.dtype, device=u4.device)
        u4 = torch.cat([u4, pad_tensor], dim=dim)

    # Slice into even and odd indices
    # We move target dim to last position for easy contiguous slicing
    if dim != u4.ndim - 1:
        u4 = u4.transpose(dim, -1)

    low = u4[..., 0::2]
    high = u4[..., 1::2]

    packed = (low & 0x0F) | ((high & 0x0F) << 4)
    packed_uint8 = packed.to(torch.uint8)

    if dim != tensor.ndim - 1:
        packed_uint8 = packed_uint8.transpose(dim, -1)

    return packed_uint8, orig_len


def unpack_int4(
    packed: torch.Tensor,
    original_len: int | None = None,
    signed: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """Vectorized unpacking of a uint8 packed tensor back into INT4 values.

    Args:
        packed: Tensor of dtype torch.uint8 containing packed pairs.
        original_len: Optional original dimension length before packing.
            Used to remove padding if original length was odd.
        signed: If True, sign-extends values to signed range [-8, 7].
            If False, returns unsigned values in [0, 15].
        dim: Dimension along which elements were packed. Defaults to -1.

    Returns:
        Unpacked tensor with dtype torch.int8 (if signed) or torch.uint8 (if unsigned),
        restored to the original length along `dim`.
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"Packed tensor must have dtype torch.uint8, got {packed.dtype}.")

    dim = dim if dim >= 0 else packed.ndim + dim
    if dim != packed.ndim - 1:
        packed = packed.transpose(dim, -1)

    # Extract low nibble and high nibble
    p32 = packed.to(torch.int32)
    low = p32 & 0x0F
    high = (p32 >> 4) & 0x0F

    # Interleave low and high along a new sub-dimension
    # Shape becomes (*batch_dims, packed_len, 2) -> flatten to (*batch_dims, packed_len * 2)
    interleaved = torch.stack([low, high], dim=-1)
    flat_shape = list(interleaved.shape[:-2]) + [interleaved.shape[-2] * 2]
    unpacked = interleaved.view(flat_shape)

    # Truncate padding if original_len is specified
    if original_len is not None:
        unpacked = unpacked[..., :original_len]

    if signed:
        # Values in 8..15 have the sign bit set; subtract 16
        unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
        unpacked = unpacked.to(torch.int8)
    else:
        unpacked = unpacked.to(torch.uint8)

    if dim != packed.ndim - 1:
        unpacked = unpacked.transpose(dim, -1)

    return unpacked
