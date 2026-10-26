"""2:4 Structured Sparsity metadata bit-packing and matrix compression runtime.

Implements the NVIDIA Ampere/Ada/Hopper Sparse Tensor Core data layout:
- Invariant: In every consecutive chunk of 4 values along the row dimension, exactly 2
  values must be retained (50% sparsity).
- Retained values are stored in a dense condensed matrix of shape (M, N / 2).
- Column position indices (0, 1, 2, or 3) require log2(4) = 2 bits each.
- Two 2-bit indices for a 4-element block fit in 4 bits (1 nibble).
- Two consecutive 4-element blocks (8 dense elements, 4 retained non-zeros) pack into
  a single uint8 byte!

Byte layout for two 4-element blocks (A and B):
- Bits [0:1]: Block A, first non-zero index (0..3)
- Bits [2:3]: Block A, second non-zero index (0..3)
- Bits [4:5]: Block B, first non-zero index (0..3)
- Bits [6:7]: Block B, second non-zero index (0..3)
"""

from __future__ import annotations

import torch


def pack_2_4_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit position indices for 2:4 structured sparsity into a uint8 tensor.

    Args:
        indices: Tensor of shape (..., K) where K is a multiple of 4, containing
            position indices in {0, 1, 2, 3}. Every pair (2k, 2k+1) represents
            the two non-zero positions in a 4-element block.

    Returns:
        Packed uint8 tensor of shape (..., K // 4).
    """
    if indices.shape[-1] % 4 != 0:
        raise ValueError(
            f"Last dimension must be a multiple of 4 (representing pairs of 4-element blocks), "
            f"got {indices.shape[-1]}."
        )

    i32 = indices.to(torch.int32)
    if (i32 < 0).any() or (i32 > 3).any():
        raise ValueError("2:4 sparsity position indices must be strictly within {0, 1, 2, 3}.")

    # Each byte holds 4 indices: (idx0, idx1) from Block A, and (idx2, idx3) from Block B
    idx0 = i32[..., 0::4]
    idx1 = i32[..., 1::4]
    idx2 = i32[..., 2::4]
    idx3 = i32[..., 3::4]

    packed = (
        (idx0 & 0x03)
        | ((idx1 & 0x03) << 2)
        | ((idx2 & 0x03) << 4)
        | ((idx3 & 0x03) << 6)
    )
    return packed.to(torch.uint8)


def unpack_2_4_indices(packed: torch.Tensor) -> torch.Tensor:
    """Unpack packed uint8 metadata tensor back into 2-bit position indices.

    Args:
        packed: Tensor of dtype torch.uint8 and shape (..., M).

    Returns:
        Tensor of dtype torch.int8 and shape (..., M * 4) containing indices in {0, 1, 2, 3}.
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"Packed tensor must have dtype torch.uint8, got {packed.dtype}.")

    p32 = packed.to(torch.int32)
    idx0 = p32 & 0x03
    idx1 = (p32 >> 2) & 0x03
    idx2 = (p32 >> 4) & 0x03
    idx3 = (p32 >> 6) & 0x03

    interleaved = torch.stack([idx0, idx1, idx2, idx3], dim=-1)
    flat_shape = list(interleaved.shape[:-2]) + [interleaved.shape[-2] * 4]
    return interleaved.view(flat_shape).to(torch.int8)


def prune_and_compress_2_4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune a dense matrix to 2:4 structured sparsity and pack its values and metadata.

    For every block of 4 elements along the last dimension:
    1. Finds the 2 elements with largest absolute magnitude.
    2. Stores their values in condensed_values of shape (..., N // 2).
    3. Encodes their original within-block positions into packed_metadata of shape (..., N // 8).

    Args:
        weight: Dense weight tensor with last dimension divisible by 8.

    Returns:
        Tuple of (condensed_values, packed_metadata).
        condensed_values has shape (..., N // 2) and same dtype as input.
        packed_metadata has shape (..., N // 8) and dtype torch.uint8.
    """
    orig_shape = weight.shape
    n = orig_shape[-1]
    if n % 8 != 0:
        raise ValueError(f"Last dimension must be divisible by 8 for 2:4 byte alignment, got {n}.")

    # Reshape into blocks of 4
    num_blocks = n // 4
    blocks = weight.view(*orig_shape[:-1], num_blocks, 4)

    # Find top 2 values by absolute magnitude in each block of 4
    abs_blocks = blocks.abs()
    # topk returns values and indices sorted descending
    _, top2_indices = torch.topk(abs_blocks, k=2, dim=-1, largest=True, sorted=True)

    # Sort indices ascending to preserve relative left-to-right order
    sorted_indices, _ = torch.sort(top2_indices, dim=-1)

    # Gather the top 2 values
    condensed = torch.gather(blocks, dim=-1, index=sorted_indices)

    # Flatten condensed values to (..., N // 2)
    condensed_shape = list(orig_shape[:-1]) + [n // 2]
    condensed_values = condensed.view(condensed_shape)

    # Flatten indices to (..., N // 2) and pack into uint8 bytes
    flat_indices = sorted_indices.view(condensed_shape)
    packed_metadata = pack_2_4_indices(flat_indices)

    return condensed_values, packed_metadata


def decompress_2_4(
    condensed_values: torch.Tensor,
    packed_metadata: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
) -> torch.Tensor:
    """Decompress condensed values and packed metadata back to a sparse 2:4 tensor.

    Args:
        condensed_values: Tensor of shape (..., N // 2).
        packed_metadata: Tensor of shape (..., N // 8) and dtype torch.uint8.
        original_shape: Shape of original dense tensor to reconstruct (..., N).

    Returns:
        Reconstructed tensor with exact 2:4 sparsity (exactly 2 non-zeros per 4 elements).
    """
    n = original_shape[-1]
    if n % 8 != 0:
        raise ValueError(f"Original shape last dimension must be divisible by 8, got {n}.")

    num_blocks = n // 4
    unpacked_indices = unpack_2_4_indices(packed_metadata)

    # Reshape condensed and indices to (..., num_blocks, 2)
    val_blocks = condensed_values.view(*original_shape[:-1], num_blocks, 2)
    idx_blocks = unpacked_indices.view(*original_shape[:-1], num_blocks, 2).to(torch.int64)

    # Allocate zero tensor for blocks (..., num_blocks, 4)
    reconstructed_blocks = torch.zeros(
        *original_shape[:-1],
        num_blocks,
        4,
        dtype=condensed_values.dtype,
        device=condensed_values.device,
    )

    # Scatter condensed values back into their exact positions
    reconstructed_blocks.scatter_(dim=-1, index=idx_blocks, src=val_blocks)

    return reconstructed_blocks.view(original_shape)
