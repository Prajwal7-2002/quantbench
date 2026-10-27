"""Exhaustive unit tests for integer bit-packing runtimes.

Tests all 256 byte permutations, sign extensions, odd shape padding,
BitNet 1.58b ternary representations, and 2:4 structured sparsity compression.
"""

import itertools

import pytest
import torch

from quantbench.quant.packing import (
    decompress_2_4,
    pack_2_4_indices,
    pack_int4,
    pack_ternary,
    prune_and_compress_2_4,
    unpack_2_4_indices,
    unpack_int4,
    unpack_ternary,
)
from quantbench.quant.packing.int4_pack import pack_nibbles, unpack_nibbles
from quantbench.quant.packing.ternary_pack import pack_ternary_quad, unpack_ternary_quad


class TestInt4Packing:
    """Test suite for INT4 sub-byte packing and sign extension."""

    def test_all_256_byte_scalar_roundtrip(self) -> None:
        """Exhaustively verify all 256 uint8 byte combinations for unsigned and signed INT4."""
        for byte_val in range(256):
            # Unsigned roundtrip
            low_u, high_u = unpack_nibbles(byte_val, signed=False)
            assert 0 <= low_u <= 15
            assert 0 <= high_u <= 15
            repacked_u = pack_nibbles(low_u, high_u)
            assert repacked_u == byte_val

            # Signed roundtrip
            low_s, high_s = unpack_nibbles(byte_val, signed=True)
            assert -8 <= low_s <= 7
            assert -8 <= high_s <= 7
            repacked_s = pack_nibbles(low_s, high_s)
            assert repacked_s == byte_val

    def test_unsigned_tensor_roundtrip_multi_dim(self) -> None:
        """Test vectorized packing of unsigned INT4 tensors across 1D, 2D, and 3D shapes."""
        shapes = [(16,), (8, 32), (2, 4, 64)]
        for shape in shapes:
            orig = torch.randint(0, 16, shape, dtype=torch.int32)
            packed, orig_len = pack_int4(orig, signed=False)

            assert packed.dtype == torch.uint8
            assert packed.shape[-1] == shape[-1] // 2

            unpacked = unpack_int4(packed, original_len=orig_len, signed=False)
            assert unpacked.dtype == torch.uint8
            assert torch.equal(orig.to(torch.uint8), unpacked)

    def test_signed_tensor_roundtrip_boundaries(self) -> None:
        """Test signed two's complement packing including extreme values (-8, -1, 0, 7)."""
        boundaries = torch.tensor([-8, -7, -1, 0, 1, 6, 7, -8], dtype=torch.int32)
        packed, orig_len = pack_int4(boundaries, signed=True)

        assert packed.shape[-1] == 4
        unpacked = unpack_int4(packed, original_len=orig_len, signed=True)
        assert unpacked.dtype == torch.int8
        assert torch.equal(boundaries.to(torch.int8), unpacked)

    def test_odd_dimension_padding_and_truncation(self) -> None:
        """Verify automatic padding and truncation for odd-dimension tensors."""
        for odd_len in [3, 7, 15, 33, 101]:
            orig = torch.randint(0, 16, (odd_len,), dtype=torch.int32)
            packed, orig_len = pack_int4(orig, signed=False)

            expected_packed_len = (odd_len + 1) // 2
            assert packed.shape[0] == expected_packed_len

            unpacked = unpack_int4(packed, original_len=orig_len, signed=False)
            assert unpacked.shape[0] == odd_len
            assert torch.equal(orig.to(torch.uint8), unpacked)

    def test_bounds_validation(self) -> None:
        """Ensure out-of-bounds inputs raise ValueError."""
        with pytest.raises(ValueError, match="strictly within range \\[0, 15\\]"):
            pack_int4(torch.tensor([0, 16]), signed=False)

        with pytest.raises(ValueError, match="strictly within range \\[0, 15\\]"):
            pack_int4(torch.tensor([-1, 5]), signed=False)

        with pytest.raises(ValueError, match="strictly within range \\[-8, 7\\]"):
            pack_int4(torch.tensor([0, 8]), signed=True)

        with pytest.raises(ValueError, match="strictly within range \\[-8, 7\\]"):
            pack_int4(torch.tensor([-9, 0]), signed=True)


class TestTernaryPacking:
    """Test suite for BitNet 1.58b ternary packing."""

    def test_all_81_quad_permutations(self) -> None:
        """Exhaustively verify all 3^4 = 81 permutations of four ternary values."""
        for quad in itertools.product([-1, 0, 1], repeat=4):
            byte_val = pack_ternary_quad(*quad)
            assert 0 <= byte_val <= 255
            unpacked = unpack_ternary_quad(byte_val)
            assert unpacked == quad

    def test_ternary_2bit_tensor_roundtrip(self) -> None:
        """Test vectorized 2-bit packing (4 ternary values per byte) on multi-dim tensors."""
        shapes = [(16,), (8, 32), (2, 4, 64)]
        for shape in shapes:
            choices = torch.tensor([-1, 0, 1])
            orig = choices[torch.randint(0, 3, shape)]

            packed, orig_len = pack_ternary(orig, mode="2bit")
            assert packed.dtype == torch.uint8
            assert packed.shape[-1] == shape[-1] // 4

            unpacked = unpack_ternary(packed, original_len=orig_len, mode="2bit")
            assert unpacked.dtype == torch.int8
            assert torch.equal(orig.to(torch.int8), unpacked)

    def test_ternary_base3_tensor_roundtrip(self) -> None:
        """Test high-density Radix-3 base3 packing (5 ternary values per byte)."""
        shapes = [(20,), (5, 50)]
        for shape in shapes:
            choices = torch.tensor([-1, 0, 1])
            orig = choices[torch.randint(0, 3, shape)]

            packed, orig_len = pack_ternary(orig, mode="base3")
            assert packed.dtype == torch.uint8
            assert packed.shape[-1] == shape[-1] // 5

            unpacked = unpack_ternary(packed, original_len=orig_len, mode="base3")
            assert unpacked.dtype == torch.int8
            assert torch.equal(orig.to(torch.int8), unpacked)

    def test_ternary_non_divisible_lengths(self) -> None:
        """Test ternary packing with lengths not evenly divisible by 4 or 5."""
        choices = torch.tensor([-1, 0, 1])
        for length in [7, 13, 27]:
            orig = choices[torch.randint(0, 3, (length,))]

            # 2-bit mode
            packed_2b, len_2b = pack_ternary(orig, mode="2bit")
            unpacked_2b = unpack_ternary(packed_2b, original_len=len_2b, mode="2bit")
            assert torch.equal(orig.to(torch.int8), unpacked_2b)

            # Base-3 mode
            packed_b3, len_b3 = pack_ternary(orig, mode="base3")
            unpacked_b3 = unpack_ternary(packed_b3, original_len=len_b3, mode="base3")
            assert torch.equal(orig.to(torch.int8), unpacked_b3)

    def test_ternary_invalid_input(self) -> None:
        """Verify invalid values outside {-1, 0, 1} are rejected."""
        with pytest.raises(ValueError, match="outside the ternary set"):
            pack_ternary(torch.tensor([-2, 0, 1]))


class TestSparse24Packing:
    """Test suite for 2:4 structured sparsity pruning and metadata bit-packing."""

    def test_index_packing_roundtrip(self) -> None:
        """Test packing and unpacking of 2-bit column position indices."""
        indices = torch.tensor(
            [
                [0, 2, 1, 3],
                [1, 2, 0, 3],
            ],
            dtype=torch.int32,
        )

        packed = pack_2_4_indices(indices)
        assert packed.dtype == torch.uint8
        assert packed.shape == (2, 1)

        unpacked = unpack_2_4_indices(packed)
        assert torch.equal(indices.to(torch.int8), unpacked)

    def test_prune_and_compress_2_4_exactness(self) -> None:
        """Verify 2:4 pruning preserves 50% largest values and exact positions."""
        torch.manual_seed(42)
        # 16 rows, 32 columns
        weight = torch.randn(16, 32, dtype=torch.float32)

        condensed_values, packed_metadata = prune_and_compress_2_4(weight)

        # Value tensor should have exactly half columns
        assert condensed_values.shape == (16, 16)
        assert condensed_values.dtype == torch.float32

        # Metadata should have N // 8 columns (1 byte per 8 original values)
        assert packed_metadata.shape == (16, 4)
        assert packed_metadata.dtype == torch.uint8

        # Decompress back to original shape
        reconstructed = decompress_2_4(
            condensed_values, packed_metadata, original_shape=weight.shape
        )
        assert reconstructed.shape == weight.shape

        # Verify 2:4 sparsity structural invariant:
        # In EVERY block of 4 elements, exactly 2 must be non-zero and 2 must be zero
        blocks = reconstructed.view(16, 8, 4)
        non_zero_counts = (blocks != 0.0).sum(dim=-1)
        assert (non_zero_counts == 2).all(), "Every 4-element block must have exactly 2 non-zeros!"

        # Verify retained non-zero values match the top 2 magnitudes from original weight
        orig_blocks = weight.view(16, 8, 4)
        top2_orig_vals, _ = torch.topk(orig_blocks.abs(), k=2, dim=-1)
        recon_vals = blocks[blocks != 0.0].view(16, 8, 2).abs()
        # Sort both to compare values regardless of column order
        sorted_top2, _ = torch.sort(top2_orig_vals, dim=-1)
        sorted_recon, _ = torch.sort(recon_vals, dim=-1)
        assert torch.allclose(sorted_top2, sorted_recon, atol=1e-6)
